"""Volatility surface: from a raw option chain to a queryable, arbitrage-checked surface.

Pipeline per expiry:
  1. Forward F and discount D from put-call parity on strikes quoted both ways
     (or D from a supplied rate, with only F implied from parity).
  2. Implied vols from out-of-the-money mids (puts below F, calls at or above F),
     which are more liquid and carry less early-exercise premium than ITM quotes.
  3. Raw SVI fit to total variance w = iv^2 T against log-moneyness k = ln(K/F),
     with a penalty that keeps the slice free of butterfly arbitrage.

Slices are fitted independently. Calendar arbitrage is checked, not imposed:
constraining each expiry against the previous one's fit lets a single bad wing
cascade through every later expiry. Across expiries, total variance is
interpolated linearly in T at fixed k, which keeps the surface
calendar-arbitrage-free whenever the fitted slices are. Before
the first expiry and after the last, implied vol is held constant in T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .arbitrage import ButterflyCheck, CalendarViolation, butterfly_check, calendar_violations
from .black_scholes import black_price
from .implied_vol import implied_vol
from .parity import implied_forward
from .svi import SVIParams, fit_svi

REQUIRED_COLUMNS = {"T", "strike", "kind", "bid", "ask"}
CHECK_MARGIN = 0.25  # check grid extends this fraction of the quoted k-range past each end
CHECK_POINTS = 201


@dataclass(frozen=True)
class Slice:
    T: float
    forward: float
    discount: float
    params: SVIParams
    rmse_vol: float  # RMSE of fitted vs market mid implied vol, in vol units
    quotes: pd.DataFrame = field(repr=False)  # the OTM quotes used in the fit

    @property
    def k_check(self) -> np.ndarray:
        """Log-moneyness grid on which this slice is kept arbitrage-free."""
        return _check_grid(self.quotes["k"])


@dataclass(frozen=True)
class ArbitrageReport:
    butterfly: dict[float, ButterflyCheck]
    calendar: list[CalendarViolation]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.butterfly.values()) and not self.calendar


def _check_grid(k: ArrayLike) -> np.ndarray:
    """The quoted k-range plus a margin at each end, where butterfly arbitrage is penalised and checked."""
    k = np.asarray(k, dtype=float)
    pad = CHECK_MARGIN * (k.max() - k.min())
    return np.linspace(k.min() - pad, k.max() + pad, CHECK_POINTS)


def _overlap(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Grid over the k-range two slices were both quoted on. Outside it at least one
    slice is extrapolation, so comparing their variances says nothing about the market."""
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    return np.linspace(lo, hi, CHECK_POINTS) if hi > lo else np.empty(0)


def _prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    q = quotes[(quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])].copy()
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    q["kind"] = q["kind"].str.lower().str[0]
    return q


def _fit_slice(
    T: float,
    quotes: pd.DataFrame,
    rate: float | None,
    n_parity: int,
    min_quotes: int,
    min_price: float,
    max_std_moneyness: float,
) -> Slice | None:
    q = _prepare_quotes(quotes)
    both = q.pivot_table(index="strike", columns="kind", values="mid").dropna()
    if len(both) < 2 or not {"c", "p"} <= set(both.columns):
        return None
    D = None if rate is None else float(np.exp(-rate * T))
    F, D = implied_forward(both.index, both["c"], both["p"], n_nearest=n_parity, discount=D)

    is_otm = ((q["kind"] == "p") & (q["strike"] < F)) | ((q["kind"] == "c") & (q["strike"] >= F))
    otm = q[is_otm & (q["mid"] >= min_price)].copy()
    K, kind = otm["strike"], otm["kind"]
    otm["iv"] = implied_vol(otm["mid"], F, K, T, D, kind)
    otm["iv_bid"] = implied_vol(otm["bid"], F, K, T, D, kind)
    otm["iv_ask"] = implied_vol(otm["ask"], F, K, T, D, kind)
    otm = otm.dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)
    if otm.empty:
        return None
    otm["k"] = np.log(otm["strike"] / F)

    # Keep strikes within max_std_moneyness ATM standard deviations of the forward.
    # Far-wing quotes a tick or two wide carry little information about vol and
    # can drag a five-parameter smile around.
    atm_vol = otm["iv"].iloc[int(np.argmin(np.abs(otm["k"].to_numpy())))]
    otm = otm[np.abs(otm["k"]) <= max_std_moneyness * atm_vol * np.sqrt(T)].reset_index(drop=True)
    if len(otm) < min_quotes:
        return None

    fit = fit_svi(otm["k"], otm["iv"] ** 2 * T, k_check=_check_grid(otm["k"]))
    fitted_iv = fit.params.implied_vol(otm["k"], T)
    rmse_vol = float(np.sqrt(np.mean((fitted_iv - otm["iv"]) ** 2)))
    return Slice(T, F, D, fit.params, rmse_vol, otm)


class VolSurface:
    def __init__(self, spot: float, slices: list[Slice]):
        if not slices:
            raise ValueError("a surface needs at least one expiry slice")
        self.spot = float(spot)
        self.slices = sorted(slices, key=lambda s: s.T)
        self._T = np.array([s.T for s in self.slices])
        # Carry and rate implied by each slice; interpolated (flat beyond the ends).
        self._carry = np.log(np.array([s.forward for s in self.slices]) / self.spot) / self._T
        self._rate = -np.log(np.array([s.discount for s in self.slices])) / self._T

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        spot: float,
        rate: float | None = None,
        n_parity: int = 8,
        min_quotes: int = 8,
        min_price: float = 0.0,
        max_std_moneyness: float = 6.0,
    ) -> "VolSurface":
        """Build a surface from a chain with columns T, strike, kind ('call'/'put'), bid, ask.

        ``rate`` (continuously compounded) fixes the discount factor per expiry;
        without it the discount factor is also estimated from put-call parity.
        The smile fit uses OTM quotes with a mid of at least ``min_price`` (a few
        ticks on real chains) and |ln(K/F)| within ``max_std_moneyness`` ATM
        standard deviations. Expiries without enough usable quotes are skipped.
        """
        missing = REQUIRED_COLUMNS - set(chain.columns)
        if missing:
            raise ValueError(f"chain is missing columns: {sorted(missing)}")
        slices = []
        for T, group in chain.groupby("T"):
            if T <= 0:
                continue
            s = _fit_slice(float(T), group, rate, n_parity, min_quotes, min_price, max_std_moneyness)
            if s is not None:
                slices.append(s)
        return cls(spot, slices)

    def forward(self, T: ArrayLike) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.spot * np.exp(np.interp(T, self._T, self._carry) * T)

    def discount(self, T: ArrayLike) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return np.exp(-np.interp(T, self._T, self._rate) * T)

    def total_variance_k(self, k: ArrayLike, T: float) -> np.ndarray:
        """Total variance at log-moneyness k and maturity T."""
        k = np.asarray(k, dtype=float)
        Ts = self._T
        if T <= Ts[0]:
            return self.slices[0].params.total_variance(k) * T / Ts[0]
        if T >= Ts[-1]:
            return self.slices[-1].params.total_variance(k) * T / Ts[-1]
        i = int(np.searchsorted(Ts, T)) - 1
        t1, t2 = Ts[i], Ts[i + 1]
        w1 = self.slices[i].params.total_variance(k)
        w2 = self.slices[i + 1].params.total_variance(k)
        return w1 + (T - t1) / (t2 - t1) * (w2 - w1)

    def implied_vol(self, K: ArrayLike, T: float) -> np.ndarray:
        k = np.log(np.asarray(K, dtype=float) / self.forward(T))
        return np.sqrt(np.maximum(self.total_variance_k(k, T), 0.0) / T)

    def price(self, K: ArrayLike, T: float, kind: ArrayLike = "call") -> np.ndarray:
        return black_price(self.forward(T), K, T, self.implied_vol(K, T), self.discount(T), kind)

    def check_arbitrage(self, k: ArrayLike | None = None) -> ArbitrageReport:
        """Butterfly check per slice and calendar check per adjacent pair of expiries.

        By default butterfly is checked on each slice's quoted k-range plus a
        margin (the grid the fit was constrained on), and calendar on the k-range
        both expiries were quoted on. Pass ``k`` to check everything on one grid.
        """
        butterfly = {s.T: butterfly_check(s.params, s.k_check if k is None else k) for s in self.slices}
        calendar: list[CalendarViolation] = []
        for short, long in zip(self.slices, self.slices[1:]):
            grid = _overlap(short.quotes["k"], long.quotes["k"]) if k is None else np.asarray(k, dtype=float)
            if grid.size:
                calendar += calendar_violations([(short.T, short.params), (long.T, long.params)], grid)
        return ArbitrageReport(butterfly, calendar)

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.slices:
            p = s.params
            rows.append(
                {
                    "T": s.T,
                    "forward": s.forward,
                    "discount": s.discount,
                    "atm_vol": float(p.implied_vol(0.0, s.T)),
                    "n_quotes": len(s.quotes),
                    "rmse_vol_bp": 1e4 * s.rmse_vol,
                    "a": p.a,
                    "b": p.b,
                    "rho": p.rho,
                    "m": p.m,
                    "sigma": p.sigma,
                }
            )
        return pd.DataFrame(rows)

"""Synthetic option chains priced off a known, arbitrage-free SSVI surface.

Because the true surface is known, a fit can be scored exactly: how well does
the pipeline recover the forward, the discount factor and the smile under
realistic quote noise and bid/ask spreads?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import black_price
from .svi import SVIParams, power_law_phi, ssvi_slice


@dataclass(frozen=True)
class SSVISurface:
    """SSVI with a decaying ATM vol term structure and power-law curvature."""

    vol_short: float = 0.24
    vol_long: float = 0.19
    decay: float = 0.5  # years
    rho: float = -0.65
    eta: float = 1.1
    gamma: float = 0.45

    def atm_vol(self, T: float) -> float:
        return self.vol_long + (self.vol_short - self.vol_long) * np.exp(-T / self.decay)

    def theta(self, T: float) -> float:
        return self.atm_vol(T) ** 2 * T

    def slice(self, T: float) -> SVIParams:
        th = self.theta(T)
        return ssvi_slice(th, self.rho, power_law_phi(th, self.eta, self.gamma))


def make_chain(
    spot: float = 100.0,
    r: float = 0.04,
    q: float = 0.015,
    expiries: tuple[float, ...] = (1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0),
    n_strikes: int = 31,
    noise_vol: float = 0.002,
    half_spread_vol: float = 0.004,
    tick: float = 0.01,
    surface: SSVISurface | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[float, SVIParams]]:
    """Return ``(chain, truth)``: a quote table and the true SVI slice per expiry.

    Mid implied vols are the true vols plus Gaussian noise (``noise_vol``), common
    to the call and put at each strike; bids
    and asks sit ``half_spread_vol`` either side in vol terms, are rounded to the
    tick, and quotes whose bid rounds to zero are dropped, as on a real screen.
    """
    surface = surface or SSVISurface()
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for T in expiries:
        F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
        params = surface.slice(T)
        truth[T] = params

        width = surface.atm_vol(T) * np.sqrt(T)
        strikes = np.unique(np.round(F * np.exp(np.linspace(-3.0, 2.0, n_strikes) * width), 1))
        # One noisy mid vol per strike, shared by the call and the put: arbitrage
        # keeps real call and put quotes consistent with put-call parity.
        iv_mid = params.implied_vol(np.log(strikes / F), T) + rng.normal(0.0, noise_vol, strikes.size)
        for kind in ("call", "put"):
            bid = black_price(F, strikes, T, np.maximum(iv_mid - half_spread_vol, 1e-4), D, kind)
            ask = black_price(F, strikes, T, iv_mid + half_spread_vol, D, kind)
            bid = np.floor(bid / tick) * tick
            ask = np.ceil(ask / tick) * tick
            for K, b, a in zip(strikes, bid, ask):
                if b > 0:
                    rows.append({"T": T, "strike": K, "kind": kind, "bid": round(b, 2), "ask": round(a, 2)})
    return pd.DataFrame(rows), truth

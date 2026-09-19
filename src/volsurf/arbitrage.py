"""Static no-arbitrage checks for smiles, surfaces and raw quotes.

* Butterfly arbitrage (per expiry): the risk-neutral density implied by the smile
  must be non-negative. For a smile given as total variance w(k) the density is
  proportional to Durrleman's function

      g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1/w + 1/4) + w'' / 2,

  so the smile is butterfly-free iff g(k) >= 0 everywhere.
* Calendar arbitrage (across expiries): at fixed log-moneyness, total variance
  must be non-decreasing in maturity.
* Quote-level checks need no model: call prices must be non-increasing and
  convex in strike, with slope no steeper than -D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .svi import SVIParams, durrleman_g

DEFAULT_K_GRID = np.linspace(-1.5, 1.5, 601)


@dataclass(frozen=True)
class ButterflyCheck:
    ok: bool
    min_g: float
    k_at_min: float


def butterfly_check(params: SVIParams, k: ArrayLike = DEFAULT_K_GRID, tol: float = 1e-10) -> ButterflyCheck:
    k = np.asarray(k, dtype=float)
    g = durrleman_g(params, k)
    i = int(np.argmin(g))
    return ButterflyCheck(ok=bool(g[i] >= -tol), min_g=float(g[i]), k_at_min=float(k[i]))


@dataclass(frozen=True)
class CalendarViolation:
    T_short: float
    T_long: float
    k: float
    excess_variance: float  # w(k, T_short) - w(k, T_long) > 0


def calendar_violations(
    slices: list[tuple[float, SVIParams]],
    k: ArrayLike = DEFAULT_K_GRID,
    tol: float = 1e-10,
) -> list[CalendarViolation]:
    """Worst violation per adjacent pair of expiries (empty list if calendar-free)."""
    k = np.asarray(k, dtype=float)
    ordered = sorted(slices, key=lambda s: s[0])
    out = []
    for (t1, p1), (t2, p2) in zip(ordered, ordered[1:]):
        excess = p1.total_variance(k) - p2.total_variance(k)
        i = int(np.argmax(excess))
        if excess[i] > tol:
            out.append(CalendarViolation(t1, t2, float(k[i]), float(excess[i])))
    return out


@dataclass(frozen=True)
class QuoteViolation:
    kind: str  # "monotonicity", "slope" or "convexity"
    strikes: tuple[float, ...]
    amount: float


def quote_violations(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    discount: float = 1.0,
    tol: float = 1e-8,
) -> list[QuoteViolation]:
    """Model-free arbitrage checks on one expiry's call prices.

    With strikes sorted ascending and s_i the slope between neighbouring strikes:
    monotonicity  C_i >= C_{i+1}          (a call spread cannot cost less than zero)
    slope         s_i >= -D               (a call spread cannot pay more than its width)
    convexity     s_i <= s_{i+1}          (a butterfly cannot cost less than zero)
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    slopes = np.diff(C) / np.diff(K)

    out = []
    for i, s in enumerate(slopes):
        pair = (float(K[i]), float(K[i + 1]))
        if s > tol:
            out.append(QuoteViolation("monotonicity", pair, float(s)))
        if s < -discount - tol:
            out.append(QuoteViolation("slope", pair, float(-discount - s)))
    for i in range(len(slopes) - 1):
        if slopes[i] > slopes[i + 1] + tol:
            triple = (float(K[i]), float(K[i + 1]), float(K[i + 2]))
            out.append(QuoteViolation("convexity", triple, float(slopes[i] - slopes[i + 1])))
    return out

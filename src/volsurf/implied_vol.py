"""Black implied volatility via a vectorised, safeguarded Newton iteration.

Every quote is first mapped to the out-of-the-money option at its strike with
put-call parity on undiscounted prices (c - p = F - K): calls for K >= F, puts
for K < F. OTM prices carry all of the time value, so solving on that side
avoids the cancellation error of backing a small time value out of a large
intrinsic value.

Each option keeps a bracket [lo, hi] that always contains the root: a Newton
step that leaves the bracket (or is not finite, e.g. when vega underflows deep
in the wings) is replaced by bisection. Newton gives quadratic convergence
near the root; the bracket guarantees convergence.

Volatility is identified only through time value. When the OTM-equivalent
price is too small to resolve in double precision (a deep ITM quote whose time
value is lost to cancellation, or a far-wing price near 1e-12 of the forward),
no vol can be recovered reliably and the solver returns NaN rather than a
confident wrong answer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from .black_scholes import is_call

SIGMA_MAX = 10.0  # 1000% vol: anything above is treated as unsolvable
RESOLUTION = 1e3 * np.finfo(float).eps  # smallest usable time value, relative to the prices involved
RESOLUTION = 1e3 * np.finfo(float).eps  # smallest usable time value, relative to the prices involved


def _otm_price_and_vega(F, K, T, sigma, otm_call):
    sqrt_t = np.sqrt(T)
    sd = sigma * sqrt_t
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    price = np.where(otm_call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))
    vega = F * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * sqrt_t
    return price, vega


def implied_vol(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    discount: ArrayLike = 1.0,
    kind: ArrayLike = "call",
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Implied Black volatility of European option prices.

    Returns NaN where no volatility reproduces the price: at or below discounted
    intrinsic value, at or above the upper bound (D * F for calls, D * K for
    puts), for non-positive expiries, or beyond ``SIGMA_MAX``; and where the time
    value is below double-precision resolution (see module docstring).
    """
    price, F, K, T, D = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, F, K, T, discount))
    )
    calls = np.broadcast_to(is_call(kind), price.shape)

    otm_call = K >= F
    u = price / D
    # Quotes already on the OTM side are used as-is; ITM quotes go through parity.
    v = np.where(calls == otm_call, u, np.where(calls, u - (F - K), u + (F - K)))
    upper = np.where(otm_call, F, K)
    resolvable = v > RESOLUTION * np.maximum(np.abs(u), F)
    valid = (T > 0) & resolvable & (v < upper)

    # Solve only the valid entries; placeholder inputs keep the maths finite.
    Tv = np.where(valid, T, 1.0)
    vv = np.where(valid, v, 0.5 * upper)
    lo = np.zeros_like(vv)
    hi = np.full_like(vv, SIGMA_MAX)

    # Start near the inflection point of the price in sigma, where Newton is
    # most robust: sigma0 = sqrt(2 |ln(F/K)| / T), floored at 20%.
    sigma = np.clip(np.sqrt(2.0 * np.abs(np.log(F / K)) / Tv), 0.2, SIGMA_MAX / 2)

    done = ~valid
    for _ in range(max_iter):
        model, vega = _otm_price_and_vega(F, K, Tv, sigma, otm_call)
        diff = model - vv
        # Option prices increase in sigma, so the sign of diff moves the bracket.
        hi = np.where(diff > 0, sigma, hi)
        lo = np.where(diff <= 0, sigma, lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = sigma - diff / vega
        bisect = ~np.isfinite(step) | (step <= lo) | (step >= hi)
        new = np.where(bisect, 0.5 * (lo + hi), step)
        converged = (np.abs(new - sigma) < tol) | (hi - lo < tol)
        sigma = np.where(done, sigma, new)
        done |= converged
        if done.all():
            break

    # A root pinned against the upper bracket means the price needs sigma > SIGMA_MAX.
    solvable = valid & (sigma < SIGMA_MAX - 1e-6)
    return np.where(solvable, sigma, np.nan)

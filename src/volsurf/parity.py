"""Implied forward and discount factor from put-call parity.

For European options on the same expiry, C(K) - P(K) = D (F - K) holds at every
strike. Regressing C - P on K therefore gives slope -D and intercept D * F,
which recovers the market-implied forward and discount factor without having to
assume a rate, a dividend yield, or a borrow cost.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

# Implied rates outside this band almost always mean the regression is unreliable
# (strikes too close together, stale or American quotes), not a real market rate.
PLAUSIBLE_RATE = (-0.02, 0.20)


def implied_forward(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    put_prices: ArrayLike,
    n_nearest: int | None = None,
    discount: float | None = None,
    T: float | None = None,
) -> tuple[float, float]:
    """Return ``(forward, discount)`` implied by put-call parity.

    Without ``discount``, both are estimated from an OLS fit of C - P on K.
    That needs strikes spread widely enough to pin down the slope. On real
    chains the near-the-money strikes are closely spaced, so it is more robust
    to take the discount factor from a rate curve and pass it in: the forward
    is then the median of K + (C - P) / D, one estimate per strike. Pass the
    expiry ``T`` to get a warning when the regression implies an implausible rate.

    ``n_nearest`` restricts the estimate to the strikes with the smallest
    |C - P|, i.e. those closest to the forward, where both quotes are most
    liquid and, for American options, carry the least early-exercise premium.
    """
    K = np.asarray(strikes, dtype=float)
    spread = np.asarray(call_prices, dtype=float) - np.asarray(put_prices, dtype=float)
    if n_nearest is not None:
        idx = np.argsort(np.abs(spread))[:n_nearest]
        K, spread = K[idx], spread[idx]

    if discount is not None:
        if K.size == 0:
            raise ValueError("need at least one strike quoted both ways")
        return float(np.median(K + spread / discount)), float(discount)

    if np.unique(K).size < 2:
        raise ValueError("need at least two distinct strikes to estimate a forward")
    slope, intercept = np.polyfit(K, spread, 1)
    discount = -slope
    if discount <= 0.0:
        raise ValueError(f"non-positive discount factor {discount:.4f}; check the quotes")
    if T is not None and T > 0:
        rate = -np.log(discount) / T
        if not PLAUSIBLE_RATE[0] <= rate <= PLAUSIBLE_RATE[1]:
            warnings.warn(
                f"put-call parity implies a {rate:.2%} rate at T={T:.3f} (D={discount:.4f}); "
                "pass a rate from the curve instead",
                stacklevel=2,
            )
    return intercept / discount, discount

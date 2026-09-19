"""Implied forward and discount factor from put-call parity.

For European options on the same expiry, C(K) - P(K) = D (F - K) holds at every
strike. Regressing C - P on K therefore gives slope -D and intercept D * F,
which recovers the market-implied forward and discount factor without having to
assume a rate, a dividend yield, or a borrow cost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def implied_forward(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    put_prices: ArrayLike,
    n_nearest: int | None = None,
    discount: float | None = None,
) -> tuple[float, float]:
    """Return ``(forward, discount)`` implied by put-call parity.

    Without ``discount``, both are estimated from an OLS fit of C - P on K.
    That needs strikes spread widely enough to pin down the slope. On real
    chains the near-the-money strikes are closely spaced, so it is more robust
    to take the discount factor from a rate curve and pass it in: the forward
    is then the median of K + (C - P) / D, one estimate per strike.

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
    if not 0.0 < discount <= 1.5:
        raise ValueError(f"implausible discount factor {discount:.4f}; check the quotes")
    return intercept / discount, discount

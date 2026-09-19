"""Black-Scholes-Merton pricing and Greeks.

All functions broadcast NumPy array inputs. Rates and the dividend yield are
continuously compounded and time is in years.

Two parameterisations are provided:

* ``black_price`` works on the forward ``F`` and discount factor ``D`` (Black-76).
  This is the natural form for fitting a volatility surface, because the forward
  and discount factor can be read directly off the option chain.
* ``bs_price`` / ``greeks`` work on spot ``S``, rate ``r`` and dividend yield ``q``,
  using ``F = S * exp((r - q) T)`` and ``D = exp(-r T)``.

Conventions: vega and rho are per unit change (not per 1%). Theta is the
derivative with respect to calendar time, per year, so a long option's theta is
usually negative.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def is_call(kind: ArrayLike) -> np.ndarray:
    """Map 'call'/'put' (or 'c'/'p', any case) to a boolean array."""
    k = np.char.lower(np.asarray(kind, dtype=str))
    calls = np.char.startswith(k, "c")
    if not np.all(calls | np.char.startswith(k, "p")):
        raise ValueError("kind must be 'call' or 'put' (or 'c' / 'p')")
    return calls


def _d1_d2(F, K, T, sigma):
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + 0.5 * sd
    return d1, d1 - sd, sd


def black_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    discount: ArrayLike = 1.0,
    kind: ArrayLike = "call",
) -> np.ndarray:
    """Black-76 price of a European option on a forward.

    call = D * (F N(d1) - K N(d2))
    put  = D * (K N(-d2) - F N(-d1))
    d1   = ln(F/K) / (sigma sqrt(T)) + sigma sqrt(T) / 2,   d2 = d1 - sigma sqrt(T)

    With zero total volatility the price collapses to discounted intrinsic value.
    """
    F, K, T, sigma, D = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (F, K, T, sigma, discount))
    )
    calls = np.broadcast_to(is_call(kind), F.shape)
    d1, d2, sd = _d1_d2(F, K, T, sigma)
    call = D * (F * ndtr(d1) - K * ndtr(d2))
    put = D * (K * ndtr(-d2) - F * ndtr(-d1))
    price = np.where(calls, call, put)
    intrinsic = D * np.where(calls, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(sd > 0, price, intrinsic)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    kind: ArrayLike = "call",
) -> np.ndarray:
    """Black-Scholes-Merton price from spot, rate and continuous dividend yield."""
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, q))
    F = S * np.exp((r - q) * T)
    return black_price(F, K, T, sigma, np.exp(-r * T), kind)


class Greeks(NamedTuple):
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    kind: ArrayLike = "call",
) -> Greeks:
    """Price and analytic Greeks under Black-Scholes-Merton (requires T > 0, sigma > 0).

    delta = e^{-qT} N(d1)                       (put: e^{-qT} (N(d1) - 1))
    gamma = e^{-qT} n(d1) / (S sigma sqrt(T))
    vega  = S e^{-qT} n(d1) sqrt(T)
    theta = -S e^{-qT} n(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
            (put: ... + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1))
    rho   = K T e^{-rT} N(d2)                   (put: -K T e^{-rT} N(-d2))
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )
    calls = np.broadcast_to(is_call(kind), S.shape)
    F = S * np.exp((r - q) * T)
    d1, d2, _ = _d1_d2(F, K, T, sigma)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf = _norm_pdf(d1)
    sqrt_t = np.sqrt(T)

    price = black_price(F, K, T, sigma, dr, np.where(calls, "call", "put"))
    delta = np.where(calls, dq * ndtr(d1), dq * (ndtr(d1) - 1.0))
    gamma = dq * pdf / (S * sigma * sqrt_t)
    vega = S * dq * pdf * sqrt_t
    decay = -S * dq * pdf * sigma / (2.0 * sqrt_t)
    theta = np.where(
        calls,
        decay - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1),
        decay + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1),
    )
    rho = np.where(calls, K * T * dr * ndtr(d2), -K * T * dr * ndtr(-d2))
    return Greeks(price, delta, gamma, vega, theta, rho)

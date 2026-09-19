"""Cox-Ross-Rubinstein binomial tree for European and American options.

u = exp(sigma sqrt(dt)), d = 1/u, and the risk-neutral up-probability is
p = (exp((r - q) dt) - d) / (u - d). Values are rolled back one time step at a
time as a NumPy vector; for American options each node takes the larger of the
continuation value and immediate exercise.
"""

from __future__ import annotations

import numpy as np

from .black_scholes import is_call

from .black_scholes import is_call


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    american: bool = False,
    steps: int = 500,
) -> float:
    call = bool(is_call(kind))
    if steps < 1 or T <= 0 or sigma <= 0:
        raise ValueError("need steps >= 1, T > 0 and sigma > 0")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0, 1); increase steps")
    disc = np.exp(-r * dt)
    sign = 1.0 if call else -1.0

    # Terminal spots, ordered from the most down-moves to the most up-moves.
    spots = S * u ** np.arange(-steps, steps + 1, 2, dtype=float)
    values = np.maximum(sign * (spots - K), 0.0)
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            spots = S * u ** np.arange(-n, n + 1, 2, dtype=float)
            values = np.maximum(values, sign * (spots - K))
    return float(values[0])

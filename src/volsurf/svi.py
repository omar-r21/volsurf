"""Raw SVI smile model (Gatheral, 2004) and its SSVI special case.

Raw SVI gives total implied variance w = sigma_imp^2 * T as a function of
log-moneyness k = ln(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Five parameters: ``a`` sets the level, ``b`` the wing slope, ``rho`` the skew,
``m`` shifts the smile horizontally and ``sigma`` controls ATM curvature.
Its wings are linear in k, consistent with Lee's moment formula, which is why
SVI extrapolates sensibly where polynomial fits in strike blow up.

SSVI (Gatheral & Jacquier, 2014) is a surface parameterisation that is free of
static arbitrage under simple parameter conditions; each SSVI slice is a raw
SVI slice, so it is used here to build arbitrage-free synthetic test surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


# Least squares only drives a penalty close to zero, so the butterfly constraint
# is imposed with a small margin to leave the fitted slice strictly arbitrage-free.
G_BUFFER = 1e-3  # Durrleman g(k) >= G_BUFFER
BUTTERFLY_PENALTY = 100.0
LEE_SLOPE = 2.0  # Lee (2004): total-variance wing slopes b (1 +/- rho) are at most 2


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: ArrayLike) -> np.ndarray:
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))

    def dw(self, k: ArrayLike) -> np.ndarray:
        """First derivative of total variance with respect to k."""
        x = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))

    def d2w(self, k: ArrayLike) -> np.ndarray:
        """Second derivative of total variance with respect to k."""
        x = np.asarray(k, dtype=float) - self.m
        return self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5

    def implied_vol(self, k: ArrayLike, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    @property
    def min_total_variance(self) -> float:
        """Minimum of w over k, attained at k = m - rho * sigma / sqrt(1 - rho^2)."""
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def is_valid(self) -> bool:
        """Parameter constraints that keep w(k) a well-defined, non-negative variance."""
        return (
            self.b >= 0.0
            and abs(self.rho) < 1.0
            and self.sigma > 0.0
            and self.min_total_variance >= 0.0
        )


def ssvi_slice(theta: float, rho: float, phi: float) -> SVIParams:
    """Raw SVI parameters of the SSVI slice w(k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)).

    ``theta`` is ATM total variance and ``phi`` the ATM curvature scale.
    """
    return SVIParams(
        a=0.5 * theta * (1.0 - rho**2),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        sigma=np.sqrt(1.0 - rho**2) / phi,
    )


def power_law_phi(theta: float, eta: float, gamma: float) -> float:
    """SSVI power-law curvature phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma)).

    With 0 < gamma <= 1/2 and eta (1 + |rho|) <= 2 the resulting surface is free of
    static arbitrage (Gatheral & Jacquier 2014, Section 4), provided theta(T) is
    non-decreasing in T.
    """
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    rmse: float  # root-mean-square error in total variance
    n_points: int


def durrleman_g(params: SVIParams, k: ArrayLike) -> np.ndarray:
    """Durrleman's function; the smile is free of butterfly arbitrage iff g(k) >= 0.

    g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1/w + 1/4) + w'' / 2
    """
    k = np.asarray(k, dtype=float)
    w, dw, d2w = params.total_variance(k), params.dw(k), params.d2w(k)
    return (1.0 - k * dw / (2.0 * w)) ** 2 - 0.25 * dw**2 * (1.0 / w + 0.25) + 0.5 * d2w


def fit_svi(
    k: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike | None = None,
    k_check: ArrayLike | None = None,
) -> SVIFit:
    """Fit raw SVI to one smile by weighted least squares on total variance.

    SVI's objective has several local minima, so the fit runs from a small grid
    of starting points and keeps the best. Box bounds enforce b >= 0, |rho| < 1,
    sigma > 0; penalty residuals enforce non-negative minimum variance and Lee's
    moment bound b (1 + |rho|) <= 2 on the wing slopes. That bound is also what
    Durrleman's g(k) needs to stay positive as |k| goes to infinity: along a
    linear wing of slope s, g tends to 1/4 - s^2/16.

    With ``k_check``, a further penalty keeps g(k) >= 0 on that grid, so the
    fitted smile is free of butterfly arbitrage there. The penalty vanishes on
    slices that clear the constraint with a small margin, so a clean smile fits
    exactly as it would unconstrained. Weights are normalised to mean one.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.size < 5:
        raise ValueError("SVI has five parameters; need at least five quotes")
    wts = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    sw = np.sqrt(wts / wts.mean())
    kc = None if k_check is None else np.asarray(k_check, dtype=float)
    # Data residuals are in total-variance units, which grow with maturity;
    # scaling the dimensionless butterfly penalty by the variance level keeps
    # its strength the same on every expiry.
    g_scale = BUTTERFLY_PENALTY * float(np.mean(w))

    w_max = float(w.max())
    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, 1e-3)
    lower = [-w_max, 1e-8, -0.999, k_lo - span, 1e-4]
    upper = [w_max, 10.0, 0.999, k_hi + span, 10.0]

    def residuals(x: np.ndarray) -> np.ndarray:
        p = SVIParams(*x)
        parts = [
            sw * (p.total_variance(k) - w),
            [1e3 * max(0.0, -p.min_total_variance), 1e3 * max(0.0, p.b * (1.0 + abs(p.rho)) - LEE_SLOPE)],
        ]
        if kc is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.nan_to_num(durrleman_g(p, kc), nan=-1.0)
            parts.append(g_scale * np.maximum(0.0, G_BUFFER - g))
        return np.concatenate(parts)

    k_atm = float(k[np.argmin(np.abs(k))])
    best = None
    for rho0, m0, sig0 in product([-0.6, -0.2, 0.2], [k_atm, 0.5 * (k_lo + k_hi)], [0.05, 0.2, 0.5]):
        b0 = 0.1
        a0 = float(w.min()) - b0 * sig0 * np.sqrt(1.0 - rho0**2)  # start at the smile's minimum
        x0 = np.clip([a0, b0, rho0, m0, sig0], lower, upper)
        res = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res

    params = SVIParams(*best.x)
    rmse = float(np.sqrt(np.mean((params.total_variance(k) - w) ** 2)))
    return SVIFit(params=params, rmse=rmse, n_points=int(k.size))

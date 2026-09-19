"""volsurf: option pricing, implied volatility and arbitrage-checked SVI volatility surfaces."""

from .arbitrage import butterfly_check, calendar_violations, durrleman_g, quote_violations
from .binomial import crr_price
from .black_scholes import Greeks, black_price, bs_price, greeks
from .implied_vol import implied_vol
from .parity import implied_forward
from .surface import ArbitrageReport, Slice, VolSurface
from .svi import SVIFit, SVIParams, fit_svi, power_law_phi, ssvi_slice

__all__ = [
    "ArbitrageReport",
    "Greeks",
    "SVIFit",
    "SVIParams",
    "Slice",
    "VolSurface",
    "black_price",
    "bs_price",
    "butterfly_check",
    "calendar_violations",
    "crr_price",
    "durrleman_g",
    "fit_svi",
    "greeks",
    "implied_forward",
    "implied_vol",
    "power_law_phi",
    "quote_violations",
    "ssvi_slice",
]

__version__ = "0.1.0"

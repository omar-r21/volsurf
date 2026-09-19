"""Snapshot a listed option chain from Yahoo Finance into volsurf's chain format.

    python examples/fetch_chain.py SPY

Writes data/<ticker>_<date>.csv with columns T, strike, kind, bid, ask, plus the
spot price (`spot`), a continuously compounded risk-free rate (`rate`) taken
from the 13-week T-bill yield, and the quote time (`as_of`). Requires the
optional `yfinance` dependency. Yahoo data is delayed and for personal/research
use only.

Time to expiry is measured from the close of the last trading session, not from
when the script runs: outside market hours the quotes are that session's
closing quotes, and a weekend run would otherwise understate every T by up to
two and a half days, which matters most on the shortest expiries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Target maturities in years; the nearest listed expiry to each is used.
TARGETS = (2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5)
MAX_REL_SPREAD = 0.5  # drop quotes whose bid/ask spread exceeds 50% of mid


NY = "America/New_York"


def market_close(day) -> pd.Timestamp:
    """16:00 New York time on the given date: US equity options' last trading moment."""
    return pd.Timestamp(pd.Timestamp(day).date()).tz_localize(NY) + pd.Timedelta(hours=16)


def year_fraction(expiry: str, as_of: pd.Timestamp) -> float:
    return (market_close(expiry) - as_of).total_seconds() / (365.0 * 86400.0)


def tbill_rate() -> float:
    """13-week T-bill (^IRX, discount basis in %) as a continuously compounded ACT/365 rate."""
    quoted = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100.0
    days = 91
    price = 1.0 - quoted * days / 360.0
    return -np.log(price) / (days / 365.0)


def fetch(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    history = tk.history(period="5d")
    spot = float(history["Close"].iloc[-1])
    # Quotes and spot are both as of the last session's close; while the market
    # is open this is slightly stale, which is fine for delayed data.
    as_of = market_close(history.index[-1])

    listed = {e: year_fraction(e, as_of) for e in tk.options if year_fraction(e, as_of) > 0}
    chosen = sorted({min(listed, key=lambda e: abs(listed[e] - t)) for t in TARGETS})

    frames = []
    for expiry in chosen:
        chain = tk.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            q = df[["strike", "bid", "ask"]].copy()
            q["kind"] = kind
            q["T"] = listed[expiry]
            q["expiry"] = expiry
            frames.append(q)

    out = pd.concat(frames, ignore_index=True)
    mid = 0.5 * (out["bid"] + out["ask"])
    keep = (out["bid"] > 0) & (out["ask"] > out["bid"]) & ((out["ask"] - out["bid"]) / mid < MAX_REL_SPREAD)
    out = out[keep].copy()
    out["spot"] = spot
    out["rate"] = tbill_rate()
    out["as_of"] = as_of.isoformat()
    return out[["as_of", "expiry", "T", "strike", "kind", "bid", "ask", "spot", "rate"]]


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    chain = fetch(ticker)
    as_of = pd.Timestamp(chain["as_of"].iloc[0])
    path = Path("data") / f"{ticker.lower()}_{as_of:%Y%m%d}.csv"
    path.parent.mkdir(exist_ok=True)
    chain.to_csv(path, index=False)
    print(
        f"{len(chain)} quotes across {chain['expiry'].nunique()} expiries, "
        f"spot {chain['spot'].iloc[0]:.2f}, rate {chain['rate'].iloc[0]:.4f}, as of {as_of} -> {path}"
    )


if __name__ == "__main__":
    main()

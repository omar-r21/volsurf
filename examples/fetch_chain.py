"""Snapshot a listed option chain from Yahoo Finance into volsurf's chain format.

    python examples/fetch_chain.py SPY

Writes data/<ticker>_<date>.csv with columns T, strike, kind, bid, ask, plus the
spot price (`spot`) and a continuously compounded risk-free rate (`rate`) taken
from the 13-week T-bill yield. Requires the optional `yfinance` dependency.
Yahoo data is delayed and for personal/research use only.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Target maturities in years; the nearest listed expiry to each is used.
TARGETS = (2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5)
MAX_REL_SPREAD = 0.5  # drop quotes whose bid/ask spread exceeds 50% of mid


def year_fraction(expiry: str, now: datetime) -> float:
    # US equity options stop trading at 16:00 New York time (20:00/21:00 UTC); 20:30 is close enough.
    expiry_dt = datetime.fromisoformat(expiry).replace(hour=20, minute=30, tzinfo=timezone.utc)
    return (expiry_dt - now).total_seconds() / (365.0 * 86400.0)


def tbill_rate() -> float:
    """13-week T-bill (^IRX, discount basis in %) as a continuously compounded ACT/365 rate."""
    quoted = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100.0
    days = 91
    price = 1.0 - quoted * days / 360.0
    return -np.log(price) / (days / 365.0)


def fetch(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    listed = {e: year_fraction(e, now) for e in tk.options}
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
    return out[["expiry", "T", "strike", "kind", "bid", "ask", "spot", "rate"]]


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    chain = fetch(ticker)
    path = Path("data") / f"{ticker.lower()}_{datetime.now():%Y%m%d}.csv"
    path.parent.mkdir(exist_ok=True)
    chain.to_csv(path, index=False)
    print(
        f"{len(chain)} quotes across {chain['expiry'].nunique()} expiries, "
        f"spot {chain['spot'].iloc[0]:.2f}, rate {chain['rate'].iloc[0]:.4f} -> {path}"
    )


if __name__ == "__main__":
    main()

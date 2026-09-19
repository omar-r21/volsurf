"""Fit a volatility surface and plot it.

    python examples/demo.py                      # synthetic chain with a known true surface
    python examples/demo.py data/spy_YYYYMMDD.csv  # a chain saved by fetch_chain.py

Prints a per-expiry summary and the arbitrage report, and writes
docs/<name>_smiles.png and docs/<name>_surface.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurf import VolSurface
from volsurf.synthetic import make_chain

DOCS = Path(__file__).resolve().parent.parent / "docs"


def load(argv: list[str]):
    if len(argv) > 1:
        path = Path(argv[1])
        chain = pd.read_csv(path)
        surface = VolSurface.from_chain(
            chain,
            spot=float(chain["spot"].iloc[0]),
            rate=float(chain["rate"].iloc[0]) if "rate" in chain else None,
            min_price=0.05,
        )
        return surface, path.stem.split("_")[0].upper(), None
    chain, truth = make_chain(seed=7)
    return VolSurface.from_chain(chain, spot=100.0), "synthetic", truth


def plot_smiles(surface: VolSurface, name: str, truth) -> Path:
    n = len(surface.slices)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, s in zip(axes.flat, surface.slices):
        q = s.quotes
        k = q["k"].to_numpy()
        grid = np.linspace(k.min(), k.max(), 300)
        lo = 100 * q["iv_bid"].fillna(0.0)
        hi = 100 * q["iv_ask"]
        ax.vlines(k, lo, hi, color="#b8c4d6", lw=1.6, label="bid/ask")
        ax.plot(k, 100 * q["iv"], "o", ms=2.4, color="#1f4e8c", label="mid")
        ax.plot(grid, 100 * s.params.implied_vol(grid, s.T), color="#d1495b", lw=1.6, label="SVI fit")
        if truth is not None:
            ax.plot(grid, 100 * truth[s.T].implied_vol(grid, s.T), "--", color="#333333", lw=1, label="true")
        days = s.T * 365
        ax.set_title(f"T = {days:.0f}d   RMSE {1e4 * s.rmse_vol:.0f} bp", fontsize=10)
        ax.set_xlabel("log-moneyness  ln(K/F)", fontsize=8)
        ax.set_ylabel("implied vol (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{name}: SVI smile fits per expiry", fontsize=13)
    fig.tight_layout()
    out = DOCS / f"{name.lower()}_smiles.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_surface(surface: VolSurface, name: str) -> Path:
    # Plot against standardised moneyness z = ln(K/F) / (atm_vol sqrt(T)). A fixed
    # k-range would be deep extrapolation at short maturities (a 10% OTM strike is
    # ~4 standard deviations out at one month but under one at a year), whereas a
    # fixed z-range stays inside the quoted strikes at every maturity.
    z = np.linspace(-4.0, 2.0, 60)
    T = np.linspace(surface.slices[0].T, surface.slices[-1].T, 60)
    iv = []
    for t in T:
        atm_std = np.sqrt(surface.total_variance_k(0.0, t))
        iv.append(np.sqrt(surface.total_variance_k(z * atm_std, t) / t))
    iv = np.array(iv)

    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(projection="3d")
    zz, tt = np.meshgrid(z, T)
    ax.plot_surface(zz, tt * 12, 100 * iv, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("ATM std devs  ln(K/F) / (σ√T)")
    ax.set_ylabel("maturity (months)")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=24, azim=-128)
    ax.set_title(f"{name}: implied volatility surface")
    fig.tight_layout()
    out = DOCS / f"{name.lower()}_surface.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    surface, name, truth = load(sys.argv)
    DOCS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    summary = surface.summary()
    summary["days"] = (365 * summary["T"]).round().astype(int)
    print(summary[["days", "forward", "discount", "atm_vol", "n_quotes", "rmse_vol_bp", "rho"]].round(4).to_string(index=False))

    report = surface.check_arbitrage()
    worst_g = min(c.min_g for c in report.butterfly.values())
    print(f"\nstatic arbitrage: {'none found' if report.ok else 'VIOLATIONS'}"
          f"  (min Durrleman g = {worst_g:.4f}, calendar violations = {len(report.calendar)})")
    if truth is not None:
        errs = [
            s.params.implied_vol(s.quotes["k"], s.T) - truth[s.T].implied_vol(s.quotes["k"], s.T)
            for s in surface.slices
        ]
        print(f"fit vs true surface: RMSE {1e4 * np.sqrt(np.mean(np.concatenate(errs) ** 2)):.1f} bp of vol")
    for path in (plot_smiles(surface, name, truth), plot_surface(surface, name)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

# volsurf

[![tests](https://github.com/omar-r21/volsurf/actions/workflows/tests.yml/badge.svg)](https://github.com/omar-r21/volsurf/actions/workflows/tests.yml)

Option pricing, implied volatility and arbitrage-checked SVI volatility surfaces in Python.

`volsurf` takes a raw option chain (strikes, bids, asks) and turns it into a smooth implied-volatility
surface you can query across strikes and maturities. Along the way it backs the forward and discount
factor out of put-call parity, inverts prices to implied vols, fits Gatheral's SVI model per expiry,
and checks the result for static arbitrage.

![SPY smile fits](docs/spy_smiles.png)

## Results

**Real data: SPY, close of 18 Sep 2026** (Yahoo Finance, 8 expiries from 2 weeks to 16 months,
1,058 out-of-the-money quotes after filtering):

| days | forward | ATM vol | quotes | fit RMSE (bp of vol) |
|---:|---:|---:|---:|---:|
| 14 | 763.96 | 10.7% | 94 | 18 |
| 28 | 765.13 | 11.7% | 123 | 15 |
| 63 | 768.14 | 13.4% | 70 | 8 |
| 91 | 769.78 | 14.0% | 89 | 9 |
| 182 | 776.66 | 15.1% | 129 | 14 |
| 272 | 783.79 | 16.0% | 160 | 17 |
| 364 | 790.98 | 16.6% | 187 | 27 |
| 490 | 800.11 | 17.1% | 206 | 40 |

Time to expiry is measured from the 16:00 New York close at which the quotes were struck.
No butterfly arbitrage on any slice out to |ln(K/F)| = 5 (checked on a grid independent of the
fit), and no calendar arbitrage on the strike range each pair of expiries is quoted on.

**Synthetic data with a known answer:** quotes priced off an arbitrage-free SSVI surface, with
20 bp of Gaussian noise on each mid vol, bid/ask spreads and tick rounding. The fitted surface is
**6 bp** from the true one: the fit averages out most of the quote noise instead of chasing it.
Forwards are recovered to about 1e-5 relative. This is a best case by construction: every SSVI
slice is exactly representable in SVI, and the call and put at each strike share one noisy vol, so
put-call parity holds in the quotes. Real chains are harder, which is what the SPY fit is for.

![SPY surface](docs/spy_surface.png)

## What's inside

| module | what it does |
|---|---|
| `black_scholes` | Black-76 and Black-Scholes-Merton prices with a dividend yield; analytic delta, gamma, vega, theta, rho. Array inputs broadcast. |
| `binomial` | Cox-Ross-Rubinstein tree for European and American options, as a check on the closed form and for early exercise. |
| `implied_vol` | Vectorised implied-vol solver: Newton's method with a bisection fallback, so it always converges. |
| `parity` | Forward and discount factor implied by put-call parity. |
| `svi` | Raw SVI smile, multi-start least-squares fit with butterfly-arbitrage and Lee wing-slope penalties, and SSVI slices. |
| `arbitrage` | Durrleman butterfly test, calendar test, and model-free checks on raw quotes. |
| `surface` | `VolSurface.from_chain(...)`: the full pipeline, interpolation across maturities, pricing, arbitrage report. |
| `synthetic` | Option chains priced off a known SSVI surface, for tests and benchmarking. |

```python
import pandas as pd
from volsurf import VolSurface

chain = pd.read_csv("data/spy_20260918.csv")        # columns: T, strike, kind, bid, ask
surface = VolSurface.from_chain(chain, spot=761.69, rate=0.0405, min_price=0.05)

surface.implied_vol(K=700, T=0.5)                    # 0.1931
surface.price(K=[700, 800], T=0.5, kind="put")
surface.check_arbitrage().ok                         # True
surface.summary()                                    # per-expiry forward, ATM vol, SVI params, fit error
```

## Method

**1. Forward and discount from put-call parity.** For European options on one expiry,
`C(K) - P(K) = D (F - K)` at every strike. Given a discount factor `D` from the T-bill curve,
each strike quoted both ways gives an estimate `F = K + (C - P) / D`. The median of those estimates,
over the strikes nearest the money, is the forward. That forward already includes dividends and
borrow cost, so neither has to be modelled. With no rate supplied, `D` comes from regressing
`C - P` on `K` across every strike quoted both ways. That works on synthetic chains, but on real
SPY quotes (American, and not synchronised across strikes) the implied rates scatter well away from
the T-bill rate, so the fit warns when an implied rate is implausible and supplying a rate is the
recommended path.

**2. Implied vol from out-of-the-money quotes.** Puts below the forward, calls above. They are
the liquid side, and for American options they carry the least early-exercise premium. The solver
converts every quote to its OTM equivalent via parity before inverting. Backing a tiny time value
out of a large in-the-money price loses precision to cancellation; OTM prices *are* the time value.
Each option keeps a bracket known to contain the root, and a Newton step that leaves it (common
deep in the wings, where vega underflows) falls back to bisection.

**3. SVI per expiry.** Raw SVI (Gatheral, 2004) models total variance `w = σ²T` against
log-moneyness `k = ln(K/F)`:

    w(k) = a + b (ρ (k − m) + √((k − m)² + σ²))

Its wings are linear in `k`, as Lee's moment formula requires, so it extrapolates far more sensibly
than a polynomial in strike. The fit enforces Lee's bound on the wing slopes, `b(1 + |ρ|) ≤ 2`. The
five-parameter least-squares problem has several local minima, so the fit runs from 18 starting
points and keeps the best.

**4. Static arbitrage.** A smile has no butterfly arbitrage iff the implied risk-neutral density is
non-negative, which for total variance `w(k)` is Durrleman's condition

    g(k) = (1 − k w′/(2w))² − (w′²/4)(1/w + 1/4) + w″/2 ≥ 0.

The fit penalises `g < 0` on a fine grid over the quoted range and a coarse grid out to `|k| = 3`.
Beyond that, Lee's bound does the work: along a linear wing of slope `s`, `g → 1/4 − s²/16`, which is
positive exactly when `s < 2`. A clean smile fits exactly as it would unconstrained. Calendar arbitrage (total variance falling with maturity at fixed `k`) is checked on
the range both expiries are quoted on. The tests include Vogt's well-known slice from Gatheral &
Jacquier (2014): its parameters look valid, but it implies a negative density, and the checker
catches it.

**5. Across maturities.** Total variance is interpolated linearly in `T` at fixed `k`. That
preserves calendar-arbitrage-freeness between slices that are, though not butterfly-freeness (see
Limitations). Before the first and after the last expiry, implied vol is held constant in `T`.
Carry and rates are interpolated from the per-expiry forwards and discount factors.

## Design decisions, including what didn't work

- **Calendar arbitrage is checked, not imposed.** The first version fitted expiries in order and
  penalised each one for dipping below the previous fit. On SPY this cascaded: the short-dated
  slices' extrapolated call wings forced every longer expiry upward, and fit error grew from ~15 bp
  to over 1,000 bp. Fitted independently on clean quotes, the SPY slices have no calendar arbitrage
  at all on their shared range, so the constraint was fixing a problem that the data filtering had
  already removed.
- **Weighting by bid/ask width made fits worse.** SPY spreads are a cent or two near the money, so
  inverse-spread weights put nearly all the weight on a handful of ATM strikes and the wings drifted.
  Uniform weights on filtered quotes fit better everywhere.
- **An arbitrage check has to be independent of the fit.** The first version checked butterfly
  arbitrage on the same grid the fit was penalised on, so a clean report was all but guaranteed.
  An independent review found that a 2-month slice had negative butterflies at strikes just past
  the last quote. The penalty grid now reaches far into the wings, Lee's wing bound covers the rest,
  and the tests check on a grid the fit never saw.
- **Time to expiry is measured from the quote, not the clock.** Fetching the chain on a Saturday and
  measuring T from "now" shortened every expiry by about a day. On the 2-week expiry that shifted ATM
  vol by 35 bp, more than the fit error. T is now measured from the session close.
- **Quote filtering matters more than the optimiser.** Dropping OTM quotes with a mid under 5¢ or
  beyond 6 ATM standard deviations cut short-dated fit error from ~40 bp to under 20 bp. Far-wing quotes
  a tick wide carry almost no information about vol, but a five-parameter model still tries to fit them.

## Limitations

- SPY options are American. Out-of-the-money quotes and near-the-money parity keep the early-exercise
  premium small, but it is not modelled; a de-Americanisation step (binomial tree per quote) would be
  the next refinement for single stocks with large dividends.
- One flat rate (13-week T-bill) discounts every expiry; a proper curve matters more past a year.
- Linear interpolation of total variance between two butterfly-free slices is calendar-free but not
  guaranteed butterfly-free, and the same goes for constant-vol extrapolation past the last expiry.
  Arbitrage is checked on the fitted slices, not between them.
- Slices are fitted independently. A global parameterisation such as SSVI (arbitrage-free by
  construction, used here for synthetic data) trades some fit quality for guaranteed consistency.
- Yahoo data is delayed and not synchronised across strikes; it is fine for research, not trading.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,data]"

pytest                                                  # 43 tests
python examples/demo.py                                 # synthetic surface, plots to docs/
python examples/fetch_chain.py SPY                      # snapshot a live chain to data/
python examples/demo.py data/spy_YYYYMMDD.csv           # fit and plot it
```

The tests cover textbook prices (Hull's 10.4506 call), put-call parity, every Greek against finite
differences, binomial convergence to Black-Scholes, early-exercise behaviour, implied-vol round trips
from 5% to 150% vol and 1 week to 5 years, SVI parameter recovery, Vogt's arbitrage example, all
three quote-level arbitrage checks (with and without discounting), the butterfly penalty repairing
Vogt's slice, arbitrage checks on a grid independent of the fit, NaN rather than a wrong answer for
unresolvable prices, extrapolation and forward/discount interpolation, and end-to-end recovery of a
known surface from noisy quotes, with and without a supplied rate.

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization with
  application to the valuation of volatility derivatives.* Global Derivatives & Risk Management.
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance 14(1).
- Lee, R. (2004). *The moment formula for implied volatility at extreme strikes.* Mathematical Finance 14(3).
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option pricing: a simplified approach.* Journal of Financial Economics 7(3).

import numpy as np
import pytest

from volsurf import VolSurface
from volsurf.synthetic import make_chain

SPOT, R, Q = 100.0, 0.04, 0.015


@pytest.fixture(scope="module")
def fitted():
    chain, truth = make_chain(spot=SPOT, r=R, q=Q, seed=7)
    return VolSurface.from_chain(chain, spot=SPOT), truth


def test_every_expiry_is_fitted(fitted):
    surface, truth = fitted
    assert [s.T for s in surface.slices] == sorted(truth)


def test_forward_and_discount_recovered_from_parity(fitted):
    surface, _ = fitted
    for s in surface.slices:
        # Residual error comes from rounding bids down and asks up to the tick.
        assert s.forward == pytest.approx(SPOT * np.exp((R - Q) * s.T), rel=5e-5)
        assert s.discount == pytest.approx(np.exp(-R * s.T), rel=1e-4)


def test_supplied_rate_fixes_discount_and_still_recovers_forward():
    chain, _ = make_chain(spot=SPOT, r=R, q=Q, seed=7)
    surface = VolSurface.from_chain(chain, spot=SPOT, rate=R)
    for s in surface.slices:
        assert s.discount == pytest.approx(np.exp(-R * s.T), rel=1e-12)
        assert s.forward == pytest.approx(SPOT * np.exp((R - Q) * s.T), rel=5e-5)


def test_forward_and_discount_interpolate_between_and_beyond_slices(fitted):
    surface, _ = fitted
    # Constant r and q in the data, so the interpolated curves must reproduce them anywhere.
    for T in (0.02, 0.4, 1.5, 3.0):
        assert surface.forward(T) == pytest.approx(SPOT * np.exp((R - Q) * T), rel=1e-4)
        assert surface.discount(T) == pytest.approx(np.exp(-R * T), rel=1e-3)


def test_fitted_smiles_track_the_true_surface(fitted):
    surface, truth = fitted
    for s in surface.slices:
        # Market noise is 20bp of vol; the fit should sit well inside that on average.
        assert s.rmse_vol < 0.004
        k = s.quotes["k"].to_numpy()
        err = s.params.implied_vol(k, s.T) - truth[s.T].implied_vol(k, s.T)
        assert np.sqrt(np.mean(err**2)) < 0.003


def test_fitted_surface_is_arbitrage_free(fitted):
    surface, _ = fitted
    assert surface.check_arbitrage().ok


def test_no_butterfly_arbitrage_far_outside_the_fitted_grid(fitted):
    # An independent grid, much wider than anything the fit was penalised on.
    surface, _ = fitted
    report = surface.check_arbitrage(k=np.linspace(-6.0, 6.0, 2401))
    assert all(c.ok for c in report.butterfly.values()), report.butterfly


def test_interpolation_between_expiries(fitted):
    surface, _ = fitted
    s1, s2 = surface.slices[3], surface.slices[4]
    T = 0.5 * (s1.T + s2.T)
    w_mid = surface.total_variance_k(0.0, T)
    assert s1.params.total_variance(0.0) < w_mid < s2.params.total_variance(0.0)
    # On a slice, the surface returns that slice exactly.
    assert surface.total_variance_k(0.1, s1.T) == pytest.approx(s1.params.total_variance(0.1))


def test_extrapolation_holds_implied_vol_constant_in_maturity(fitted):
    surface, _ = fitted
    first, last = surface.slices[0], surface.slices[-1]
    k = np.linspace(-0.3, 0.2, 7)
    for s, T in ((first, 0.5 * first.T), (last, 2.0 * last.T)):
        vol = np.sqrt(surface.total_variance_k(k, T) / T)
        np.testing.assert_allclose(vol, s.params.implied_vol(k, s.T), rtol=1e-12)


def test_surface_queries_need_a_single_positive_maturity(fitted):
    surface, _ = fitted
    with pytest.raises(ValueError):
        surface.implied_vol(100.0, np.array([0.5, 1.0]))
    with pytest.raises(ValueError):
        surface.implied_vol(100.0, -0.5)


def test_surface_prices_satisfy_put_call_parity(fitted):
    surface, _ = fitted
    K, T = np.array([80.0, 100.0, 120.0]), 0.75
    lhs = surface.price(K, T, "call") - surface.price(K, T, "put")
    rhs = surface.discount(T) * (surface.forward(T) - K)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_chain_validation():
    chain, _ = make_chain()
    with pytest.raises(ValueError, match="missing columns"):
        VolSurface.from_chain(chain.drop(columns="bid"), spot=SPOT)

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
        assert s.discount == pytest.approx(np.exp(-R * s.T), rel=1e-3)


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
    report = surface.check_arbitrage(k=np.linspace(-0.8, 0.5, 261))
    assert report.ok, report


def test_interpolation_between_expiries(fitted):
    surface, _ = fitted
    s1, s2 = surface.slices[3], surface.slices[4]
    T = 0.5 * (s1.T + s2.T)
    w_mid = surface.total_variance_k(0.0, T)
    assert s1.params.total_variance(0.0) < w_mid < s2.params.total_variance(0.0)
    # On a slice, the surface returns that slice exactly.
    assert surface.total_variance_k(0.1, s1.T) == pytest.approx(s1.params.total_variance(0.1))


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

import numpy as np
import pytest

from volsurf import black_price, implied_forward, implied_vol

F, D = 100.0, 0.98


@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_across_strikes_expiries_and_vols(kind):
    K, T, vol = np.meshgrid(
        np.linspace(50, 200, 16), [0.02, 0.25, 1.0, 5.0], [0.05, 0.2, 0.6, 1.5], indexing="ij"
    )
    price = black_price(F, K, T, vol, D, kind)
    # Volatility is only identified by time value. Deep in the money it can fall
    # below double-precision resolution of the price, leaving nothing to invert.
    intrinsic = D * np.maximum(F - K if kind == "call" else K - F, 0.0)
    usable = price - intrinsic > 1e-8 * F
    iv = implied_vol(price, F, K, T, D, kind)
    np.testing.assert_allclose(iv[usable], vol[usable], atol=1e-7)


def test_mixed_calls_and_puts_in_one_call():
    K = np.array([80.0, 100.0, 120.0])
    kind = np.array(["put", "call", "call"])
    price = black_price(F, K, 0.5, 0.3, D, kind)
    np.testing.assert_allclose(implied_vol(price, F, K, 0.5, D, kind), 0.3, atol=1e-9)


def test_deep_otm_wing_is_solved_accurately():
    # A 50%-OTM put worth a fraction of a cent: solving on the OTM side keeps full precision.
    price = black_price(F, 50.0, 0.25, 0.45, D, "put")
    assert 0 < price < 0.01
    assert implied_vol(price, F, 50.0, 0.25, D, "put") == pytest.approx(0.45, abs=1e-9)


def test_prices_outside_no_arbitrage_bounds_are_nan():
    T = 1.0
    below_intrinsic = D * (F - 90) - 0.01
    above_upper = D * F + 0.01
    iv = implied_vol([below_intrinsic, above_upper, 5.0], F, [90, 90, 100], [T, T, 0.0], D, "call")
    assert np.isnan(iv).all()


def test_implied_forward_recovers_forward_and_discount():
    K = np.linspace(80, 120, 9)
    call = black_price(F, K, 0.5, 0.25, D, "call")
    put = black_price(F, K, 0.5, 0.25, D, "put")
    fwd, disc = implied_forward(K, call, put)
    assert fwd == pytest.approx(F, rel=1e-10)
    assert disc == pytest.approx(D, rel=1e-10)


def test_implied_forward_needs_two_strikes():
    with pytest.raises(ValueError):
        implied_forward([100, 100], [5, 5], [4, 4])

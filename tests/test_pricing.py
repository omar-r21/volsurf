import numpy as np
import pytest

from volsurf import black_price, bs_price, crr_price, greeks

S, K, T, R, VOL, Q = 100.0, 105.0, 0.75, 0.04, 0.25, 0.02


def test_textbook_values():
    # Hull's standard example: S=K=100, T=1, r=5%, sigma=20%, no dividends.
    assert bs_price(100, 100, 1, 0.05, 0.2, kind="call") == pytest.approx(10.450584, abs=1e-6)
    assert bs_price(100, 100, 1, 0.05, 0.2, kind="put") == pytest.approx(5.573526, abs=1e-6)


def test_put_call_parity():
    strikes = np.linspace(60, 140, 17)
    call = bs_price(S, strikes, T, R, VOL, Q, "call")
    put = bs_price(S, strikes, T, R, VOL, Q, "put")
    np.testing.assert_allclose(call - put, S * np.exp(-Q * T) - strikes * np.exp(-R * T), atol=1e-12)


def test_zero_vol_is_discounted_intrinsic():
    F, D = 100.0, 0.97
    assert black_price(F, 90, 1.0, 0.0, D, "call") == pytest.approx(D * 10)
    assert black_price(F, 90, 1.0, 0.0, D, "put") == pytest.approx(0.0)


def test_negative_vol_or_expiry_is_nan():
    assert np.isnan(black_price(100, 90, 1.0, -0.2))
    assert np.isnan(black_price(100, 90, -1.0, 0.2))


def test_kind_accepts_arrays_and_rejects_garbage():
    prices = black_price(100, [90, 110], 1.0, 0.2, 1.0, ["C", "put"])
    assert prices.shape == (2,)
    with pytest.raises(ValueError):
        black_price(100, 100, 1.0, 0.2, 1.0, "straddle")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = greeks(S, K, T, R, VOL, Q, kind)
    price = lambda **kw: bs_price(**{"S": S, "K": K, "T": T, "r": R, "sigma": VOL, "q": Q, "kind": kind, **kw})

    h = 1e-3
    assert g.price == pytest.approx(price())
    assert g.delta == pytest.approx((price(S=S + h) - price(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((price(S=S + h) - 2 * price() + price(S=S - h)) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((price(sigma=VOL + h) - price(sigma=VOL - h)) / (2 * h), rel=1e-6)
    assert g.rho == pytest.approx((price(r=R + h) - price(r=R - h)) / (2 * h), rel=1e-6)
    # Theta is the derivative in calendar time, i.e. minus the derivative in T.
    assert g.theta == pytest.approx(-(price(T=T + h) - price(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_converges_to_black_scholes(kind):
    exact = float(bs_price(S, K, T, R, VOL, Q, kind))
    assert crr_price(S, K, T, R, VOL, Q, kind, steps=2000) == pytest.approx(exact, abs=2e-3)


def test_binomial_accepts_short_kind_names():
    assert crr_price(S, K, T, R, VOL, Q, "p", steps=200) == crr_price(S, K, T, R, VOL, Q, "put", steps=200)


def test_american_put_carries_early_exercise_premium():
    euro = crr_price(S, 120, 1.0, 0.08, 0.2, 0.0, "put", american=False)
    amer = crr_price(S, 120, 1.0, 0.08, 0.2, 0.0, "put", american=True)
    assert amer > euro + 0.1
    assert amer >= 120 - S  # never worth less than immediate exercise


def test_american_call_without_dividends_is_european():
    # With q = 0 early exercise of a call is never optimal (Merton, 1973).
    euro = crr_price(S, K, 1.0, R, VOL, 0.0, "call")
    amer = crr_price(S, K, 1.0, R, VOL, 0.0, "call", american=True)
    assert amer == pytest.approx(euro, abs=1e-10)

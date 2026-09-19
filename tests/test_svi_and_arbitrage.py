import numpy as np
import pytest

from volsurf import (
    SVIParams,
    butterfly_check,
    calendar_violations,
    durrleman_g,
    fit_svi,
    power_law_phi,
    quote_violations,
    ssvi_slice,
)
from volsurf.synthetic import SSVISurface

# Axel Vogt's slice from Gatheral & Jacquier (2014): valid SVI parameters with butterfly arbitrage.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_svi_derivatives_match_finite_differences():
    p = SVIParams(a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
    k, h = np.linspace(-1, 1, 11), 1e-5
    np.testing.assert_allclose(p.dw(k), (p.total_variance(k + h) - p.total_variance(k - h)) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(
        p.d2w(k), (p.total_variance(k + h) - 2 * p.total_variance(k) + p.total_variance(k - h)) / h**2, atol=1e-4
    )


def test_ssvi_slice_matches_its_closed_form():
    theta, rho = 0.04, -0.6
    phi = power_law_phi(theta, eta=1.1, gamma=0.45)
    k = np.linspace(-1, 1, 21)
    expected = 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    np.testing.assert_allclose(ssvi_slice(theta, rho, phi).total_variance(k), expected, atol=1e-14)
    assert ssvi_slice(theta, rho, phi).total_variance(0.0) == pytest.approx(theta)


def test_fit_recovers_exact_smile():
    true = SSVISurface().slice(0.5)
    k = np.linspace(-0.6, 0.4, 25)
    fit = fit_svi(k, true.total_variance(k))
    assert fit.rmse < 1e-7
    assert fit.params.is_valid()
    np.testing.assert_allclose(fit.params.total_variance(k), true.total_variance(k), atol=1e-7)


def test_fit_needs_five_points():
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1, 0.2, 0.3], [0.04] * 4)


def test_ssvi_surface_is_free_of_static_arbitrage():
    surface = SSVISurface()
    slices = [(T, surface.slice(T)) for T in (0.05, 0.25, 0.5, 1.0, 2.0, 5.0)]
    for _, p in slices:
        assert butterfly_check(p).ok
    assert calendar_violations(slices) == []


def test_vogt_slice_has_butterfly_arbitrage():
    assert VOGT.is_valid()  # the parameters themselves look fine...
    check = butterfly_check(VOGT)
    assert not check.ok  # ...but the implied density goes negative
    assert durrleman_g(VOGT, check.k_at_min) < 0


def test_crossing_slices_are_calendar_arbitrage():
    short = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    long = SVIParams(a=0.03, b=0.1, rho=-0.3, m=0.0, sigma=0.2)  # lower variance later
    violations = calendar_violations([(0.5, short), (1.0, long)])
    assert len(violations) == 1
    assert violations[0].excess_variance == pytest.approx(0.02)


def test_quote_checks_flag_each_kind_of_violation():
    K = [90, 95, 100, 105, 110]
    clean = [12.0, 8.0, 5.0, 3.0, 2.0]
    assert quote_violations(K, clean) == []

    # Call price rises from K=105 to K=110.
    rising = quote_violations(K, [12.0, 8.0, 5.0, 3.0, 3.2])
    assert [(v.kind, v.strikes) for v in rising] == [("monotonicity", (105.0, 110.0))]

    # Slopes -0.8, -0.2, -0.6: the 95/100/105 butterfly has negative cost.
    concave = quote_violations(K, [12.0, 8.0, 7.0, 4.0, 2.5])
    assert [(v.kind, v.strikes) for v in concave] == [("convexity", (95.0, 100.0, 105.0))]

    # Call spread 90/95 costs 12 for a payoff of at most 5.
    steep = quote_violations(K, [20.0, 8.0, 5.0, 3.0, 2.0])
    assert {v.kind for v in steep} == {"slope"}

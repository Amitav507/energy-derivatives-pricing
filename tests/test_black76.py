import math

import pytest

from engine.black76 import Black76
from engine.black_scholes import BlackScholes, OptionType


def test_matches_bsm_substitution_across_a_grid():
    # Black-76 is the zero-carry case of BSM, so spot=F with q=r must agree
    # on price exactly, not approximately.
    for F in [50, 80, 100, 130]:
        for K in [60, 90, 100, 120]:
            for T in [0.1, 0.5, 2.0]:
                for vol in [0.1, 0.35, 0.8]:
                    for kind in [OptionType.CALL, OptionType.PUT]:
                        b76 = Black76(F, K, T, 0.04, vol, kind)
                        bsm = BlackScholes(F, K, T, 0.04, vol, kind, dividend_yield=0.04)
                        assert b76.price() == pytest.approx(bsm.price(), abs=1e-12)


def test_rho_is_negative_and_equals_minus_t_times_price():
    # The rate enters Black-76 only through the discount factor, so a higher
    # rate can only reduce the present value.
    b76 = Black76(100, 100, 2.0, 0.05, 0.25, OptionType.CALL)
    g = b76.greeks()
    assert g.rho == pytest.approx(-2.0 * b76.price(), abs=1e-12)
    assert g.rho < 0


def test_only_rho_differs_from_the_bsm_substitution():
    b76 = Black76(100, 95, 1.0, 0.05, 0.3, OptionType.CALL)
    bsm = BlackScholes(100, 95, 1.0, 0.05, 0.3, OptionType.CALL, dividend_yield=0.05)
    a, b = b76.greeks(), bsm.greeks()

    assert a.delta == pytest.approx(b.delta, abs=1e-12)
    assert a.gamma == pytest.approx(b.gamma, abs=1e-12)
    assert a.vega == pytest.approx(b.vega, abs=1e-12)
    assert a.theta == pytest.approx(b.theta, abs=1e-12)

    # The substitution reports a positive rho because it differentiates r with
    # q held fixed. In forward mode r and q are the same number, so that figure
    # is meaningless and carries the wrong sign.
    assert b.rho > 0
    assert a.rho < 0


def test_put_call_parity_for_futures_options():
    call = Black76(105, 100, 1.5, 0.04, 0.28, OptionType.CALL)
    put = Black76(105, 100, 1.5, 0.04, 0.28, OptionType.PUT)
    expected = math.exp(-0.04 * 1.5) * (105 - 100)
    assert call.price() - put.price() == pytest.approx(expected, abs=1e-10)


def test_zero_vol_price_is_discounted_intrinsic_on_the_forward():
    call = Black76(100, 90, 2.0, 0.05, 0.0, OptionType.CALL)
    assert call.price() == pytest.approx(math.exp(-0.10) * 10.0, abs=1e-12)

    put = Black76(90, 100, 2.0, 0.05, 0.0, OptionType.PUT)
    assert put.price() == pytest.approx(math.exp(-0.10) * 10.0, abs=1e-12)


def test_zero_time_to_expiry_returns_intrinsic():
    call = Black76(110, 100, 0.0, 0.05, 0.2, OptionType.CALL)
    assert call.price() == pytest.approx(10.0, abs=1e-12)


def test_delta_is_bounded_by_the_discount_factor():
    disc = math.exp(-0.04 * 1.0)
    call = Black76(100, 100, 1.0, 0.04, 0.3, OptionType.CALL)
    put = Black76(100, 100, 1.0, 0.04, 0.3, OptionType.PUT)
    assert 0.0 < call.greeks().delta < disc
    assert -disc < put.greeks().delta < 0.0
    # Call delta minus put delta is exactly the discount factor.
    assert call.greeks().delta - put.greeks().delta == pytest.approx(disc, abs=1e-12)


def test_gamma_and_vega_match_for_call_and_put():
    call = Black76(100, 90, 0.75, 0.03, 0.22, OptionType.CALL)
    put = Black76(100, 90, 0.75, 0.03, 0.22, OptionType.PUT)
    assert call.greeks().gamma == pytest.approx(put.greeks().gamma, abs=1e-12)
    assert call.greeks().vega == pytest.approx(put.greeks().vega, abs=1e-12)


def test_theta_is_negative_for_a_long_atm_option():
    b76 = Black76(100, 100, 0.25, 0.04, 0.3, OptionType.CALL)
    assert b76.greeks().theta < 0


def test_implied_volatility_round_trip():
    for kind in [OptionType.CALL, OptionType.PUT]:
        truth = 0.37
        target = Black76(85, 95, 0.8, 0.035, truth, kind).price()
        solved = Black76(85, 95, 0.8, 0.035, 0.2, kind).implied_volatility(target)
        assert solved == pytest.approx(truth, abs=1e-6)


def test_implied_volatility_rejects_prices_outside_bounds():
    b76 = Black76(100, 100, 1.0, 0.04, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        b76.implied_volatility(-1.0)
    with pytest.raises(ValueError):
        b76.implied_volatility(1000.0)


def test_negative_forward_is_rejected_with_a_pointer_to_bachelier():
    with pytest.raises(ValueError, match="Bachelier"):
        Black76(-5, 10, 1.0, 0.04, 0.5, OptionType.CALL)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        Black76(100, 0, 1.0, 0.04, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        Black76(100, 100, -1.0, 0.04, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        Black76(100, 100, 1.0, 0.04, -0.2, OptionType.CALL)

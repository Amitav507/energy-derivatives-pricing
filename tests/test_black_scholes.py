import math

import pytest

from engine.black_scholes import BlackScholes, OptionType


# Reference case (Hull, "Options, Futures, and Other Derivatives"):
# S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0 -> call ~= 4.76, put ~= 0.81
def test_hull_reference_call():
    bs = BlackScholes(42, 40, 0.5, 0.10, 0.20, OptionType.CALL)
    assert bs.price() == pytest.approx(4.759, abs=1e-3)


def test_hull_reference_put():
    bs = BlackScholes(42, 40, 0.5, 0.10, 0.20, OptionType.PUT)
    assert bs.price() == pytest.approx(0.809, abs=1e-3)


def test_put_call_parity():
    call = BlackScholes(100, 95, 1.0, 0.05, 0.30, OptionType.CALL, dividend_yield=0.02)
    put = BlackScholes(100, 95, 1.0, 0.05, 0.30, OptionType.PUT, dividend_yield=0.02)
    lhs = call.price() - put.price()
    rhs = 100 * math.exp(-0.02 * 1.0) - 95 * math.exp(-0.05 * 1.0)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_delta_bounds():
    call = BlackScholes(100, 100, 1.0, 0.03, 0.25, OptionType.CALL)
    put = BlackScholes(100, 100, 1.0, 0.03, 0.25, OptionType.PUT)
    assert 0.0 <= call.greeks().delta <= 1.0
    assert -1.0 <= put.greeks().delta <= 0.0


def test_gamma_matches_for_call_and_put():
    call = BlackScholes(100, 90, 0.75, 0.04, 0.22, OptionType.CALL)
    put = BlackScholes(100, 90, 0.75, 0.04, 0.22, OptionType.PUT)
    assert call.greeks().gamma == pytest.approx(put.greeks().gamma, abs=1e-10)


def test_deep_itm_call_price_near_intrinsic_minus_pv_strike():
    bs = BlackScholes(200, 50, 1.0, 0.02, 0.20, OptionType.CALL)
    forward_intrinsic = 200 - 50 * math.exp(-0.02 * 1.0)
    assert bs.price() == pytest.approx(forward_intrinsic, abs=1e-2)


def test_zero_time_to_expiry_returns_intrinsic():
    call = BlackScholes(110, 100, 0.0, 0.05, 0.20, OptionType.CALL)
    put = BlackScholes(90, 100, 0.0, 0.05, 0.20, OptionType.PUT)
    assert call.price() == pytest.approx(10.0)
    assert put.price() == pytest.approx(10.0)


def test_zero_vol_prices_discounted_forward_not_spot_intrinsic():
    # With sigma=0 the forward is deterministic, so the option is worth the
    # discounted payoff on that forward -- not max(S-K, 0).
    call = BlackScholes(100, 90, 2.0, 0.05, 0.0, OptionType.CALL)
    expected = 100 - 90 * math.exp(-0.05 * 2.0)
    assert call.price() == pytest.approx(expected, abs=1e-10)

    put = BlackScholes(90, 100, 2.0, 0.05, 0.0, OptionType.PUT)
    expected_put = 100 * math.exp(-0.05 * 2.0) - 90
    assert put.price() == pytest.approx(expected_put, abs=1e-10)


def test_deep_itm_european_put_can_trade_below_spot_intrinsic():
    # A deep-ITM European put legitimately prices below K - S because it cannot
    # be exercised early. The IV solver must accept such a quote, not reject it
    # as an arbitrage.
    bs = BlackScholes(60, 100, 2.0, 0.08, 0.25, OptionType.PUT)
    price = bs.price()
    spot_intrinsic = 100 - 60
    assert price < spot_intrinsic

    iv = BlackScholes(60, 100, 2.0, 0.08, 0.40, OptionType.PUT).implied_volatility(price)
    assert iv == pytest.approx(0.25, abs=1e-6)


def test_implied_volatility_round_trip():
    true_vol = 0.27
    bs = BlackScholes(100, 105, 0.8, 0.04, true_vol, OptionType.CALL)
    market_price = bs.price()

    solver = BlackScholes(100, 105, 0.8, 0.04, 0.20, OptionType.CALL)
    iv = solver.implied_volatility(market_price)
    assert iv == pytest.approx(true_vol, abs=1e-6)


def test_implied_volatility_put_round_trip():
    true_vol = 0.35
    bs = BlackScholes(100, 90, 1.5, 0.03, true_vol, OptionType.PUT, dividend_yield=0.01)
    market_price = bs.price()

    solver = BlackScholes(100, 90, 1.5, 0.03, 0.20, OptionType.PUT, dividend_yield=0.01)
    iv = solver.implied_volatility(market_price)
    assert iv == pytest.approx(true_vol, abs=1e-6)


def test_implied_volatility_rejects_price_outside_bounds():
    bs = BlackScholes(100, 100, 1.0, 0.02, 0.20, OptionType.CALL)
    with pytest.raises(ValueError):
        bs.implied_volatility(-1.0)
    with pytest.raises(ValueError):
        bs.implied_volatility(1000.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BlackScholes(-10, 100, 1.0, 0.05, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        BlackScholes(100, 0, 1.0, 0.05, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        BlackScholes(100, 100, -1.0, 0.05, 0.2, OptionType.CALL)

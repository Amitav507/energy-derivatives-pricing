import math

import pytest

from engine.black76 import Black76
from engine.black_scholes import OptionType
from engine.monte_carlo import price_european


# Simulation and the closed form are independent routes to the same number,
# so agreement is a real cross-check rather than a formula compared to itself.
# The threshold is four standard errors: wide enough that a correct
# implementation will not fail on an unlucky seed, tight enough that a wrong
# one will not pass.
TOLERANCE_IN_STANDARD_ERRORS = 4.0


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_converges_to_the_black76_closed_form(option_type):
    args = (100.0, 100.0, 1.0, 0.04, 0.30)
    closed = Black76(*args, option_type).price()
    mc = price_european(*args, option_type, paths=200_000, seed=7)

    assert abs(mc.price - closed) < TOLERANCE_IN_STANDARD_ERRORS * mc.standard_error


@pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
def test_converges_across_moneyness(strike):
    args = (100.0, strike, 0.5, 0.03, 0.45)
    closed = Black76(*args, OptionType.CALL).price()
    mc = price_european(*args, OptionType.CALL, paths=200_000, seed=11)

    assert abs(mc.price - closed) < TOLERANCE_IN_STANDARD_ERRORS * mc.standard_error


def test_closed_form_sits_inside_the_confidence_interval():
    args = (85.0, 95.0, 0.75, 0.035, 0.38)
    closed = Black76(*args, OptionType.CALL).price()
    mc = price_european(*args, OptionType.CALL, paths=400_000, seed=3)

    low, high = mc.confidence_95
    assert low < closed < high


def test_antithetic_sampling_reduces_the_standard_error():
    args = (100.0, 105.0, 1.0, 0.04, 0.35)
    plain = price_european(*args, paths=100_000, seed=5, antithetic=False)
    paired = price_european(*args, paths=100_000, seed=5, antithetic=True)

    # Same number of payoff evaluations, lower sampling error, because a
    # monotone payoff makes Z and -Z negatively correlated.
    assert paired.standard_error < plain.standard_error


def test_standard_error_falls_as_paths_rise():
    args = (100.0, 100.0, 1.0, 0.04, 0.3)
    small = price_european(*args, paths=50_000, seed=2)
    large = price_european(*args, paths=200_000, seed=2)
    assert large.standard_error < small.standard_error


def test_put_call_parity_holds_in_simulation():
    args = (110.0, 100.0, 1.25, 0.04, 0.3)
    call = price_european(*args, OptionType.CALL, paths=300_000, seed=13)
    put = price_european(*args, OptionType.PUT, paths=300_000, seed=13)

    expected = math.exp(-0.04 * 1.25) * (110.0 - 100.0)
    combined_error = call.standard_error + put.standard_error
    assert abs((call.price - put.price) - expected) < 4.0 * combined_error


def test_deterministic_for_a_fixed_seed():
    args = (100.0, 100.0, 1.0, 0.04, 0.3)
    first = price_european(*args, paths=10_000, seed=99)
    second = price_european(*args, paths=10_000, seed=99)
    assert first.price == second.price


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        price_european(-100.0, 100.0, 1.0, 0.04, 0.3)
    with pytest.raises(ValueError):
        price_european(100.0, 0.0, 1.0, 0.04, 0.3)
    with pytest.raises(ValueError):
        price_european(100.0, 100.0, 1.0, 0.04, -0.3)
    with pytest.raises(ValueError):
        price_european(100.0, 100.0, 1.0, 0.04, 0.3, paths=1)

"""Black-Scholes-Merton pricing engine for European options.

Supports an optional continuous dividend yield (q), full Greeks, and an
implied-volatility solver. Pure standard library — no external deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ._solver import solve_implied_vol


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1.00 (100%) change in vol; divide by 100 for a 1-vol-point move
    theta: float  # per year; divide by 365 for per-day decay
    rho: float    # per 1.00 (100%) change in rate; divide by 100 for a 1% move


class BlackScholes:
    """Prices a single European option under Black-Scholes-Merton.

    Args:
        spot: current underlying price (S)
        strike: strike price (K)
        time_to_expiry: time to expiry in years (T)
        rate: continuously-compounded risk-free rate (r)
        volatility: annualized volatility, e.g. 0.20 for 20% (sigma)
        option_type: OptionType.CALL or OptionType.PUT
        dividend_yield: continuous dividend yield (q), default 0
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
    ):
        if spot <= 0:
            raise ValueError("spot must be positive")
        if strike <= 0:
            raise ValueError("strike must be positive")
        if time_to_expiry < 0:
            raise ValueError("time_to_expiry cannot be negative")
        if volatility < 0:
            raise ValueError("volatility cannot be negative")

        self.spot = float(spot)
        self.strike = float(strike)
        self.time_to_expiry = float(time_to_expiry)
        self.rate = float(rate)
        self.volatility = float(volatility)
        self.option_type = OptionType(option_type)
        self.dividend_yield = float(dividend_yield)

    def _d1_d2(self) -> tuple[float, float]:
        S, K, T, r, sigma, q = (
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.rate,
            self.volatility,
            self.dividend_yield,
        )
        if T == 0 or sigma == 0:
            raise ZeroDivisionError(
                "d1/d2 undefined at T=0 or sigma=0; use intrinsic value instead"
            )
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _zero_vol_price(self) -> float:
        """Present value in the sigma -> 0 limit.

        With no volatility the forward F = S*e^((r-q)T) is deterministic, so the
        option is worth the discounted payoff on that forward, NOT the
        undiscounted spot intrinsic. Because price is monotonically increasing
        in sigma, this is also the European no-arbitrage lower bound — which is
        why a deep-ITM European put legitimately trades below K - S.
        """
        discounted_spot = self.spot * math.exp(-self.dividend_yield * self.time_to_expiry)
        discounted_strike = self.strike * math.exp(-self.rate * self.time_to_expiry)
        if self.option_type is OptionType.CALL:
            return max(discounted_spot - discounted_strike, 0.0)
        return max(discounted_strike - discounted_spot, 0.0)

    def price(self) -> float:
        """Present value of the option."""
        if self.time_to_expiry == 0 or self.volatility == 0:
            return self._zero_vol_price()

        S, K, T, r, q = (
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.rate,
            self.dividend_yield,
        )
        d1, d2 = self._d1_d2()
        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if self.option_type is OptionType.CALL:
            return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)

    def greeks(self) -> Greeks:
        """Analytic first- and second-order sensitivities."""
        if self.time_to_expiry == 0 or self.volatility == 0:
            # Degenerate case: payoff is a deterministic function of the forward,
            # so delta is the discount factor when forward-in-the-money, else 0.
            disc_q = math.exp(-self.dividend_yield * self.time_to_expiry)
            discounted_spot = self.spot * disc_q
            discounted_strike = self.strike * math.exp(-self.rate * self.time_to_expiry)
            if self.option_type is OptionType.CALL:
                delta = disc_q if discounted_spot > discounted_strike else 0.0
            else:
                delta = -disc_q if discounted_strike > discounted_spot else 0.0
            return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

        S, K, T, r, sigma, q = (
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.rate,
            self.volatility,
            self.dividend_yield,
        )
        d1, d2 = self._d1_d2()
        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)
        pdf_d1 = _norm_pdf(d1)
        sqrt_T = math.sqrt(T)

        gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
        vega = S * disc_q * pdf_d1 * sqrt_T

        if self.option_type is OptionType.CALL:
            delta = disc_q * _norm_cdf(d1)
            theta = (
                -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
                - r * K * disc_r * _norm_cdf(d2)
                + q * S * disc_q * _norm_cdf(d1)
            )
            rho = K * T * disc_r * _norm_cdf(d2)
        else:
            delta = disc_q * (_norm_cdf(d1) - 1.0)
            theta = (
                -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
                + r * K * disc_r * _norm_cdf(-d2)
                - q * S * disc_q * _norm_cdf(-d1)
            )
            rho = -K * T * disc_r * _norm_cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def implied_volatility(
        self,
        market_price: float,
        tol: float = 1e-8,
        max_iterations: int = 100,
    ) -> float:
        """Solve for the volatility that reproduces market_price."""
        S, K, T = self.spot, self.strike, self.time_to_expiry
        if T <= 0:
            raise ValueError("cannot solve implied volatility at T=0")

        disc_q = math.exp(-self.dividend_yield * T)
        disc_r = math.exp(-self.rate * T)
        upper_bound = S * disc_q if self.option_type is OptionType.CALL else K * disc_r

        def price_and_vega(sigma: float) -> tuple[float, float]:
            trial = BlackScholes(
                S, K, T, self.rate, sigma, self.option_type, self.dividend_yield
            )
            return trial.price(), trial.greeks().vega

        return solve_implied_vol(
            price_and_vega,
            market_price,
            lower_bound=self._zero_vol_price(),
            upper_bound=upper_bound,
            initial_guess=self.volatility,
            tol=tol,
            max_iterations=max_iterations,
        )

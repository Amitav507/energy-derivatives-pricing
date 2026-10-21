"""Black-76: European options on a forward or futures price.

The commodity market standard. Prices off an observed forward F rather than
spot plus carry, which is why storage cost and convenience yield never have to
be estimated: the exchange's forward quote already contains them. It is also
the only workable choice for power, where the underlying cannot be stored and
no spot-forward arbitrage exists to define a carry at all.

Black-76 is the zero-carry case of Black-Scholes-Merton, so Black76 and
BlackScholes(spot=F, dividend_yield=rate) agree on price and on delta, gamma,
vega and theta. They disagree on rho, and this class is the correct one: see
the rho comment in greeks().
"""

from __future__ import annotations

import math

from ._solver import solve_implied_vol
from .black_scholes import Greeks, OptionType, _norm_cdf, _norm_pdf


class Black76:
    """Prices a single European option on a forward or futures price.

    Args:
        forward: forward or futures price (F), as quoted by the exchange
        strike: strike price (K), in the same units as the forward
        time_to_expiry: time to the OPTION's expiry in years (T). Commodity
            options usually expire before their underlying futures contract.
        rate: continuously-compounded discount rate (r)
        volatility: annualized volatility of the forward (sigma)
        option_type: OptionType.CALL or OptionType.PUT
    """

    def __init__(
        self,
        forward: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
    ):
        if forward <= 0:
            raise ValueError(
                "forward must be positive: Black-76 is lognormal and cannot "
                "represent a negative forward. Power and some spreads do trade "
                "negative, which needs an arithmetic (Bachelier) model instead."
            )
        if strike <= 0:
            raise ValueError("strike must be positive")
        if time_to_expiry < 0:
            raise ValueError("time_to_expiry cannot be negative")
        if volatility < 0:
            raise ValueError("volatility cannot be negative")

        self.forward = float(forward)
        self.strike = float(strike)
        self.time_to_expiry = float(time_to_expiry)
        self.rate = float(rate)
        self.volatility = float(volatility)
        self.option_type = OptionType(option_type)

    @property
    def discount_factor(self) -> float:
        return math.exp(-self.rate * self.time_to_expiry)

    def _d1_d2(self) -> tuple[float, float]:
        F, K, T, sigma = (
            self.forward,
            self.strike,
            self.time_to_expiry,
            self.volatility,
        )
        if T == 0 or sigma == 0:
            raise ZeroDivisionError(
                "d1/d2 undefined at T=0 or sigma=0; use the zero-vol price instead"
            )
        sqrt_T = math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
        return d1, d1 - sigma * sqrt_T

    def _zero_vol_price(self) -> float:
        """Present value in the sigma -> 0 limit.

        The forward is already the expected terminal price, so with no
        volatility the option is worth its discounted intrinsic value on F.
        Price is monotonically increasing in sigma, so this is also the
        no-arbitrage lower bound.
        """
        disc = self.discount_factor
        if self.option_type is OptionType.CALL:
            return disc * max(self.forward - self.strike, 0.0)
        return disc * max(self.strike - self.forward, 0.0)

    def price(self) -> float:
        """Present value of the option."""
        if self.time_to_expiry == 0 or self.volatility == 0:
            return self._zero_vol_price()

        F, K = self.forward, self.strike
        disc = self.discount_factor
        d1, d2 = self._d1_d2()

        if self.option_type is OptionType.CALL:
            return disc * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
        return disc * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

    def greeks(self) -> Greeks:
        """Analytic sensitivities. Delta and gamma are with respect to the
        forward F, not to spot.
        """
        T, r = self.time_to_expiry, self.rate
        disc = self.discount_factor
        price = self.price()

        # Rate enters Black-76 only through the discount factor: carry is zero,
        # so r is absent from d1 and d2 and does not shape the distribution.
        # Raising r can therefore only discount the same payoff more heavily,
        # making rho negative. BlackScholes(spot=F, dividend_yield=rate) reports
        # a positive rho because it differentiates r with q held fixed, which in
        # forward mode is meaningless: there r and q are the same number.
        rho = -T * price

        if T == 0 or self.volatility == 0:
            # Deterministic forward, so only the discounting still moves.
            if self.option_type is OptionType.CALL:
                delta = disc if self.forward > self.strike else 0.0
            else:
                delta = -disc if self.strike > self.forward else 0.0
            return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=r * price, rho=rho)

        F, K, sigma = self.forward, self.strike, self.volatility
        d1, d2 = self._d1_d2()
        pdf_d1 = _norm_pdf(d1)
        sqrt_T = math.sqrt(T)

        if self.option_type is OptionType.CALL:
            delta = disc * _norm_cdf(d1)
        else:
            delta = -disc * _norm_cdf(-d1)

        # Gamma, vega and theta take the same form for calls and puts once the
        # option's own price is substituted, because C - P = disc * (F - K) is
        # independent of sigma and of F's second derivative.
        gamma = disc * pdf_d1 / (F * sigma * sqrt_T)
        vega = F * disc * pdf_d1 * sqrt_T
        theta = r * price - F * disc * pdf_d1 * sigma / (2 * sqrt_T)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def implied_volatility(
        self,
        market_price: float,
        tol: float = 1e-8,
        max_iterations: int = 100,
    ) -> float:
        """Solve for the volatility that reproduces market_price."""
        T = self.time_to_expiry
        if T <= 0:
            raise ValueError("cannot solve implied volatility at T=0")

        disc = self.discount_factor
        # As sigma grows without bound a call tends to disc * F and a put to
        # disc * K.
        upper_bound = (
            disc * self.forward
            if self.option_type is OptionType.CALL
            else disc * self.strike
        )

        def price_and_vega(sigma: float) -> tuple[float, float]:
            trial = Black76(
                self.forward, self.strike, T, self.rate, sigma, self.option_type
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

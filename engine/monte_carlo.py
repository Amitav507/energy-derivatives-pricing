"""Monte Carlo pricing by simulation of the forward.

Two jobs. It cross-validates the closed forms by an independent numerical
route, which is a stronger check than comparing a formula to itself. And it
is the foundation for contracts that have no closed form at all: path
dependent payoffs, average price options, and the volume optionality in a
flexible supply contract.

Under the forward measure F is a martingale, so the simulated process carries
no drift beyond the Itô correction:

    F_T = F_0 * exp(-0.5*sigma^2*T + sigma*sqrt(T)*Z),   Z ~ N(0,1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .black_scholes import OptionType


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    samples: int

    @property
    def confidence_95(self) -> tuple[float, float]:
        half_width = 1.96 * self.standard_error
        return (self.price - half_width, self.price + half_width)

    def __str__(self) -> str:
        lo, hi = self.confidence_95
        return (
            f"{self.price:.6f} +/- {self.standard_error:.6f} "
            f"(95% CI [{lo:.6f}, {hi:.6f}], {self.samples} samples)"
        )


def simulate_terminal_forward(
    forward: float,
    volatility: float,
    time_to_expiry: float,
    paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    """Terminal forward prices under the forward measure.

    With antithetic=True each draw Z is paired with -Z. For a monotone payoff
    the pair is negatively correlated, which cancels part of the sampling
    error at no extra cost in draws.
    """
    if paths < 2:
        raise ValueError("paths must be at least 2")

    if antithetic:
        half = paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(paths)

    drift = -0.5 * volatility * volatility * time_to_expiry
    diffusion = volatility * math.sqrt(time_to_expiry) * z
    return forward * np.exp(drift + diffusion)


def price_european(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
    paths: int = 200_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Price a European option on a forward by simulation.

    Returns the estimate with its standard error, so the result can be
    compared against a closed form on statistical terms rather than against
    an arbitrary tolerance.
    """
    if forward <= 0:
        raise ValueError("forward must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if time_to_expiry < 0:
        raise ValueError("time_to_expiry cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    rng = np.random.default_rng(seed)
    terminal = simulate_terminal_forward(
        forward, volatility, time_to_expiry, paths, rng, antithetic=antithetic
    )

    if OptionType(option_type) is OptionType.CALL:
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)

    discounted = math.exp(-rate * time_to_expiry) * payoff

    if antithetic:
        # Antithetic draws are not independent, so the standard error comes
        # from the pair averages. Using all 2N payoffs would overstate it.
        half = len(discounted) // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])
    else:
        samples = discounted

    return MonteCarloResult(
        price=float(samples.mean()),
        standard_error=float(samples.std(ddof=1) / math.sqrt(len(samples))),
        samples=len(samples),
    )

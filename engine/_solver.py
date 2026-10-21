"""Implied-volatility solver shared by the pricing models.

Bracketed Newton-Raphson: fast through vega where vega is trustworthy, falling
back to bisection where it is not (deep in or out of the money, near expiry).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

LOW_VOL = 1e-6
HIGH_VOL = 5.0


def solve_implied_vol(
    price_and_vega: Callable[[float], Tuple[float, float]],
    market_price: float,
    lower_bound: float,
    upper_bound: float,
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Volatility that reproduces market_price.

    Args:
        price_and_vega: given a volatility, returns (price, vega). Both come
            from one call so the model is not evaluated twice per iteration.
        lower_bound, upper_bound: the no-arbitrage range for this contract.
            The lower bound is the price in the sigma -> 0 limit, since price
            is monotonically increasing in sigma.
    """
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"market_price {market_price} is outside no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    lo, hi = LOW_VOL, HIGH_VOL
    if price_and_vega(hi)[0] < market_price:
        raise ValueError(
            f"implied volatility exceeds the {hi:.0f} search ceiling; check inputs"
        )

    sigma = min(max(initial_guess if initial_guess > 0 else 0.2, lo), hi)

    for _ in range(max_iterations):
        price, vega = price_and_vega(sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Price is monotonically increasing in sigma, so the current point
        # always tightens one side of the bracket.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        candidate: Optional[float] = sigma - diff / vega if vega > 1e-10 else None
        # Bisect when vega is too small to trust, or when the Newton step
        # lands outside the live bracket.
        sigma = (
            candidate
            if candidate is not None and lo < candidate < hi
            else 0.5 * (lo + hi)
        )

    raise RuntimeError(
        f"implied volatility did not converge after {max_iterations} iterations"
    )

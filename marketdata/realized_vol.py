"""Historical realized volatility, used as a sigma proxy when no market IV
(options quotes/vendor vol surface) is available — the Tier 1 fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    prices: pd.Series,
    window: int = 30,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Rolling annualized realized volatility from a close-price series.

    sigma_t = std(log returns over `window` obs) * sqrt(trading_days_per_year)

    252 is the standard equity/most-commodity-futures convention. Physical
    power markets sometimes use 365 (continuous delivery) — pass that in
    explicitly if pricing UK power options.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(trading_days_per_year)


def latest_realized_vol(
    prices: pd.Series,
    window: int = 30,
    trading_days_per_year: int = 252,
) -> float:
    """Most recent rolling realized vol value, as a plain float."""
    series = realized_volatility(prices, window=window, trading_days_per_year=trading_days_per_year)
    series = series.dropna()
    if series.empty:
        raise ValueError(f"not enough observations to compute a {window}-day realized vol")
    return float(series.iloc[-1])

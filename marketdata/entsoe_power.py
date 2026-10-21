"""European day-ahead power prices from the ENTSOE Transparency Platform.

ENTSOE is the *electricity* transparency platform: day-ahead prices,
generation and load for EU bidding zones. It does not publish gas prices.
There is no equivalent free, clean price API for NBP or TTF either, since
ENTSOG covers gas flows and capacity rather than price. Henry Hub via
`marketdata.futures_prices` remains the gas proxy in this project.

Great Britain is not available here, and has no clean free substitute. Since
Brexit GB sits outside the EU internal energy market and its day-ahead prices
are not published on this platform: a GB query returns NoMatchingDataError
rather than an empty series, verified against a live key. Nor is there a
direct public free API for GB historical day-ahead prices elsewhere. Elexon
publishes Market Index Data, but that is an index rather than the day-ahead
price series itself, so it is not a drop-in replacement.

What you get is the day-ahead spot fixing in EUR/MWh, not a forward curve.
Power forwards sit behind vendor terminals (ICE, EEX, Montel, LSEG), so this
feeds realized volatility and distributional work rather than Black-76
pricing, which needs a forward.

Note the granularity. European day-ahead now clears on quarter-hourly market
time units, so expect roughly 96 points per day rather than 24. Annualising
volatility from this series requires the right periods-per-year or the answer
is wrong by a factor of two.

Requires:
    pip install energy-derivatives-pricing[entsoe]
    A free API key from the ENTSOE Transparency Platform
    (https://transparency.entsoe.eu): register, then generate a token under
    your account's "Web API Security Token" settings.
"""

from __future__ import annotations

import pandas as pd

try:
    from entsoe import EntsoePandasClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "entsoe-py is required for marketdata.entsoe_power. Install with: "
        "pip install energy-derivatives-pricing[entsoe]  (or: pip install entsoe-py)"
    ) from exc

# Verified against a live key. GB is deliberately absent, see module docstring.
NETHERLANDS = "NL"
GERMANY_LUXEMBOURG = "DE_LU"
FRANCE = "FR"


def get_day_ahead_prices(
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bidding_zone: str = NETHERLANDS,
) -> pd.Series:
    """Day-ahead power prices (EUR/MWh) for one ENTSOE bidding zone.

    Args:
        api_key: your ENTSOE Transparency Platform token.
        start, end: tz-aware pandas Timestamps, for example
            pd.Timestamp("2026-09-01", tz="Europe/Amsterdam").
        bidding_zone: ENTSOE zone code, such as "NL", "DE_LU" or "FR".

    Returns:
        A price series indexed by market time unit. Values can be negative,
        which is routine in these markets and is the reason a lognormal
        model cannot be used on them.
    """
    client = EntsoePandasClient(api_key=api_key)
    prices = client.query_day_ahead_prices(bidding_zone, start=start, end=end)
    prices.name = "day_ahead_price_eur_mwh"
    return prices


def negative_price_summary(prices: pd.Series) -> dict:
    """How often, and how far, a price series went negative.

    The headline number for why lognormal models fail on power: a lognormal
    assigns probability zero to this region, yet these settlements occurred.
    """
    total = len(prices)
    negative = int((prices < 0).sum())
    zero = int((prices == 0).sum())
    return {
        "observations": int(total),
        "negative_observations": negative,
        "zero_observations": zero,
        "negative_share": float(negative / total) if total else 0.0,
        "zero_share": float(zero / total) if total else 0.0,
        # Both of the above sit outside a lognormal's support, which is
        # strictly positive, so this is the share the model calls impossible.
        "non_positive_share": float((negative + zero) / total) if total else 0.0,
        "minimum": float(prices.min()),
        "mean": float(prices.mean()),
        "maximum": float(prices.max()),
    }

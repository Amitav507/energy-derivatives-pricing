"""Optional: UK day-ahead power prices from the ENTSOE Transparency Platform.

Correction vs. earlier discussion: ENTSOE is the *electricity* transparency
platform (day-ahead prices, generation, load) for EU/UK bidding zones — it
does not publish gas prices. There is no equivalent free, clean price API
for UK gas (NBP) or TTF; ENTSOG (the gas transparency platform) covers flows
and capacity, not price. For gas, Henry Hub via `marketdata.futures_prices`
(ticker NG=F) remains the best free proxy in this project.

Requires:
    pip install black-scholes-engine[entsoe]
    A free API key from https://transparencyplatform.eu (register, then
    generate a token under your account's "Web API Security Token" settings).

This gives you a day-ahead price series (EUR/MWh, spot — not a forward
curve), useful as a realized-vol input for a UK power option demo. It is not
a substitute for actual forward/futures power prices, which sit behind
vendor terminals (ICE, EEX, Montel, LSEG).
"""

from __future__ import annotations

import pandas as pd

try:
    from entsoe import EntsoePandasClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "entsoe-py is required for marketdata.entsoe_power. Install with: "
        "pip install black-scholes-engine[entsoe]  (or: pip install entsoe-py)"
    ) from exc

GB_BIDDING_ZONE = "GB"


def get_uk_day_ahead_prices(
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = GB_BIDDING_ZONE,
) -> pd.Series:
    """UK day-ahead power price series (EUR/MWh) from ENTSOE.

    Args:
        api_key: your ENTSOE Transparency Platform token.
        start, end: tz-aware pandas Timestamps, e.g.
            pd.Timestamp("2025-01-01", tz="Europe/London").
        country_code: ENTSOE bidding-zone code; "GB" for Great Britain.
    """
    client = EntsoePandasClient(api_key=api_key)
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices.name = "day_ahead_price_eur_mwh"
    return prices

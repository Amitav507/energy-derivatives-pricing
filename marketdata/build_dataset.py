"""Orchestrates Tier 1 data collection into pricing-ready inputs.

Pulls futures price history, derives an ATM forward and realized-vol pair per
asset, and attaches a matching risk-free rate, producing one row per asset
that feeds straight into engine.Black76. See notebooks/multi_asset_demo.py.
"""

from __future__ import annotations

import pandas as pd

from .futures_prices import ASSET_UNIVERSE, get_multi_asset_futures, latest_price
from .rates import US_CURVE_SERIES, get_rate_curve, rate_for_tenor
from .realized_vol import latest_realized_vol


def build_pricing_inputs(
    labels: list[str] | None = None,
    period: str = "2y",
    tenor_years: float = 0.25,
    realized_vol_window: int = 30,
) -> pd.DataFrame:
    """One row per asset with everything BlackScholes.price()/.greeks() needs.

    Columns: label, asset_class, description, ticker, as_of, forward_price,
    strike (ATM), tenor_years, rate, realized_vol.

    `strike` defaults to at-the-money (== forward_price); override per-row
    on the returned DataFrame to test other moneyness.
    """
    spec_by_label = {spec.label: spec for spec in ASSET_UNIVERSE}
    labels = list(spec_by_label) if labels is None else labels

    histories = get_multi_asset_futures(labels=labels, period=period)
    rate_curve = get_rate_curve(US_CURVE_SERIES)
    rate = rate_for_tenor(rate_curve, tenor_years)

    rows = []
    for label in labels:
        spec = spec_by_label[label]
        history = histories[label]

        forward = latest_price(history)
        vol = latest_realized_vol(history["close"], window=realized_vol_window)

        rows.append(
            {
                "label": label,
                "asset_class": spec.asset_class,
                "description": spec.description,
                "ticker": spec.ticker,
                "as_of": history.index[-1],
                "forward_price": forward,
                "strike": forward,  # ATM by default
                "tenor_years": tenor_years,
                "rate": rate,
                "realized_vol": vol,
            }
        )

    return pd.DataFrame(rows)

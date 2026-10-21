"""Free, scriptable futures/forward price pulls (Tier 1 data).

Uses Yahoo Finance continuous front-month futures contracts via `yfinance`.
These are a reasonable free proxy for the forward price (F) used as the
underlying in Black-76-style commodity option pricing. They are delayed,
continuous (rolled) contracts, not exact forward-curve points for a specific
delivery month — good enough for building/testing a pricing engine, not for
trading.

UK power has no free Yahoo Finance ticker; `marketdata.entsoe_power` pulls UK
day-ahead power prices from the ENTSOE Transparency Platform instead. UK gas
(NBP) has neither — no free price API exists for it, so Henry Hub (NG=F) is
the gas proxy used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import guard for clearer error
    raise ImportError(
        "yfinance is required for marketdata.futures_prices. Install with: "
        "pip install energy-derivatives-pricing[data]  (or: pip install yfinance)"
    ) from exc


@dataclass(frozen=True)
class AssetSpec:
    label: str          # short key, e.g. "oil_wti"
    ticker: str          # Yahoo Finance ticker
    asset_class: str     # "Gas & Power", "Oil Products", "Metals"
    description: str


# Default asset universe spanning the three sub-asset-classes on the CV framing.
ASSET_UNIVERSE: list[AssetSpec] = [
    AssetSpec("gas_henry_hub", "NG=F", "Gas & Power", "Henry Hub natural gas futures (US proxy for gas)"),
    AssetSpec("oil_wti", "CL=F", "Oil Products", "WTI crude futures"),
    AssetSpec("oil_brent", "BZ=F", "Oil Products", "Brent crude futures"),
    AssetSpec("metals_gold", "GC=F", "Metals", "COMEX gold futures"),
    AssetSpec("metals_copper", "HG=F", "Metals", "COMEX copper futures"),
    AssetSpec("metals_silver", "SI=F", "Metals", "COMEX silver futures"),
]

_UNIVERSE_BY_LABEL = {spec.label: spec for spec in ASSET_UNIVERSE}


def get_futures_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Daily OHLCV history for one Yahoo Finance futures ticker.

    Returns a DataFrame indexed by date with at least a 'close' column.
    """
    history = yf.Ticker(ticker).history(period=period, interval=interval)
    if history.empty:
        raise ValueError(f"no data returned for ticker {ticker!r}; check the symbol")
    history = history.rename(columns=str.lower)
    history.index.name = "date"
    return history[["open", "high", "low", "close", "volume"]]


def get_multi_asset_futures(
    labels: list[str] | None = None,
    period: str = "2y",
) -> dict[str, pd.DataFrame]:
    """Pull price history for a set of asset labels from ASSET_UNIVERSE.

    Args:
        labels: subset of AssetSpec.label to pull; defaults to the full universe.
        period: yfinance period string, e.g. "1y", "2y", "5y".

    Returns:
        {label: DataFrame} with each DataFrame carrying open/high/low/close/volume.
    """
    labels = list(_UNIVERSE_BY_LABEL) if labels is None else labels
    unknown = set(labels) - set(_UNIVERSE_BY_LABEL)
    if unknown:
        raise KeyError(
            f"unknown asset labels {sorted(unknown)}; "
            f"available: {sorted(_UNIVERSE_BY_LABEL)}"
        )

    result: dict[str, pd.DataFrame] = {}
    for label in labels:
        spec = _UNIVERSE_BY_LABEL[label]
        result[label] = get_futures_history(spec.ticker, period=period)
    return result


def latest_price(history: pd.DataFrame) -> float:
    """Most recent close price from a history DataFrame."""
    return float(history["close"].iloc[-1])

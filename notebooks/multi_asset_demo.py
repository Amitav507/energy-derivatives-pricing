"""Tier 1 end-to-end demo: pull real futures data, price an ATM option on
each underlying with the Black-Scholes engine, and print a summary table.

Run in Colab (or any environment with internet + the [data] extra installed):

    !pip install -e .[data]
    !python notebooks/multi_asset_demo.py

Options on commodity futures are priced with Black-76, off the observed
forward rather than spot plus carry.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from marketdata import build_pricing_inputs
from engine import Black76, OptionType


def price_row(row) -> dict:
    engine = Black76(
        forward=row["forward_price"],
        strike=row["strike"],
        time_to_expiry=row["tenor_years"],
        rate=row["rate"],
        volatility=row["realized_vol"],
        option_type=OptionType.CALL,
    )
    g = engine.greeks()
    return {
        "label": row["label"],
        "asset_class": row["asset_class"],
        "forward": round(row["forward_price"], 3),
        "strike (ATM)": round(row["strike"], 3),
        "T (yrs)": row["tenor_years"],
        "rate": round(row["rate"], 4),
        "realized_vol": round(row["realized_vol"], 4),
        "call_price": round(engine.price(), 4),
        "delta": round(g.delta, 4),
        "vega_per_vol_pt": round(g.vega / 100, 4),
    }


def main() -> None:
    inputs = build_pricing_inputs(tenor_years=0.25, realized_vol_window=30)
    results = [price_row(row) for _, row in inputs.iterrows()]

    summary = pd.DataFrame(results)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

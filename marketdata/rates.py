"""Free risk-free rate curves pulled directly from FRED (no API key needed).

FRED publishes a plain CSV export per series at a stable URL, so this avoids
needing an API key just to get a handful of rate points.

Compounding: the pricing engine wants a *continuously compounded* rate, but
FRED publishes these series on their market conventions — Treasury constant
maturity yields are bond-equivalent (semiannual), overnight rates are simple
annualized. They are converted here rather than fed in raw. At 5% rate levels
the error from ignoring this is about 6bp on a semiannual quote and about
12bp on a simple one, neither of which is noise for a rates-sensitive Greek
like rho.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class RateSeries:
    series_id: str
    compounding: str  # "semiannual" (Treasury bond-equivalent) or "simple" (overnight)


# tenor in years -> FRED series
US_CURVE_SERIES: dict[float, RateSeries] = {
    0.0: RateSeries("SOFR", "simple"),          # Secured Overnight Financing Rate
    1 / 12: RateSeries("DGS1MO", "semiannual"),  # 1-Month Treasury CMT
    0.25: RateSeries("DGS3MO", "semiannual"),    # 3-Month Treasury CMT
    0.5: RateSeries("DGS6MO", "semiannual"),     # 6-Month Treasury CMT
    1.0: RateSeries("DGS1", "semiannual"),       # 1-Year Treasury CMT
}

# SONIA has no FRED-mirrored term curve beyond overnight, so a UK curve built
# from this is a flat-rate approximation.
UK_CURVE_SERIES: dict[float, RateSeries] = {
    0.0: RateSeries("IUDSOIA", "simple"),        # Sterling Overnight Index Average
}


def _to_continuous(rate: float, compounding: str) -> float:
    """Convert a quoted rate to its continuously compounded equivalent."""
    if compounding == "semiannual":
        return 2.0 * math.log(1.0 + rate / 2.0)
    if compounding == "simple":
        return math.log(1.0 + rate)
    raise ValueError(f"unknown compounding convention {compounding!r}")


def _fetch_fred_series(series_id: str) -> pd.DataFrame:
    resp = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] != 2:
        raise ValueError(
            f"unexpected FRED CSV layout for {series_id!r}: got columns {list(df.columns)}"
        )
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    # FRED writes "." for missing observations (holidays, non-publication days).
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    if df.empty:
        raise ValueError(f"no usable observations returned for FRED series {series_id!r}")
    return df


def get_rate_curve(curve_series: dict[float, RateSeries] = US_CURVE_SERIES) -> pd.DataFrame:
    """Latest observation for each tenor, as continuously compounded decimals.

    Returns columns [tenor, rate, as_of], sorted by tenor. Series publish on
    different schedules, so `as_of` can differ between rows.
    """
    rows = []
    for tenor, series in curve_series.items():
        observations = _fetch_fred_series(series.series_id)
        last = observations.iloc[-1]
        quoted = last["value"] / 100.0
        rows.append(
            {
                "tenor": tenor,
                "rate": _to_continuous(quoted, series.compounding),
                "as_of": last["date"],
            }
        )
    return pd.DataFrame(rows).sort_values("tenor").reset_index(drop=True)


def rate_for_tenor(curve: pd.DataFrame, tenor_years: float) -> float:
    """Rate at a given tenor, linearly interpolated between bracketing curve
    points, clamped to the nearest endpoint outside the curve's range.
    """
    curve = curve.sort_values("tenor").reset_index(drop=True)
    if len(curve) == 1:
        return float(curve["rate"].iloc[0])

    if tenor_years <= curve["tenor"].iloc[0]:
        return float(curve["rate"].iloc[0])
    if tenor_years >= curve["tenor"].iloc[-1]:
        return float(curve["rate"].iloc[-1])

    upper_idx = int(curve["tenor"].searchsorted(tenor_years))
    lower_idx = upper_idx - 1
    t_lo, r_lo = curve["tenor"].iloc[lower_idx], curve["rate"].iloc[lower_idx]
    t_hi, r_hi = curve["tenor"].iloc[upper_idx], curve["rate"].iloc[upper_idx]
    weight = (tenor_years - t_lo) / (t_hi - t_lo)
    return float(r_lo + weight * (r_hi - r_lo))

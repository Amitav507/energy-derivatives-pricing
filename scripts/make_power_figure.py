"""Generate the negative power prices figure.

Unlike scripts/make_figures.py this one needs live data and an ENTSOE API
key, so it is run deliberately rather than as part of a routine rebuild. The
committed figure carries its own as-of date.

    export ENTSOE_API_KEY=...        # or set it in the environment
    pip install -e ".[entsoe,figures,data]"
    python scripts/make_power_figure.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketdata.entsoe_power import get_day_ahead_prices, negative_price_summary

OUT = Path(__file__).resolve().parent.parent / "figures"

# 60 days of quarter-hourly data per zone. The ENTSOE API returns a 599 on
# requests it considers too large or too frequent, so this stays close to a
# window that is known to work and pauses between zones.
DAYS = 60
PAUSE_BETWEEN_ZONES_SECONDS = 5

INK = "#16202a"
GRID = "#d8dee3"
NEGATIVE = "#9c3658"
ZERO = "#3b4fa8"
BINS = 120
ZONES = {"NL": ("Netherlands", "#0f6b6b"), "DE_LU": ("Germany and Luxembourg", "#8a5a0c")}


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def fetch(api_key):
    end = pd.Timestamp.now(tz="Europe/Amsterdam").normalize()
    start = end - pd.Timedelta(days=DAYS)
    series = {}
    for i, zone in enumerate(ZONES):
        if i:
            time.sleep(PAUSE_BETWEEN_ZONES_SECONDS)
        series[zone] = get_day_ahead_prices(api_key, start, end, bidding_zone=zone)
        print(f"{zone}: {len(series[zone])} points")
    return series, start, end


def lognormal_pdf(x, values):
    """Density of a lognormal fitted to the positive observations only."""
    logs = np.log(values[values > 0])
    mu, sigma = logs.mean(), logs.std(ddof=1)
    return np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))


def build(series, start, end):
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.38, wspace=0.22)

    ax_ts = fig.add_subplot(gs[0, :])
    _style(ax_ts)
    for zone, (label, colour) in ZONES.items():
        ax_ts.plot(series[zone].index, series[zone].values, linewidth=0.5,
                   color=colour, alpha=0.85, label=label)
    ax_ts.axhline(0, color=NEGATIVE, linewidth=1.1)
    ax_ts.set_ylabel("EUR / MWh", fontsize=9)
    ax_ts.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax_ts.set_title(
        "Day-ahead power settles below zero, and a lognormal model cannot\n"
        f"ENTSOE day-ahead fixings, {start:%d %b %Y} to {end:%d %b %Y}, "
        "quarter-hourly",
        color=INK, fontsize=11.5, loc="left",
    )

    for ax, zone in zip((fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])), ZONES):
        _style(ax)
        label, colour = ZONES[zone]
        values = series[zone].values.astype(float)
        stats = negative_price_summary(series[zone])

        # Anchor a bin edge exactly at zero. With arbitrary bins the one
        # straddling zero holds negatives and small positives together, and
        # colouring it red overstates the negative count by half.
        width = (values.max() - values.min()) / BINS
        bin_edges = np.concatenate([
            np.arange(0, values.min() - width, -width)[::-1],
            np.arange(width, values.max() + width, width),
        ])

        # Exact zeros are drawn separately: they are a distinct market state,
        # and binning them with the cheap positives hides them entirely.
        nonzero = values[values != 0]
        zero_count = int((values == 0).sum())

        counts, edges, patches = ax.hist(nonzero, bins=bin_edges, color=colour, alpha=0.75)
        for patch, left in zip(patches, edges[:-1]):
            if left < 0:
                patch.set_facecolor(NEGATIVE)

        if zero_count:
            ax.bar([0.0], [zero_count], width=max(width * 0.8, 1.0), align="center",
                   color=ZERO, zorder=3, label=f"exactly zero ({zero_count})")

        positive = values[values > 0]
        # Start the grid at 1 EUR: the lognormal density goes as 1/x, so a
        # near-zero settlement would spike the curve and squash the histogram.
        grid = np.linspace(max(positive.min(), 1.0), values.max(), 400)
        width = edges[1] - edges[0]
        ax.plot(grid, lognormal_pdf(grid, values) * len(positive) * width,
                color=INK, linewidth=1.5, linestyle="--",
                label="lognormal fitted to\npositive prices")
        ax.set_ylim(0, max(counts.max(), zero_count) * 1.15)

        ax.axvline(0, color=NEGATIVE, linewidth=1.1)
        ax.set_xlabel("EUR / MWh", fontsize=9)
        ax.set_ylabel("Observations", fontsize=9)
        ax.set_title(
            f"{label}\n"
            f"{stats['negative_observations']} below zero, "
            f"{stats['zero_observations']} at exactly zero\n"
            f"{stats['non_positive_share']:.1%} of {stats['observations']} outside the "
            f"lognormal's support. Low {stats['minimum']:.2f}",
            color=INK, fontsize=9.5, loc="left",
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.tight_layout()
    path = OUT / "power-negative-prices.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit(
            "Set ENTSOE_API_KEY in the environment. Do not hardcode it, and do "
            "not commit it."
        )
    OUT.mkdir(exist_ok=True)
    series, start, end = fetch(key)
    for zone in ZONES:
        print(zone, negative_price_summary(series[zone]))
    build(series, start, end)
    print(f"as of {date.today():%Y-%m-%d}")

"""Generate the README figures.

Both figures are deliberately independent of live market data, so they stay
correct indefinitely and demonstrate the engine rather than a snapshot of
prices.

    pip install -e ".[figures]"
    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from engine import Black76, OptionType, price_european

OUT = Path(__file__).resolve().parent.parent / "figures"
INK = "#16202a"
ACCENT = "#0f6b6b"
BAND = "#9fc7c4"
GRID = "#d8dee3"


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


def monte_carlo_convergence():
    """Simulation converging onto the closed form, with its error band."""
    forward, strike, expiry, rate, vol = 100.0, 100.0, 1.0, 0.04, 0.40
    closed = Black76(forward, strike, expiry, rate, vol, OptionType.CALL).price()

    path_counts = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    prices, errors = [], []
    for i, paths in enumerate(path_counts):
        result = price_european(
            forward, strike, expiry, rate, vol,
            OptionType.CALL, paths=paths, seed=1234 + i,
        )
        prices.append(result.price)
        errors.append(result.standard_error)

    prices = np.array(prices)
    errors = np.array(errors)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("white")
    _style(ax)

    ax.fill_between(
        path_counts, prices - 2 * errors, prices + 2 * errors,
        color=BAND, alpha=0.45, label="Monte Carlo, +/- 2 standard errors",
    )
    ax.plot(path_counts, prices, "o-", color=ACCENT, linewidth=1.6, markersize=5,
            label="Monte Carlo estimate")
    ax.axhline(closed, color=INK, linewidth=1.4, linestyle="--",
               label=f"Black-76 closed form = {closed:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Simulated paths")
    ax.set_ylabel("Option value")
    ax.set_title(
        "Simulation converges on the closed form\n"
        f"ATM call, F = {forward:.0f}, T = {expiry:.0f}y, "
        f"sigma = {vol:.0%}, r = {rate:.0%}, antithetic sampling",
        color=INK, fontsize=11, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()

    path = OUT / "mc-convergence.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def greek_profiles():
    """Delta, gamma and vega across strikes for a fixed forward."""
    forward, expiry, rate, vol = 100.0, 0.5, 0.04, 0.40
    strikes = np.linspace(50, 160, 300)

    delta, gamma, vega = [], [], []
    for strike in strikes:
        g = Black76(forward, float(strike), expiry, rate, vol, OptionType.CALL).greeks()
        delta.append(g.delta)
        gamma.append(g.gamma)
        vega.append(g.vega / 100.0)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], delta, "Delta", "per unit of forward"),
        (axes[1], gamma, "Gamma", "per unit squared"),
        (axes[2], vega, "Vega", "per volatility point"),
    ]
    for ax, series, name, units in panels:
        _style(ax)
        ax.plot(strikes, series, color=ACCENT, linewidth=1.8)
        ax.axvline(forward, color=INK, linewidth=1.0, linestyle=":", alpha=0.7)
        ax.set_ylabel(f"{name}\n({units})", fontsize=9)

    axes[0].set_title(
        "Greeks across strikes\n"
        f"Black-76 call, F = {forward:.0f} (dotted), T = {expiry}y, "
        f"sigma = {vol:.0%}, r = {rate:.0%}",
        color=INK, fontsize=11, loc="left",
    )
    axes[2].set_xlabel("Strike")
    fig.tight_layout()

    path = OUT / "greek-profiles.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    monte_carlo_convergence()
    greek_profiles()

"""Charts for the README, from results/ written by analyze.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

R = Path("results")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sharpness(summary: dict) -> None:
    fits = summary["sharpness"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    x = np.linspace(-6, 6, 400)
    for who, color, label in [("umpire", BLUE, "Umpire"), ("catcher", ORANGE, "Catcher"), ("batter", AQUA, "Batter")]:
        f = fits[who]
        ax.plot(x, stats.norm.cdf(-x / f["sigma"]) * 100, color=color, lw=2.2,
                label=f"{label}: {f['sigma']:.2f} in  (95% CI {f['sigma_lo']:.2f}–{f['sigma_hi']:.2f})")
    leg = ax.legend(frameon=True, loc="lower left", title="Blur (σ)", title_fontsize=10.5, alignment="left",
                    labelcolor=INK, fontsize=10.5, facecolor=SURFACE, edgecolor="none", framealpha=1)
    leg.set_zorder(5)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("Inches from each person's own 50/50 point  (left = deeper in the zone)")
    ax.set_ylabel("Says \"strike\" (share of their maximum)")
    fig.suptitle("How sharp is each person's strike zone?", x=0.02, ha="left", fontsize=14, fontweight="bold")
    ax.set_title("Steeper = sharper read of where the pitch crossed. 2026 regular season, ABS zone.",
                 loc="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(R / "zone_sharpness.png")
    plt.close(fig)


def catcher_challenges() -> None:
    emp = pd.read_csv(R / "catcher_challenge_margins.csv")
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    bins = np.arange(-5, 5.01, 0.25)
    won = emp.loc[emp["margin"] <= 0, "margin"]
    lost = emp.loc[emp["margin"] > 0, "margin"]
    ax.hist(won, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label=f"Won ({len(won):,})")
    ax.hist(lost, bins=bins, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label=f"Lost ({len(lost):,})")
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    ax.axvline(0, color=INK2, lw=1)
    ax.text(-0.15, top * 0.98, "← inside the zone", ha="right", va="top", color=INK2, fontsize=10)
    ax.text(0.15, top * 0.98, "outside →", ha="left", va="top", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Where the challenged pitch really crossed: inches from the ABS zone edge")
    ax.set_ylabel("Catcher challenges")
    ax.legend(frameon=False, loc="upper left")
    within = (lost < 1).mean()
    fig.suptitle("Where catchers' challenges landed", x=0.02, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(f"{within:.0%} of lost challenges were within an inch of the zone. "
                 "2026 regular season.", loc="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(R / "catcher_challenges.png")
    plt.close(fig)


if __name__ == "__main__":
    summary = json.loads((R / "summary.json").read_text())
    sharpness(summary)
    catcher_challenges()

"""Shared figure style for the report figures (matplotlib, static PNG).

One look for every chart: light warm surface, hairline horizontal grid, no
frame except the baseline, 2 px round-capped lines, markers with a surface
ring, rounded bar tops, soft confidence bands instead of capped error bars,
text in ink colours (never in a series colour).

Colours follow the entity, never its rank: every method keeps its slot in
every figure (the categorical order below passes the colour-vision checks
of the dataviz method, adjacent pairs; slots 1-3 also pass all pairs).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# one sequential hue (blue), for ordered categories
BLUE_STEPS = {250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281"}

# identities used across the reports
METHOD_COLOR = {"Adaptive@0.5": SLOTS[0], "B0": SLOTS[1], "B0+PA": SLOTS[2], "B1+PA": SLOTS[3],
                "B2+PA": SLOTS[4], "B3+PA": SLOTS[5]}
METHOD_LABEL = {"Adaptive@0.5": "Ours", "B0": "B0", "B0+PA": "B0+PA", "B1+PA": "B1+PA", "B2+PA": "B2+PA",
                "B3+PA": "B3+PA"}
METHOD_ORDER = ["Adaptive@0.5", "B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
TOPO_ORDER = ["nsfnet", "abilene", "geant", "germany50"]
TOPO_LABEL = {"nsfnet": "NSFNET", "abilene": "Abilene", "geant": "GÉANT", "germany50": "Germany50"}
# topologies take their own validated order (adjacent CVD >= 9, normal >= 20), away from "Ours" blue
TOPO_COLOR = {"nsfnet": "#1baf7a", "abilene": "#4a3aa7", "geant": "#eda100", "germany50": "#e34948"}
NEUTRAL = "#a9a79f"  # "best baseline" / reference series, a recessive gray


def tint(color: str, amount: float) -> str:
    """Opaque mix of `color` with the surface (amount 1 = the colour itself).
    Used for the lighter twin of a series instead of transparency, so
    overlapping shapes never darken where they meet."""
    from matplotlib.colors import to_hex, to_rgb

    c, s = to_rgb(color), to_rgb(SURFACE)
    return to_hex(tuple(amount * ci + (1 - amount) * si for ci, si in zip(c, s)))


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round", "lines.markersize": 7,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
        "figure.titlesize": 13, "figure.titleweight": "semibold",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def bars(ax, x, heights, width: float, color: str, alpha: float = 1.0, label: str | None = None,
         hatch: str | None = None, edgecolor: str | None = None):
    """Plain bars with a surface gap between neighbours; call `round_tops(fig)`
    after the layout is final to round their data ends."""
    return ax.bar(x, heights, width, color=color, alpha=alpha, label=label, hatch=hatch,
                  edgecolor=edgecolor or SURFACE, linewidth=1.2 if edgecolor is None else 0.8, zorder=2)


def round_tops(fig, radius_px: float = 6.0) -> None:
    """Round the top corners of every bar in the figure (square base kept on
    the baseline). Needs final axes limits and size, so call just before save."""
    fig.canvas.draw()
    for ax in fig.axes:
        sx = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
        sy = abs(ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
        if sx == 0 or sy == 0:
            continue
        for container in ax.containers:
            for rect in getattr(container, "patches", []):
                x, y, w, h = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
                if h <= 0 or w <= 0 or not rect.get_visible():
                    continue
                rx = min(radius_px / sx, w / 2)
                ry = min(radius_px / sy, h)
                style = dict(facecolor=rect.get_facecolor(), edgecolor="none", zorder=rect.get_zorder(),
                             hatch=rect.get_hatch())
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                            mutation_aspect=ry / rx, **style))
                ax.add_patch(plt.Rectangle((x, y), w, h - ry, **style))  # square base
                rect.set_visible(False)


def line(ax, x, y, color: str, label: str | None = None, marker: str = "o", ls: str = "-",
         lw: float = 2.0, alpha: float = 1.0, ms: float = 7, zorder: int = 3):
    """Line with markers that wear a surface ring (so crossings stay legible)."""
    return ax.plot(x, y, ls=ls, lw=lw, color=color, marker=marker, ms=ms, mfc=color, mec=SURFACE,
                   mew=1.8, alpha=alpha, label=label, zorder=zorder)[0]


def band(ax, x, lo, hi, color: str, alpha: float = 0.14):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def label_end(ax, x, y, text: str, color: str = INK_2, dx: float = 6, weight: str = "semibold"):
    """Direct label at the end of a line, in ink (the marker beside it carries identity)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=color, fontweight=weight)


def subtitle(ax, text: str) -> None:
    """Muted line under the title (the title is lifted to make room)."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
    ax.text(0, 1.012, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom", ha="left")


def save(fig, path: str | Path) -> None:
    round_tops(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

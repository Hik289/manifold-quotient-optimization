#!/usr/bin/env python3
"""Draw the QuotientOpt method figure with native vector text and matrix tiles.

The composition follows the ManifoldFlow update figure's two-row algebraic
storyboard. All equations and step ordering here are specific to QuotientOpt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
STEM = "fig_method_main"
NAVY = "#173A62"
BLUE = "#236AC8"
BLUE_DARK = "#11549D"
BLUE_LIGHT = "#A8D1F5"
BLUE_FILL = "#F0F7FF"
GREEN = "#138B5B"
GREEN_DARK = "#08714A"
GREEN_LIGHT = "#B4E4C9"
GREEN_FILL = "#F0FBF5"
ORANGE = "#D97720"
ORANGE_DARK = "#AE4F14"
ORANGE_LIGHT = "#F8D3A7"
ORANGE_FILL = "#FFF7EF"
PURPLE = "#7441B9"
PURPLE_LIGHT = "#D4B5EE"
PURPLE_FILL = "#FAF5FF"
INK = "#18364D"
MUTED = "#557084"
WHITE = "#FFFFFF"
LINE = "#B5CEDD"

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def rounded(ax, x, y, w, h, edge, face, radius=0.85, lw=1.15):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
        joinstyle="round", zorder=1,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=BLUE, lw=1.55, ms=13, z=9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0", zorder=z,
    ))


def label(ax, x, y, text, size=10.0, color=INK, weight=None, ha="left", va="center", z=10):
    return ax.text(
        x, y, text, fontsize=size, color=color, weight=weight,
        ha=ha, va=va, zorder=z,
    )


def math(ax, x, y, text, size=10.6, color=INK, ha="center", va="center"):
    return label(ax, x, y, text, size=size, color=color, ha=ha, va=va)


def tile(ax, x, y, rows, cols, cell, base, light, edge, name=None,
         dim=False, name_y=1.12, dim_y=1.45):
    """Draw a true rows-by-columns matrix icon; x/y are lower-left."""
    for i in range(rows):
        for j in range(cols):
            shade = base if (i + 2*j) % 3 == 0 else light
            ax.add_patch(Rectangle(
                (x + j*cell, y + (rows-i-1)*cell), cell, cell,
                facecolor=shade, edgecolor=edge, linewidth=0.40, zorder=6,
            ))
    ax.add_patch(Rectangle(
        (x, y), cols*cell, rows*cell,
        facecolor="none", edgecolor=edge, linewidth=0.75, zorder=7,
    ))
    cx = x + cols*cell/2
    if name:
        math(ax, cx, y + rows*cell + name_y, name, size=10.0, color=INK)
    if dim:
        label(ax, cx, y-dim_y, f"{rows}×{cols}", size=7.5,
              color=MUTED, ha="center")
    return cx


def color_tile(ax, x, y, rows, cols, cell, kind, **kwargs):
    color = {
        "B": ("#56A4EE", BLUE_LIGHT, BLUE_DARK),
        "A": ("#62BD89", GREEN_LIGHT, GREEN_DARK),
        "X": ("#A16ADD", PURPLE_LIGHT, PURPLE),
        "G": ("#E99A49", ORANGE_LIGHT, ORANGE_DARK),
    }[kind]
    return tile(ax, x, y, rows, cols, cell, *color, **kwargs)


def strip(ax, x, y, w, h, color, face=WHITE):
    return rounded(ax, x, y, w, h, color, face, radius=.54, lw=.60)


def top_card(ax, x, w, title, subtitle, edge, face):
    rounded(ax, x, 45.1, w, 35.2, edge, face, radius=.85, lw=1.12)
    label(ax, x+2.2, 77.5, title, size=15.0, color=edge, weight="bold")
    if subtitle:
        label(ax, x+w-1.5, 77.4, subtitle, size=9.0,
              color=edge, ha="right")
    strip(ax, x+1.4, 47.0, w-2.8, 10.7, edge)


def pair_icon(ax, x, y, kind, title):
    """A 5×3 and a 3×5 tile form one factor-pair icon."""
    color_tile(ax, x, y, 5, 3, .86, kind)
    color_tile(ax, x+3.6, y+1.72, 3, 5, .86, kind)
    math(ax, x+3.9, y+6.25, title, size=9.0)


def mini_pair(ax, x, y, kind, title):
    """Compact factor-pair tile, retaining 5×3 and 3×5 dimensions."""
    color_tile(ax, x, y, 5, 3, .68, kind)
    color_tile(ax, x+2.55, y+1.36, 3, 5, .68, kind)
    math(ax, x+3.0, y+4.35, title, size=8.5)


def draw_top(ax):
    panels = [
        (2.0, 36.6, "Factorization", "gauge orbit", BLUE, BLUE_FILL),
        (39.7, 36.8, "Balance", "product-preserving", GREEN, GREEN_FILL),
        (77.6, 39.0, "Chain-rule gradient", None, ORANGE_DARK, ORANGE_FILL),
        (117.7, 40.8, "Horizontal split", None, BLUE, BLUE_FILL),
        (159.6, 31.4, "Recompose", None, PURPLE, PURPLE_FILL),
    ]
    for args in panels:
        top_card(ax, *args)

    # 1. A rectangular low-rank product, and its GL(r) orbit.
    color_tile(ax, 5.3, 63.3, 5, 3, 1.42, "B", name=r"$B_t$", dim=True)
    color_tile(ax, 17.1, 65.0, 3, 5, 1.42, "A", name=r"$A_t$", dim=True)
    color_tile(ax, 29.0, 63.3, 5, 5, 1.42, "X", name=r"$X_t$", dim=True)
    math(ax, 14.3, 67.0, r"$\times$", size=15.0)
    math(ax, 26.3, 67.0, r"$=$", size=15.0)
    math(ax, 20.3, 54.2, r"$B_tA_t=(B_tQ)(Q^{-1}A_t)$", size=10.0)
    math(ax, 20.3, 49.7, r"$Q\in\mathrm{GL}(r)$: same $X_t$", size=9.5,
         color=BLUE_DARK)

    # 2. The balanced product retains both factor shapes and X.
    color_tile(ax, 42.9, 63.3, 5, 3, 1.42, "B", name=r"$\bar B_t$", dim=True)
    color_tile(ax, 54.7, 65.0, 3, 5, 1.42, "A", name=r"$\bar A_t$", dim=True)
    color_tile(ax, 66.6, 63.3, 5, 5, 1.42, "X", name=r"$X_t$", dim=True)
    math(ax, 51.9, 67.0, r"$\times$", size=15.0)
    math(ax, 63.9, 67.0, r"$=$", size=15.0)
    math(ax, 58.1, 54.2,
         r"$(\bar B_t,\bar A_t)=(B_tQ_t,Q_t^{-1}A_t)$", size=9.2)
    math(ax, 58.1, 49.7,
         r"$\bar B_t^\top\bar B_t=\bar A_t\bar A_t^\top=H_t$", size=9.4,
         color=GREEN_DARK)

    # 3. The product gradient branches into two chain-rule factor gradients.
    color_tile(ax, 80.7, 63.3, 5, 5, 1.42, "G", name=r"$G_t$", dim=True)
    color_tile(ax, 96.7, 63.3, 5, 3, 1.42, "G", name=r"$g_B$", dim=True)
    color_tile(ax, 108.0, 65.0, 3, 5, 1.42, "G", name=r"$g_A$", dim=True)
    arrow(ax, (89.0, 68.0), (95.3, 68.0), color=ORANGE, lw=1.4, ms=10)
    math(ax, 104.0, 68.0, r"$,$", size=15.0)
    math(ax, 97.1, 54.2,
         r"$g_B=G_t\bar A_t^\top,\quad g_A=\bar B_t^\top G_t$", size=9.5)
    math(ax, 97.1, 49.7,
         r"$\bar B_t^\top g_B-g_A\bar A_t^\top=0$", size=9.2,
         color=ORANGE_DARK)

    # 4. Proposed memory = horizontal memory + vertical factor motion.
    pair_icon(ax, 120.4, 63.2, "G", r"$U_t$")
    pair_icon(ax, 133.8, 63.2, "B", r"$M_t$")
    pair_icon(ax, 147.2, 63.2, "A", r"$v(\Omega_t)$")
    math(ax, 131.5, 67.5, r"$=$", size=14.0)
    math(ax, 144.8, 67.5, r"$+$", size=14.0)
    math(ax, 138.0, 54.2,
         r"$U_t=\beta\widetilde M_{t-1}+(1-\beta)g_t$", size=9.5)
    math(ax, 138.0, 49.7,
         r"$M_t=U_t-v(\Omega_t)$", size=9.8, color=BLUE_DARK)

    # 5. Both factors are stepped before recomposition.
    color_tile(ax, 162.3, 63.3, 5, 3, 1.28, "B", name=r"$B_{t+1}$", dim=True)
    color_tile(ax, 171.8, 64.6, 3, 5, 1.28, "A", name=r"$A_{t+1}$", dim=True)
    color_tile(ax, 183.2, 63.3, 5, 5, 1.28, "X", name=r"$X_{t+1}$", dim=True)
    math(ax, 169.0, 66.5, r"$\times$", size=14.0)
    math(ax, 180.0, 66.5, r"$=$", size=14.0)
    math(ax, 175.3, 54.2,
         r"$B_{t+1}=\bar B_t-\eta_tM_B$", size=9.2)
    math(ax, 175.3, 49.7,
         r"$A_{t+1}=\bar A_t-\eta_tM_A$", size=9.2,
         color=PURPLE)

    for x0, x1 in ((37.9, 41.1), (75.8, 79.0), (115.8, 119.0),
                   (157.7, 160.9)):
        arrow(ax, (x0, 63.1), (x1, 63.1), color=BLUE, lw=2.35, ms=15)


def bottom_card(ax, x, w, title):
    rounded(ax, x, 10.0, w, 32.2, GREEN, GREEN_FILL, radius=.80, lw=1.10)
    label(ax, x+2.2, 39.2, title, size=14.4, color=GREEN_DARK, weight="bold")


def draw_bottom(ax):
    bottom_card(ax, 3.0, 42.3, "State transport")
    bottom_card(ax, 49.1, 83.1, "Persistent memory and normal solve")
    bottom_card(ax, 137.0, 53.0, "Finite product step")

    # State transport is completed before differentiation at the new balance.
    strip(ax, 4.8, 23.1, 38.7, 13.3, GREEN, WHITE)
    label(ax, 6.1, 34.4, "Inputs: $Q_t$, previous memory", size=9.3,
          color=BLUE_DARK, weight="bold")
    color_tile(ax, 7.2, 25.0, 5, 3, 1.08, "B", name=r"$M_{B,t-1}$")
    color_tile(ax, 18.5, 27.2, 3, 3, 1.08, "A", name=r"$Q_t$")
    color_tile(ax, 32.1, 25.0, 5, 3, 1.08, "A", name=r"$\widetilde M_B$")
    math(ax, 15.7, 27.9, r"$\times$", size=13.0)
    math(ax, 28.0, 27.9, r"$=$", size=13.0)
    strip(ax, 4.8, 12.0, 38.7, 9.5, GREEN, WHITE)
    math(ax, 24.0, 18.5,
         r"$\widetilde M_B=M_{B,t-1}Q_t$", size=10.3)
    math(ax, 24.0, 14.5,
         r"$\widetilde M_A=Q_t^{-1}M_{A,t-1}$", size=10.3)

    # The left half mirrors ManifoldFlow's persistent-state panel.
    strip(ax, 50.8, 12.0, 31.3, 24.4, GREEN, WHITE)
    label(ax, 52.2, 34.3, "Transported EMA", size=10.3,
          color=BLUE_DARK, weight="bold")
    mini_pair(ax, 53.0, 24.3, "A", r"$\beta\widetilde M$")
    mini_pair(ax, 63.4, 24.3, "G", r"$(1-\beta)g$")
    mini_pair(ax, 74.0, 24.3, "G", r"$U$")
    math(ax, 61.2, 26.4, r"$+$", size=13.0)
    math(ax, 71.8, 26.4, r"$=$", size=13.0)
    math(ax, 66.5, 18.2,
         r"$U=\beta\widetilde M+(1-\beta)g$", size=9.6)
    label(ax, 66.5, 14.3, "one state in the new gauge", size=8.5,
          color=GREEN_DARK, ha="center")

    # The right half explicitly computes the small normal solve.
    strip(ax, 83.2, 12.0, 47.2, 24.4, GREEN, WHITE)
    label(ax, 84.5, 34.3, "Vertical load, solve, horizontal lift", size=10.0,
          color=BLUE_DARK, weight="bold")
    for x, title, kind in ((84.7, "Load", "G"), (100.3, "Solve", "A"),
                           (115.9, "Correct", "B")):
        rounded(ax, x, 22.4, 13.8, 9.8, GREEN_LIGHT, "#F8FFFA",
                radius=.43, lw=.54)
        label(ax, x+1.2, 29.8, title, size=8.7,
              color=GREEN_DARK, weight="bold")
        if title == "Correct":
            color_tile(ax, x+3.0, 24.1, 5, 3, .76, "B")
            color_tile(ax, x+6.5, 25.6, 3, 5, .76, "A")
        else:
            color_tile(ax, x+4.7, 24.0, 3, 3, 1.26, kind)
    arrow(ax, (98.6, 27.0), (100.0, 27.0), color=GREEN, lw=1.20, ms=9)
    arrow(ax, (114.2, 27.0), (115.6, 27.0), color=GREEN, lw=1.20, ms=9)
    math(ax, 91.5, 21.0, r"$C_t$", size=9.0)
    math(ax, 107.1, 21.0, r"$\Omega_t$", size=9.0)
    math(ax, 122.8, 21.0, r"$M_t$", size=9.0)
    math(ax, 106.8, 19.2,
         r"$C_t=\bar B_t^\top U_B-U_A\bar A_t^\top$", size=8.6)
    math(ax, 106.8, 16.45,
         r"$(H_t+\varepsilon I)\Omega_t+\Omega_t(H_t+\varepsilon I)=C_t$",
         size=8.6)
    label(ax, 106.8, 13.25,
          "Exact at ε = 0; horizontal residual 2εΩ when ε > 0.",
          size=8.4, color=GREEN_DARK, ha="center")

    # Finite factor changes induce a tangent term plus an eta-squared term.
    strip(ax, 138.8, 22.9, 49.4, 13.5, GREEN, WHITE)
    label(ax, 140.1, 34.4, "After the projected factor step", size=9.3,
          color=BLUE_DARK, weight="bold")
    color_tile(ax, 141.1, 26.0, 5, 3, 1.0, "B", name=r"$\bar B_t$")
    color_tile(ax, 154.9, 26.0, 5, 3, 1.0, "B", name=r"$B_{t+1}$")
    color_tile(ax, 167.4, 28.0, 3, 5, 1.0, "A", name=r"$\bar A_t$")
    color_tile(ax, 180.7, 28.0, 3, 5, 1.0, "A", name=r"$A_{t+1}$")
    arrow(ax, (146.0, 28.7), (153.3, 28.7), color=BLUE, lw=1.40, ms=10)
    arrow(ax, (173.5, 29.5), (179.5, 29.5), color=GREEN, lw=1.40, ms=10)
    strip(ax, 138.8, 12.0, 49.4, 9.4, GREEN, WHITE)
    math(ax, 163.5, 18.3,
         r"$X_{t+1}=X_t-\eta_t(M_B\bar A_t+\bar B_tM_A)$", size=9.0)
    math(ax, 163.5, 14.4,
         r"$\qquad\qquad+\eta_t^2M_BM_A$", size=9.1)

    # Separate green computations and the compact next-iteration perimeter.
    arrow(ax, (57.5, 45.0), (24.1, 42.5), color=GREEN, lw=1.65, ms=12)
    arrow(ax, (137.8, 45.0), (106.8, 42.5), color=GREEN, lw=1.65, ms=12)
    arrow(ax, (171.9, 45.0), (164.0, 42.5), color=GREEN, lw=1.65, ms=12)
    arrow(ax, (45.5, 25.2), (48.8, 25.2), color=GREEN, lw=1.75, ms=12)
    arrow(ax, (132.4, 25.2), (136.7, 25.2), color=GREEN, lw=1.75, ms=12)


def feedback_and_legend(ax):
    # Purple loop ties the updated product and carried memory to the next step.
    ax.plot([190.7, 192.3, 192.3, 1.1, 1.1],
            [56.0, 56.0, 7.5, 7.5, 55.5],
            color=PURPLE, linewidth=1.6, zorder=2, solid_capstyle="round")
    arrow(ax, (1.1, 51.8), (2.0, 51.8), color=PURPLE,
          lw=1.75, ms=12)
    label(ax, 96.5, 7.48,
          r"Next iteration: $(B_t,A_t,M_{t-1})\leftarrow(B_{t+1},A_{t+1},M_t)$",
          size=9.2, color=PURPLE, weight="bold", ha="center",
          z=12).set_bbox(dict(facecolor="white", edgecolor="none", pad=2.1))
    legend = [
        (34.0, BLUE, "Factor / tangent"),
        (74.0, GREEN, "Balance / geometry"),
        (119.0, ORANGE, "Gradient / load"),
        (157.0, PURPLE, "Product"),
    ]
    for x, color, text in legend:
        rounded(ax, x, 1.5, 2.8, 2.8, color, color, radius=.25, lw=.6)
        label(ax, x+3.9, 2.9, text, size=8.8, color=NAVY)


def draw():
    fig = plt.figure(figsize=(14.2, 5.98), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 193.3)
    ax.set_ylim(0, 81.4)
    ax.axis("off")
    draw_top(ax)
    draw_bottom(ax)
    feedback_and_legend(ax)
    for ext, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 350})):
        path = OUT / f"{STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=.01, **options)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw()

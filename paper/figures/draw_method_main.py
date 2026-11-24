#!/usr/bin/env python3
"""Draw the QuotientOpt method figure with selectable vector text.

Run ``python paper/figures/draw_method_main.py`` from any directory. The PDF
is the paper asset; SVG keeps editable text, and PNG is a high-resolution
preview. All labels are Matplotlib text artists rather than raster images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent
STEM = "fig_method_main"

NAVY = "#173B61"
MUTED = "#50677E"
RULE = "#C7D6E3"
BLUE = "#2877B7"
TEAL = "#128378"
PURPLE = "#7B68B7"
GREEN = "#18845E"
ORANGE = "#BA7629"
PALE_BLUE = "#EFF7FD"
PALE_TEAL = "#EEF8F6"
PALE_PURPLE = "#F5F2FB"
PALE_GREEN = "#F0F8F2"
PALE_ORANGE = "#FFF8EE"


plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.0,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def rounded(ax, x, y, w, h, face, edge, lw=1.0, radius=1.6, dash=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=dash or "solid",
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, xy0, xy1, color=NAVY, lw=1.7, head=10, dash="solid"):
    ax.add_patch(
        FancyArrowPatch(
            xy0, xy1, arrowstyle="-|>", mutation_scale=head,
            color=color, linewidth=lw, linestyle=dash,
            shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0",
        )
    )


def badge(ax, cx, cy, number, color):
    ax.add_patch(Circle((cx, cy), 1.8, facecolor=color, edgecolor="white", lw=.7))
    ax.text(cx, cy-.08, str(number), color="white", ha="center", va="center",
            fontsize=8.6, weight="bold")


def header(ax, x, y, number, title, color):
    badge(ax, x+3.05, y, number, color)
    ax.text(x+5.55, y, title, color=NAVY, fontsize=9.5,
            ha="left", va="center", weight="bold")


def note(ax, x, y, text, color=MUTED, size=7.5, weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha="left", va="center",
            weight=weight)


def math(ax, x, y, expression, size=8.4, color=NAVY, **kwargs):
    ax.text(x, y, expression, color=color, fontsize=size, ha="left", va="center",
            **kwargs)


def draw():
    fig = plt.figure(figsize=(8.0, 4.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # The hierarchy follows figurex/4: a white canvas, restrained pastel
    # stages, numbered discs, strong directional arrows, and a lower readout.
    ax.text(2.4, 49.6, "QuotientOpt", fontsize=13.3, color=NAVY,
            weight="bold", ha="left", va="center")
    ax.text(97.5, 49.6, "One update in balanced factor coordinates",
            fontsize=8.6, color=MUTED, ha="right", va="center")
    ax.plot([2.4, 97.5], [47.35, 47.35], color=RULE, lw=.9)

    x = [2.4, 21.9, 41.4, 60.9, 80.4]
    y, w, h = 24.4, 17.2, 20.8
    fills = [PALE_PURPLE, PALE_BLUE, PALE_TEAL, PALE_GREEN, PALE_ORANGE]
    accents = [PURPLE, BLUE, TEAL, GREEN, ORANGE]
    titles = ["Gauge orbit", "Balance", "Carry state", "Project", "Factor step"]
    for i, (xi, fill, accent, title) in enumerate(zip(x, fills, accents, titles), 1):
        rounded(ax, xi, y, w, h, fill, accent, lw=1.05)
        header(ax, xi, 42.15, i, title, accent)
        if i < 5:
            arrow(ax, (xi+w+.28, 34.85), (x[i]-.34, 34.85),
                  color=NAVY, lw=1.75, head=10)

    # 1: full GL(r) orbit, rather than a misleading unique factorization.
    math(ax, x[0]+1.45, 37.95, r"$X_t=B_tA_t$", size=9.1)
    math(ax, x[0]+1.45, 34.10, r"$(B_t,A_t)\sim$", size=8.5)
    math(ax, x[0]+1.45, 31.15, r"$(B_tQ,Q^{-1}A_t)$", size=8.2)
    math(ax, x[0]+1.45, 28.15, r"$Q\in\mathrm{GL}(r)$", size=7.8)
    note(ax, x[0]+1.45, 25.95, "same X across the orbit", size=7.1,
         color=PURPLE)

    # 2: balancing preserves the product, but leaves an orthogonal gauge.
    math(ax, x[1]+1.25, 38.10, r"$(\bar B_t,\bar A_t)=$", size=8.25)
    math(ax, x[1]+1.25, 35.15, r"$(B_tQ_t,Q_t^{-1}A_t)$", size=7.8)
    math(ax, x[1]+1.25, 31.30, r"$\bar B_t^\top\bar B_t=$", size=8.35)
    math(ax, x[1]+1.25, 28.45, r"$\bar A_t\bar A_t^\top=H_t$", size=8.35)
    note(ax, x[1]+1.25, 25.95, "product unchanged", color=BLUE, size=7.3, weight="bold")

    # 3: both persistent state and gradient are represented at the balance.
    math(ax, x[2]+1.1, 38.25, r"$\widetilde M_B=M_{B,t-1}Q_t$", size=7.45)
    math(ax, x[2]+1.1, 35.30, r"$\widetilde M_A=Q_t^{-1}M_{A,t-1}$", size=7.45)
    ax.plot([x[2]+1.1, x[2]+16.1], [33.55, 33.55], color="#BBD8D4", lw=.75)
    math(ax, x[2]+1.1, 30.85, r"$g_B=G_t\bar A_t^\top$", size=8.1)
    math(ax, x[2]+1.1, 27.95, r"$g_A=\bar B_t^\top G_t$", size=8.1)
    note(ax, x[2]+1.1, 25.95, "G_t: product gradient", color=TEAL, size=7.1)

    # 4: the moment is projected; exactness depends on damping.
    math(ax, x[3]+1.15, 37.95, r"$U_t=\beta\widetilde M$", size=8.5)
    math(ax, x[3]+1.15, 34.75, r"$\qquad +(1-\beta)g$", size=8.5)
    arrow(ax, (x[3]+8.6, 32.15), (x[3]+8.6, 29.8), color=GREEN, lw=1.25, head=8)
    math(ax, x[3]+1.15, 27.75, r"$M_t=U_t-v(\Omega_t)$", size=8.35)
    note(ax, x[3]+1.15, 25.95, "gauge-reduced memory", color=GREEN, size=7.15,
         weight="bold")

    # 5: update the factors, then form the next product.
    math(ax, x[4]+1.0, 38.10, r"$(B_{t+1},A_{t+1})$", size=8.15)
    math(ax, x[4]+1.0, 34.95, r"$=(\bar B_t,\bar A_t)-\eta_t M_t$", size=7.65)
    arrow(ax, (x[4]+8.6, 32.1), (x[4]+8.6, 29.95), color=ORANGE, lw=1.2, head=8)
    math(ax, x[4]+1.0, 27.75, r"$X_{t+1}=B_{t+1}A_{t+1}$", size=8.2)
    note(ax, x[4]+1.0, 25.95, "rebalance next step", color=ORANGE, size=7.1)

    # Three explanatory readouts below the main flow; the central one spells
    # out the actual solve instead of treating projection as a black box.
    lower_y, lower_h = 3.3, 18.3
    lower = [(2.4, 29.7, PALE_PURPLE, PURPLE),
             (34.3, 31.6, PALE_GREEN, GREEN),
             (68.1, 29.5, PALE_ORANGE, ORANGE)]
    for xi, wi, fill, edge in lower:
        rounded(ax, xi, lower_y, wi, lower_h, fill, edge, lw=.9, radius=1.4)
    for cx in (11.0, 69.5, 88.9):
        ax.plot([cx, cx], [24.35, 21.65], color=RULE, lw=.9, linestyle=(0, (2, 2)))

    ax.text(4.0, 18.65, "Geometry", fontsize=9.1, color=NAVY, weight="bold")
    note(ax, 4.0, 15.90, "Vertical factor motion leaves X fixed:", size=7.55)
    math(ax, 4.0, 13.15, r"$v(\Omega)=(\bar B\Omega,-\Omega\bar A)$", size=8.05)
    math(ax, 4.0, 10.20, r"$D\pi[v(\Omega)]=0$", size=8.3)
    note(ax, 4.0, 6.15, "Balanced reps differ by an orthogonal gauge.",
         size=7.05, color=PURPLE)

    ax.text(36.0, 18.65, "Sylvester connection", fontsize=9.1,
            color=NAVY, weight="bold")
    math(ax, 36.0, 15.75, r"$C_t=\bar B_t^\top U_B-U_A\bar A_t^\top$", size=7.85)
    math(ax, 36.0, 12.80, r"$(H_t+\varepsilon I)\Omega_t+\Omega_t(H_t+\varepsilon I)=C_t$",
         size=7.45)
    math(ax, 36.0, 9.85, r"$M_t=(U_B-\bar B_t\Omega_t,\;U_A+\Omega_t\bar A_t)$",
         size=7.45)
    note(ax, 36.0, 6.15, "Exact at ε = 0; residual 2εΩ when ε > 0.",
         size=7.15, color=GREEN)

    ax.text(69.8, 18.65, "Product-level step", fontsize=9.1,
            color=NAVY, weight="bold")
    math(ax, 69.8, 15.75, r"$X_{t+1}=X_t-\eta_t(M_B\bar A_t+\bar B_tM_A)$",
         size=7.55)
    math(ax, 69.8, 12.70, r"$\quad\quad\quad\;+\eta_t^2M_BM_A$", size=7.65)
    note(ax, 69.8, 8.95, "tangent step + quadratic correction", size=7.15,
         color=ORANGE, weight="bold")
    note(ax, 69.8, 6.15, "Gauge correction affects the O(η²) term.",
         size=7.05)

    for ext, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 360})):
        path = OUTPUT / f"{STEM}.{ext}"
        fig.savefig(path, **options)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw()

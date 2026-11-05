"""Render the QuotientOpt method overview as a high-resolution PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "quotientopt_method_overview.png"
INK, TEXT, MUTED = "#14243d", "#273a52", "#687b91"
LINE = "#dce5ef"
BLUE, TEAL, ORANGE, PURPLE = "#3570b7", "#0b877a", "#db792f", "#7360ac"


def tint(color, opacity):
    return tuple(1 - (1 - channel) * opacity for channel in to_rgb(color))


def rounded(ax, x, y, w, h, edge=LINE, face="white", radius=.011, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    ))


def arrow(ax, x1, y1, x2, y2, color=BLUE, width=1.6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, shrinkA=0, shrinkB=0,
    ))


def card(ax, x, y, w, h, number, title, subtitle, color):
    rounded(ax, x, y, w, h)
    ax.add_patch(Rectangle((x, y+h-.009), w, .009, color=color, linewidth=0))
    ax.text(x+.014, y+h-.045, f"{number:02d}", color=color,
            fontsize=15, fontweight="bold", va="center")
    ax.text(x+.043, y+h-.045, title, color=INK,
            fontsize=15, fontweight="bold", va="center")
    ax.text(x+.014, y+h-.084, subtitle, color=MUTED, fontsize=9.8, va="center")


def matrix(ax, x, y, w, h, rows, cols, label, color, seed=0):
    ax.text(x+w/2, y+h+.013, label, fontsize=13, color=INK,
            ha="center", va="bottom")
    gap = .0015
    for row in range(rows):
        for col in range(cols):
            strength = .29 + .09*((2*row+3*col+seed) % 5)
            ax.add_patch(Rectangle(
                (x+col*w/cols+gap, y+(rows-row-1)*h/rows+gap),
                w/cols-2*gap, h/rows-2*gap,
                facecolor=tint(color, strength), edgecolor=color, linewidth=.32,
            ))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.1))


def strip(ax, x, y, w, h, lines, color, font=12):
    rounded(ax, x, y, w, h, edge=tint(color, .36),
            face=tint(color, .08), radius=.008, lw=.8)
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        ax.text(x+w/2, y+h*(1-(i+.5)/len(lines)), line,
                ha="center", va="center", fontsize=font, color=TEXT)


def chip(ax, x, y, w, text, color):
    rounded(ax, x, y, w, .052, edge=tint(color, .50),
            face=tint(color, .08), radius=.008, lw=.9)
    ax.text(x+w/2, y+.026, text, ha="center", va="center",
            fontsize=10.5, color=color, fontweight="bold")


def factor_row(ax, x, symbols, color):
    b, a, product = symbols
    matrix(ax, x+.020, .628, .031, .118, 4, 2, b, color)
    ax.text(x+.059, .686, r"$\times$", ha="center", va="center", fontsize=18, color=MUTED)
    matrix(ax, x+.071, .648, .050, .078, 2, 4, a, color, 2)
    ax.text(x+.129, .686, r"$=$", ha="center", va="center", fontsize=18, color=MUTED)
    matrix(ax, x+.140, .628, .030, .118, 4, 4, product, PURPLE, 3)


def top(ax):
    xs = [.013, .211, .409, .607, .805]
    w, y, h = .182, .511, .410
    specs = [
        ("Factorization", "current factors", BLUE),
        ("Balance", "same product; transport state", TEAL),
        ("Differentiate", "at balanced coordinates", ORANGE),
        ("Horizontal memory", "accumulate, then project", PURPLE),
        ("Factor step", "next low-rank product", BLUE),
    ]
    for i, (x, (title, subtitle, color)) in enumerate(zip(xs, specs), 1):
        card(ax, x, y, w, h, i, title, subtitle, color)
    for i in range(4):
        arrow(ax, xs[i]+w+.003, .686, xs[i+1]-.006, .686,
              color=specs[i][2], width=2)

    x = xs[0]
    factor_row(ax, x, (r"$B_t$", r"$A_t$", r"$X_t$"), BLUE)
    strip(ax, x+.012, .525, .158, .076,
          [r"$X_t=B_tA_t$", r"$(B,A)\sim(BQ,Q^{-1}A)$"], BLUE, 11.5)

    x = xs[1]
    factor_row(ax, x, (r"$\bar B_t$", r"$\bar A_t$", r"$X_t$"), TEAL)
    strip(ax, x+.012, .525, .158, .076,
          [r"$\bar B_t\bar A_t=X_t$",
           r"$\bar B_t^T\bar B_t=\bar A_t\bar A_t^T=H_t$"], TEAL, 10.7)

    x = xs[2]
    matrix(ax, x+.019, .628, .038, .118, 4, 4, r"$G_{X,t}$", ORANGE)
    arrow(ax, x+.062, .687, x+.085, .687, ORANGE)
    matrix(ax, x+.092, .628, .029, .118, 4, 2, r"$g_B$", ORANGE, 2)
    ax.text(x+.129, .686, ",", fontsize=18, color=MUTED, va="center")
    matrix(ax, x+.143, .648, .029, .078, 2, 4, r"$g_A$", ORANGE, 3)
    strip(ax, x+.012, .525, .158, .076,
          [r"$g_B=G_{X,t}\bar A_t^T$",
           r"$g_A=\bar B_t^TG_{X,t}$"], ORANGE, 11.1)

    x = xs[3]
    chip(ax, x+.014, .732, .069, "average", PURPLE)
    arrow(ax, x+.089, .758, x+.103, .758, PURPLE)
    chip(ax, x+.109, .732, .059, "project", PURPLE)
    ax.text(x+.091, .686, r"$\widetilde M_{t-1}\;\longrightarrow\;U_t\;\longrightarrow\;M_t$",
            ha="center", fontsize=13.5, color=INK)
    strip(ax, x+.012, .525, .158, .076,
          [r"$U_t=\beta\widetilde M_{t-1}+(1-\beta)g_t$",
           r"$M_t=\Pi_H(U_t)$"], PURPLE, 10.6)

    x = xs[4]
    factor_row(ax, x, (r"$B_{t+1}$", r"$A_{t+1}$", r"$X_{t+1}$"), BLUE)
    strip(ax, x+.012, .525, .158, .076,
          [r"$B_{t+1}=\bar B_t-\eta_tM_B$",
           r"$A_{t+1}=\bar A_t-\eta_tM_A$"], BLUE, 10.9)


def bottom(ax):
    xs = [.013, .343, .673]
    y, h, w = .071, .399, .314
    specs = [
        ("A", "Gauge symmetry", "many coordinates, one product", BLUE),
        ("B", "Product-preserving balance", "a regular representative", TEAL),
        ("C", "Horizontal connection", "remove pure gauge motion", PURPLE),
    ]
    for x, (number, title, subtitle, color) in zip(xs, specs):
        rounded(ax, x, y, w, h)
        ax.add_patch(Rectangle((x, y+h-.009), w, .009, color=color, linewidth=0))
        ax.text(x+.014, y+h-.050, number, color=color,
                fontsize=14, fontweight="bold", va="center")
        ax.text(x+.041, y+h-.050, title, color=INK,
                fontsize=15, fontweight="bold", va="center")
        ax.text(x+.014, y+h-.087, subtitle, color=MUTED,
                fontsize=10, va="center")

    x = xs[0]
    strip(ax, x+.018, .285, .278, .071,
          r"$(B,A)\mapsto(BQ,Q^{-1}A)$", BLUE, 15)
    arrow(ax, x+.155, .271, x+.155, .229, BLUE)
    strip(ax, x+.018, .145, .278, .071,
          r"$(BQ)(Q^{-1}A)=BA=X$", BLUE, 15)
    ax.text(x+.157, .104, r"$Q\in GL(r)$", ha="center", fontsize=11, color=MUTED)

    x = xs[1]
    chip(ax, x+.020, .304, .084, "QR factors", TEAL)
    arrow(ax, x+.113, .329, x+.133, .329, TEAL)
    chip(ax, x+.141, .304, .084, "small SVD", TEAL)
    arrow(ax, x+.234, .329, x+.251, .329, TEAL)
    ax.text(x+.280, .329, r"$\bar B,\bar A$", ha="center",
            va="center", fontsize=12, color=TEAL)
    strip(ax, x+.018, .172, .278, .087,
          [r"$\bar B\bar A=X$",
           r"$\bar B^T\bar B=\bar A\bar A^T=H$"], TEAL, 13)
    ax.text(x+.157, .115, "The product is unchanged by balancing.",
            fontsize=10.5, color=MUTED, ha="center")

    x = xs[2]
    strip(ax, x+.018, .310, .278, .054,
          r"$\widetilde M_B=M_BQ,\qquad\widetilde M_A=Q^{-1}M_A$",
          PURPLE, 12.1)
    strip(ax, x+.018, .220, .278, .062,
          r"$(H+\varepsilon I)\Omega+\Omega(H+\varepsilon I)=\bar B^TU_B-U_A\bar A^T$",
          PURPLE, 10.5)
    strip(ax, x+.018, .119, .278, .067,
          [r"$M_B=U_B-\bar B\Omega$",
           r"$M_A=U_A+\Omega\bar A$"], PURPLE, 11.6)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "stix",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(18.2, 8.2), dpi=205)
    fig.subplots_adjust(left=.010, right=.990, bottom=.026, top=.970)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.013, .984, "QuotientOpt", fontsize=24,
            color=INK, fontweight="bold", va="top")
    ax.text(.154, .981, "Balanced quotient updates for low-rank factors",
            fontsize=19, color=TEXT, va="top")
    top(ax)
    bottom(ax)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=205)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()

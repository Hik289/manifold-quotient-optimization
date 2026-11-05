"""Draw the repository's compact quotient-optimization intuition diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "quotientopt_intuition.png"
NAVY = "#17233b"
MUTED = "#50617b"
BLUE = "#3168ad"
TEAL = "#087f72"
ORANGE = "#cf7431"
LIGHT = "#f5f8fc"


def box(ax, x, title, subtitle, color, number):
    width = .302
    ax.add_patch(FancyBboxPatch((x, .075), width, .85,
                                boxstyle="round,pad=0.012,rounding_size=0.020",
                                facecolor="white", edgecolor="#dce4ed", linewidth=1.5))
    ax.add_patch(Rectangle((x, .914), width, .012, facecolor=color, edgecolor="none"))
    ax.text(x + .024, .865, f"{number:02d}", color=color, fontsize=15, fontweight="bold", va="center")
    ax.text(x + .024, .796, title, color=NAVY, fontsize=18, fontweight="bold", va="center")
    ax.text(x + .024, .739, subtitle, color=MUTED, fontsize=11.5, va="center")
    return width


def chip(ax, x, y, label, color, width=.100, height=.103):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                facecolor=LIGHT, edgecolor=color, linewidth=1.35))
    ax.text(x + width / 2, y + height / 2, label, fontsize=15,
            color=color, va="center", ha="center")


def arrow(ax, x1, y1, x2, y2, color="#72849b", lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=lw, color=color))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(15, 4.75), dpi=220)
    fig.subplots_adjust(left=.017, right=.986, bottom=.055, top=.92)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.01, .995, "QuotientOpt  /  the update lives on the product",
            fontsize=17, color=NAVY, fontweight="bold", va="top")

    x1, x2, x3 = .010, .350, .690
    box(ax, x1, "Gauge orbit", "Different factors, same low-rank matrix", BLUE, 1)
    chip(ax, x1 + .030, .530, r"$B$", BLUE, .070)
    ax.text(x1 + .111, .584, r"$\times$", fontsize=20, color=MUTED, ha="center", va="center")
    chip(ax, x1 + .139, .530, r"$A$", BLUE, .070)
    arrow(ax, x1 + .219, .582, x1 + .263, .582, BLUE)
    ax.text(x1 + .264, .582, r"$X$", fontsize=21, color=NAVY, va="center")
    ax.text(x1 + .150, .448, r"$Q\in GL(r)$", fontsize=14, color=MUTED, ha="center")
    chip(ax, x1 + .030, .279, r"$BQ$", BLUE, .090)
    ax.text(x1 + .132, .335, r"$\times$", fontsize=20, color=MUTED, ha="center", va="center")
    chip(ax, x1 + .155, .279, r"$Q^{-1}A$", BLUE, .096)
    ax.text(x1 + .154, .175, r"$BA=(BQ)(Q^{-1}A)$", fontsize=16.5, color=NAVY, ha="center")

    box(ax, x2, "Balance the factors", "Choose a well-conditioned representative", TEAL, 2)
    chip(ax, x2 + .047, .517, r"$\bar B$", TEAL, .087)
    ax.text(x2 + .152, .572, r"$\times$", fontsize=20, color=MUTED, ha="center", va="center")
    chip(ax, x2 + .174, .517, r"$\bar A$", TEAL, .087)
    ax.text(x2 + .151, .400, r"$\bar B\bar A=X$", fontsize=21, color=NAVY, ha="center")
    ax.add_patch(FancyBboxPatch((x2 + .024, .202), .254, .107,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                facecolor="#ecf7f3", edgecolor="none"))
    ax.text(x2 + .151, .253, r"$\bar B^T\bar B"
            r"=\bar A\bar A^T=H$", fontsize=17, color=TEAL,
            ha="center", va="center")
    ax.text(x2 + .151, .150, "Same product; balanced coordinates", fontsize=11.5, color=MUTED, ha="center")

    box(ax, x3, "Project the update", "Remove motion along the gauge orbit", ORANGE, 3)
    ax.text(x3 + .151, .585, r"$U=(U_B,U_A)$", fontsize=20, color=NAVY, ha="center")
    arrow(ax, x3 + .151, .540, x3 + .151, .468, ORANGE)
    ax.add_patch(FancyBboxPatch((x3 + .025, .335), .252, .110,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                facecolor="#fff3e9", edgecolor="none"))
    ax.text(x3 + .151, .387,
            r"$H\Omega+\Omega H=\bar B^TU_B-U_A\bar A^T$",
            fontsize=14.5, color=ORANGE, ha="center", va="center")
    ax.text(x3 + .151, .260, r"$M=\Pi_H(U)$", fontsize=20,
            color=NAVY, ha="center")
    ax.text(x3 + .151, .165,
            r"$X_{t+1}=(\bar B-\eta M_B)(\bar A-\eta M_A)$",
            fontsize=16.2, color=NAVY, ha="center")

    arrow(ax, x1 + .316, .497, x2 - .013, .497, BLUE, 2.1)
    arrow(ax, x2 + .316, .497, x3 - .013, .497, TEAL, 2.1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()

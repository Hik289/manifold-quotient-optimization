"""Regenerate the two principal paper loss figures from bundled plot data."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
METHODS = {"adamw": "AdamW", "sgdm": "SGD-M", "balance_only": "Balance-only", "quotientopt": "QuotientOpt"}
COLORS = {"adamw": "#596777", "sgdm": "#d79135", "balance_only": "#4d8cb5", "quotientopt": "#15803d"}
STYLES = {"adamw": "-", "sgdm": "--", "balance_only": ":", "quotientopt": "-"}


def save(fig, basename: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{basename}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main_core(data: dict) -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.25))
    fig.subplots_adjust(left=.085, right=.99, bottom=.10, top=.88, hspace=.56, wspace=.28)
    for panel, index in (("cora", 0), ("citeseer", 1), ("sdf", 2), ("matrix", 3)):
        ax = axes.flat[index]
        for method, label in METHODS.items():
            records = [r for r in data["core"] if r["panel"] == panel and r["method"] == method]
            assert len(records) == (3 if panel in ("cora", "sdf") else 1)
            for r in records:
                assert len(r["epoch"]) == len(r["loss"])
            if len(records) == 3:
                for r in records:
                    ax.plot(r["epoch"], r["loss"], color=COLORS[method], alpha=.22, lw=.55)
                y = np.mean([r["loss"] for r in records], axis=0)
                x = records[0]["epoch"]
            else:
                x, y = records[0]["epoch"], records[0]["loss"]
            ax.plot(x, y, color=COLORS[method], ls=STYLES[method],
                    lw=1.8 if method == "quotientopt" else 1.15, label=label)
        if panel == "cora":
            ax.set_title("(a) Cora / GCN · three initializations", loc="left", fontsize=9)
            ax.set_ylabel("Validation cross-entropy")
        elif panel == "citeseer":
            ax.set_title("(b) CiteSeer / GCN · initialization 42", loc="left", fontsize=9)
            ax.set_ylabel("Validation cross-entropy")
        elif panel == "sdf":
            ax.set_title("(c) SDF / SIREN · three initializations", loc="left", fontsize=9)
            ax.set_ylabel("Validation MSE")
            ax.set_yscale("log")
        else:
            ax.set_title("(d) Matrix completion · initialization 42", loc="left", fontsize=9)
            ax.set_ylabel("Training MSE")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.16, lw=.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", fontsize=9)
    save(fig, "icml_core_losses")


def poisson(data: dict) -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    fig.subplots_adjust(left=.16, right=.98, bottom=.17, top=.95)
    for method, label in METHODS.items():
        records = sorted((r for r in data["poisson"] if r["method"] == method), key=lambda r: r["seed"])
        assert [r["seed"] for r in records] == [42, 43, 44]
        curves = np.asarray([r["loss"] for r in records])
        x = np.asarray(records[0]["epoch"])
        for curve in curves:
            ax.plot(x, curve, color=COLORS[method], alpha=.16, lw=.4)
        ax.plot(x, curves.mean(axis=0), color=COLORS[method],
                lw=1.8 if method == "quotientopt" else 1.15, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.grid(alpha=.15, lw=.5)
    ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="upper right")
    save(fig, "poisson_initialization_loss")


if __name__ == "__main__":
    with gzip.open(Path(__file__).with_name("main_plot_data.json.gz"), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    FIG.mkdir(parents=True, exist_ok=True)
    main_core(payload)
    poisson(payload)
    print("Rebuilt main loss figures in", FIG)

"""Audit and typeset the locked Poisson initialization follow-up."""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent
PROJECT = FULL.parent.parent
FIG = PROJECT / "paper/figures"
sys.path.insert(0, str(ROOT))
from grid import run_id

METHODS = ("adamw", "sgdm", "balance_only", "quotientopt")
NAMES = {"adamw": "AdamW", "sgdm": "SGD-M", "balance_only": "Balance-only", "quotientopt": "QuotientOpt"}
COLORS = {"adamw": "#596777", "sgdm": "#d79135", "balance_only": "#4d8cb5", "quotientopt": "#15803d"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(config, folder, old=False):
    result_path = folder / "result.json"
    trajectory_path = folder / "trajectory.jsonl"
    result = read(result_path)
    trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    assert result["status"] == "complete"
    assert result["completed_epochs"] == len(trajectory) == 2000
    assert [row["epoch"] for row in trajectory] == list(range(1, 2001))
    assert all(math.isfinite(row["validation"]["mse"]) for row in trajectory)
    best_index = min(range(2000), key=lambda i: trajectory[i]["validation"]["mse"])
    assert result["best_epoch"] == best_index + 1
    if old:
        selected = trajectory[best_index]["test"]
        assert selected is not None
    else:
        selected = result["selected_test"]
        assert result["selection_key"] == "mse"
        assert math.isclose(result["best_validation"], trajectory[best_index]["validation"]["mse"], rel_tol=0, abs_tol=1e-10)
    assert all(math.isfinite(selected[key]) for key in ("relative_l2", "mse"))
    assert all(math.isfinite(result["final_test"][key]) for key in ("relative_l2", "mse"))
    return {
        "initialization_seed": config.get("initialization_seed", 42),
        "method": config["method"],
        "run_id": result["run_id"],
        "best_epoch": result["best_epoch"],
        "best_validation_mse": trajectory[best_index]["validation"]["mse"],
        "selected_test": selected,
        "final_test": result["final_test"],
        "validation_curve": [row["validation"]["mse"] for row in trajectory],
        "result_path": str(result_path.relative_to(PROJECT)),
        "result_sha256": digest(result_path),
        "trajectory_path": str(trajectory_path.relative_to(PROJECT)),
        "trajectory_sha256": digest(trajectory_path),
    }


def main():
    plan_path = ROOT / "plan.json"
    plan = read(plan_path)
    verification = read(ROOT / "verification.json")
    assert verification["status"] == "verified" and verification["completed_runs"] == 8
    data_path = ROOT / "data_snapshot/poisson_manufactured.pt"
    assert digest(data_path) == verification["data_sha256"]
    rows = []
    for method, rid in plan["original_reference_run_ids"].items():
        config = dict(family="sciml", dataset="poisson", model="pinn", rank=16,
                      epochs=2000, method=method, lr=.001, beta=.9, eps=1e-6,
                      balance_interval=1, seed=42, protocol="full_proposal_v1")
        rows.append(record(config, FULL / "runs" / rid, old=True))
    assert len(plan["configs"]) == 8
    for config in plan["configs"]:
        rows.append(record(config, ROOT / "runs" / run_id(config)))
    assert {(row["initialization_seed"], row["method"]) for row in rows} == {
        (seed, method) for seed in (42, 43, 44) for method in METHODS
    }
    by_method = {method: [next(row for row in rows if row["method"] == method and row["initialization_seed"] == seed)
                          for seed in (42, 43, 44)] for method in METHODS}
    summary = {}
    for method, group in by_method.items():
        values = [row["selected_test"]["relative_l2"] for row in group]
        summary[method] = {"values": values, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                           "final_mean": statistics.mean(row["final_test"]["relative_l2"] for row in group)}
    quotient = summary["quotientopt"]
    for method in METHODS[:-1]:
        summary[method]["quotientopt_pair_wins"] = sum(
            quotient["values"][i] < summary[method]["values"][i] for i in range(3)
        )
        summary[method]["quotientopt_mean_reduction_percent"] = 100 * (1 - quotient["mean"] / summary[method]["mean"])

    # One-column ICML table; all three paired outcomes and the descriptive mean.
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Frozen Poisson/PINN initialization check.} Validation-MSE-selected held-out relative $L_2$ error, rank 16, 2000 epochs, learning rate $10^{-3}$. Data and shuffle seed are fixed at 42; columns vary only model initialization. Mean $\pm$ sample SD is conditional on that fixed data. All four methods use the same frozen settings.}",
             r"\label{tab:poisson-initializations}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & 42 & 43 & 44 & Mean $\pm$ SD\\", r"\midrule"]
    for method in METHODS:
        s = summary[method]
        name = NAMES[method]
        if method == "quotientopt":
            name = r"\textbf{QuotientOpt}"
        vals = [f"{value:.5f}" for value in s["values"]]
        mean = f"{s['mean']:.5f} $\\pm$ {s['sample_sd']:.5f}"
        if method == "quotientopt":
            mean = r"\cellcolor{oursgoodbg}\textbf{\boldmath " + mean + "}"
        prefix = r"\rowcolor{oursbg}" if method == "quotientopt" else ""
        lines.append(prefix + name + " & " + " & ".join(vals) + " & " + mean + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "poisson_initialization_main.tex").write_text("\n".join(lines) + "\n")

    # Complete per-seed record, including final-horizon endpoints.
    lines = [r"\begin{longtable}{@{}lrlrrr@{}}",
             r"\caption{\textbf{All Poisson/PINN initialization outcomes.} Best epoch minimizes validation MSE; the selected test and final-horizon columns are held-out relative $L_2$ errors. Seed 42 is the pre-existing original-grid run; seeds 43 and 44 are the separately predeclared follow-up.}\label{tab:poisson-initialization-all}\\",
             r"\toprule Init. & Method & Best epoch & Best val. MSE & Selected test & Final test \\ \midrule\endfirsthead",
             r"\toprule Init. & Method & Best epoch & Best val. MSE & Selected test & Final test \\ \midrule\endhead"]
    for seed in (42, 43, 44):
        for method in METHODS:
            row = next(x for x in rows if x["initialization_seed"] == seed and x["method"] == method)
            prefix = r"\rowcolor{oursbg}" if method == "quotientopt" else ""
            lines.append(prefix + f"{seed} & {NAMES[method]} & {row['best_epoch']} & "
                         + f"${row['best_validation_mse']:.3g}$ & {row['selected_test']['relative_l2']:.5f} & "
                         + f"{row['final_test']['relative_l2']:.5f}" + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (FIG / "poisson_initialization_all.tex").write_text("\n".join(lines) + "\n")

    # Every logged validation epoch is shown; means have no smoothing.
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    fig.subplots_adjust(left=.16, right=.98, bottom=.17, top=.95)
    for method in METHODS:
        curves = np.asarray([row["validation_curve"] for row in by_method[method]])
        x = np.arange(1, 2001)
        for curve in curves:
            ax.plot(x, curve, color=COLORS[method], alpha=.16, lw=.4)
        ax.plot(x, curves.mean(axis=0), color=COLORS[method], lw=1.8 if method == "quotientopt" else 1.15,
                label=NAMES[method])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.grid(alpha=.15, lw=.5)
    ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"poisson_initialization_loss.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    audit = {"plan_sha256": digest(plan_path), "data_sha256": digest(data_path),
             "checkpoint_verification_sha256": digest(ROOT / "verification.json"),
             "selection": "earliest minimum validation MSE; selected held-out relative L2",
             "source_code_sha256": {name: digest(ROOT / name) for name in ("run.py", "geometry_torch.py", "models_full.py", "datasets_full.py")},
             "summary": summary,
             "rows": [{k: v for k, v in row.items() if k != "validation_curve"} for row in rows]}
    (ROOT / "paper_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"runs": len(rows), "new_full_runs": 8, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

"""Render the fixed 2026-11-04 RU01 completion snapshot for the paper."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
SNAPSHOT = ROOT / "snapshot/ru01_nov04_1305"
PROGRESS = SNAPSHOT / "progress.json"
OUTPUT = PROJECT / "paper/figures/ru01_latest_results.tex"
RUN_IDS = (
    "43632a1379ac3574d954",
    "10c3c6140b5c55f9a42f",
    "3f74c6abca90f2ceb611",
    "1b847e4bdc2bb1c8689f",
    "58c10d4e5d936896dec6",
    "1d29f17836956ca1f422",
    "7be5c412357e7f8c437f",
    "3c52bc342072fdad8ff7",
    "bf86cadfa8499e450fdf",
    "9851c8775f21bbc8da2f",
    "83ddd2fce115b89fe2ba",
    "712d3d56ac627ed583ef",
    "b1eaa79ebc79ee28f9a8",
    "c7dadd8c705933bb35aa",
    "296ffa7071c268ebf5df",
    "870aa9b5c1c92cb3e335",
    "29b3a4e3b3b025a2f4d7",
    "e478bab358014bac48b4",
    "77cd0385f01f1a5bc021",
)


def metric(record):
    family = record["config"]["family"]
    result = record["final_test"]
    if family in ("graph", "vision"):
        return "Accuracy (\\%)", f"{100 * result['accuracy']:.2f}"
    if family == "lm":
        return "NLL", f"{result['nll']:.3f}"
    if family == "inr":
        return "MSE", f"{result['mse']:.5f}"
    if family == "sciml":
        return "Relative $L_2$", f"{result['relative_l2']:.4f}"
    return "Relative error", f"{result['relative_error']:.5f}"


def label(config):
    name = {
        "cifar10": "CIFAR-10",
        "wikitext2": "WikiText-2",
        "poisson": "Poisson",
        "image": "Synthetic image",
        "cora": "Cora",
        "sensing": "Matrix sensing",
        "synthetic": "Deep linear chain",
    }[config["dataset"]]
    model = {
        "mlp": "MLP",
        "10m": "10M decoder",
        "25m": "25M decoder",
        "deeponet": "DeepONet",
        "siren": "SIREN",
        "gcn": "GCN",
        "factorization": "factorization",
        "chain": "depth-2 chain",
    }[config["model"]]
    placement = f", {config['placement']}" if "placement" in config else ""
    return f"{name} / {model}{placement}, $r={config['rank']}$"


def main():
    progress_bytes = PROGRESS.read_bytes()
    progress = json.loads(progress_bytes)
    assert progress["complete_runs"] == 158
    assert progress["complete_main_runs"] == 115
    assert progress["matched_four_method_cohorts"] == 26
    assert progress["queue_status"] == {"complete": 158, "failed": 5, "paused_scope": 4}
    records_by_id = {record["run_id"]: record for record in progress["results"]}
    assert len(RUN_IDS) == len(set(RUN_IDS)) == 19
    method_names = {"adamw": "AdamW", "sgdm": "SGD-M", "balance_only": "Balance-only", "quotientopt": "QuotientOpt"}
    rows = []
    audit = []
    for run_id in RUN_IDS:
        path = SNAPSHOT / "runs" / run_id / "result.json"
        raw = path.read_bytes()
        record = json.loads(raw)
        assert record == records_by_id[run_id]
        config = record["config"]
        assert record["status"] == "complete"
        assert record["completed_epochs"] == config["epochs"]
        assert config["lr"] == 0.001 and config["seed"] == 42
        unit, value = metric(record)
        method = method_names[config["method"]]
        if config["method"] == "quotientopt":
            method = r"\cellcolor{oursbg}\textbf{QuotientOpt}"
        rows.append(f"{label(config)} & {method} & {unit} & {value}" + r"\\")
        audit.append({"run_id": run_id, "source": str(path.relative_to(PROJECT)),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "config": config, "final_test": record["final_test"]})
    lines = [
        r"\begin{longtable}{@{}p{.42\linewidth}p{.18\linewidth}p{.19\linewidth}r@{}}",
        r"\caption{\textbf{Nineteen full-horizon RU01 completions since the 139-run snapshot.} Fixed-setting, final-epoch held-out endpoints at learning rate $10^{-3}$ and seed 42; no test-based row selection. All nineteen are listed, including unfavorable outcomes. CIFAR-10/MLP rank 32 and rank-16 matrix sensing are complete four-method cohorts led by AdamW. Cora/GCN hidden rank 4 has only an AdamW--QuotientOpt pair at this snapshot.}\label{tab:ru01-latest}\\",
        r"\toprule Task / setting & Method & Metric & Final test \\ \midrule\endfirsthead",
        r"\toprule Task / setting & Method & Metric & Final test \\ \midrule\endhead",
        *rows,
        r"\bottomrule",
        r"\end{longtable}",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n")
    (SNAPSHOT / "paper_audit.json").write_text(json.dumps({
        "snapshot_updated_unix": progress["updated_unix"],
        "progress_sha256": hashlib.sha256(progress_bytes).hexdigest(),
        "complete_runs": progress["complete_runs"],
        "complete_main_runs": progress["complete_main_runs"],
        "matched_four_method_cohorts": progress["matched_four_method_cohorts"],
        "queue_status": progress["queue_status"],
        "rows": audit,
    }, indent=2) + "\n")
    print(f"Wrote {len(rows)} verified rows to {OUTPUT}")


if __name__ == "__main__":
    main()

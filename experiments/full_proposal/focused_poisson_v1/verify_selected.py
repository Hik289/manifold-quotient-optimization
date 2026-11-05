"""Independent checkpoint re-evaluation for the eight full-horizon runs."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from grid import ROOT, run_id
from run import Task, seed


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    rows = []
    initial_by_seed = {43: [], 44: []}
    for config in plan["configs"]:
        rid = run_id(config)
        folder = ROOT / "runs" / rid
        result = json.loads((folder / "result.json").read_text())
        trajectory = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
        assert result["status"] == "complete"
        assert result["completed_epochs"] == len(trajectory) == 2000
        assert [row["epoch"] for row in trajectory] == list(range(1, 2001))
        minimum = min(range(2000), key=lambda i: trajectory[i]["validation"]["mse"])
        assert minimum + 1 == result["best_epoch"]
        assert math.isclose(trajectory[minimum]["validation"]["mse"], result["best_validation"], rel_tol=0, abs_tol=1e-10)
        seed()
        task = Task(config, "cpu")
        initial = torch.load(folder / "checkpoint_0.pt", map_location="cpu", weights_only=False)
        task.net.load_state_dict(initial["model"])
        initial_value = task.evaluate("val")["relative_l2"]
        initial_by_seed[config["initialization_seed"]].append(initial_value)
        selected = torch.load(folder / "best_validation_state.pt", map_location="cpu", weights_only=False)
        assert selected["epoch"] == result["best_epoch"]
        task.net.load_state_dict(selected["model"])
        replay = task.evaluate("test")
        for key in ("relative_l2", "mse"):
            assert math.isclose(replay[key], result["selected_test"][key], rel_tol=1e-4, abs_tol=1e-7), (rid, key, replay[key], result["selected_test"][key])
        rows.append({
            "run_id": rid,
            "initialization_seed": config["initialization_seed"],
            "method": config["method"],
            "best_epoch": result["best_epoch"],
            "initial_validation_relative_l2": initial_value,
            "selected_test_replay": replay,
            "selected_test_recorded": result["selected_test"],
            "result_sha256": hashlib.sha256((folder / "result.json").read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256((folder / "trajectory.jsonl").read_bytes()).hexdigest(),
            "best_state_sha256": hashlib.sha256((folder / "best_validation_state.pt").read_bytes()).hexdigest(),
        })
    for seed_value, values in initial_by_seed.items():
        assert len(values) == 4
        assert max(values) - min(values) < 1e-5, (seed_value, values)
    data = Path(os.environ["QUOTIENT_DATA"]) / "poisson_manufactured.pt"
    audit = {
        "status": "verified",
        "completed_runs": len(rows),
        "epochs_per_run": 2000,
        "initial_validation_by_seed": initial_by_seed,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "rows": rows,
    }
    (ROOT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "completed_runs": len(rows), "data_sha256": audit["data_sha256"]}))


if __name__ == "__main__":
    main()

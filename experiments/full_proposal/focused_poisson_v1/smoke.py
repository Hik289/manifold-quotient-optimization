"""Short, separate implementation check; never enters formal results."""
import json
import math
from pathlib import Path

from grid import ROOT
from run import Task, run, seed


plan = json.loads((ROOT / "plan.json").read_text())
configs = [c for c in plan["configs"] if c["initialization_seed"] == 43]
assert len(configs) == 4
initial = []
for config in configs:
    seed()
    task = Task(config, "cpu")
    initial.append(task.evaluate("val")["relative_l2"])
assert max(initial) - min(initial) < 1e-5, initial
rows = []
for config in configs:
    result = run(config, output=ROOT / "smoke", device="cpu", smoke_epochs=2)
    assert result["status"] == "smoke_only"
    assert result["completed_epochs"] == 2
    assert math.isfinite(result["final_test"]["relative_l2"])
    assert result["metadata"]["data_and_shuffle_seed"] == 42
    assert result["metadata"]["initialization_seed"] == 43
    rows.append({"method": config["method"], "initial_validation_relative_l2": initial[len(rows)],
                 "smoke_final_test_relative_l2": result["final_test"]["relative_l2"]})
(ROOT / "smoke_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows))

"""Reversibly pause the exhaustive workers after the focus decision.

The active jobs keep their checkpoint and trajectory trees.  The separate
focused Poisson queue is deliberately untouched.
"""

import json
import os
import signal
import sqlite3
import time
from pathlib import Path

from grid import ROOT


PROCESS_FILES = {
    "worker_0.pid": "worker.py",
    "worker_1.pid": "worker.py",
    "worker_2.pid": "worker.py",
    "priority_vision.pid": "priority_vision.py",
}


def live_processes():
    result = []
    for name, expected in PROCESS_FILES.items():
        path = ROOT / name
        if not path.exists():
            continue
        pid = int(path.read_text().strip())
        proc = Path(f"/proc/{pid}/cmdline")
        if not proc.exists():
            continue
        command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if not command:
            continue
        if expected not in command:
            raise RuntimeError(f"PID {pid} in {name} does not run {expected}: {command}")
        result.append({"pid": pid, "pid_file": name, "command": command})
    return result


def current_jobs(db):
    rows = []
    for rid, ordinal, raw, owner in db.execute(
        "SELECT id, ordinal, config, message FROM jobs WHERE status='running' ORDER BY ordinal"
    ):
        folder = ROOT / "runs" / rid
        trajectory = folder / "trajectory.jsonl"
        lines = trajectory.read_text().splitlines() if trajectory.exists() else []
        last_epoch = json.loads(lines[-1])["epoch"] if lines else 0
        checkpoints = sorted(int(p.stem.split("_")[-1]) for p in folder.glob("checkpoint_*.pt"))
        if not checkpoints or checkpoints[-1] > last_epoch:
            raise RuntimeError(f"Inconsistent checkpoint for {rid}")
        rows.append({
            "run_id": rid, "ordinal": ordinal, "config": json.loads(raw),
            "owner": owner, "last_trajectory_epoch": last_epoch,
            "last_checkpoint_epoch": checkpoints[-1],
            "checkpoint_path": str(folder / f"checkpoint_{checkpoints[-1]}.pt"),
        })
    return rows


def main():
    db = sqlite3.connect(ROOT / "queue.sqlite", timeout=120)
    processes = live_processes()
    jobs = current_jobs(db)
    if not processes:
        raise RuntimeError("No live exhaustive-grid workers found")
    audit = {
        "decision": "Pause exhaustive proposal grid; run predeclared eight-run focused Poisson/PINN follow-up",
        "scope": "Only the original three exhaustive workers and the priority vision worker",
        "reversible": True,
        "pause_requested_unix": time.time(),
        "processes": processes,
        "active_jobs_before_pause": jobs,
        "resume_instruction": "Set paused_scope jobs to retry in queue.sqlite, then relaunch launch.sh or priority_vision.py; run.py resumes from the latest checkpoint.",
    }
    path = ROOT / "scope_pause_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    for item in processes:
        os.kill(item["pid"], signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline and live_processes():
        time.sleep(0.2)
    remaining = live_processes()
    if remaining:
        raise RuntimeError(f"Workers did not stop: {remaining}")
    # A worker may have completed a run and started the next one between
    # the first audit and SIGTERM. Capture the final active set before marking it.
    jobs_at_pause = current_jobs(db)
    with db:
        for item in jobs_at_pause:
            db.execute(
                "UPDATE jobs SET status='paused_scope', message=?, updated=? WHERE id=? AND status='running'",
                ("Scope narrowed by user; checkpoint retained", time.time(), item["run_id"]),
            )
    states = dict(db.execute("SELECT status, count(*) FROM jobs GROUP BY status"))
    if states.get("running", 0):
        raise RuntimeError(f"Unexpected running queue rows: {states}")
    audit["pause_completed_unix"] = time.time()
    audit["active_jobs_at_pause"] = jobs_at_pause
    audit["queue_status_after_pause"] = states
    path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"paused_jobs": len(jobs_at_pause), "workers_stopped": len(processes), "queue": states,
                      "checkpoint_epochs": {r["run_id"]: r["last_checkpoint_epoch"] for r in jobs_at_pause}}))


if __name__ == "__main__":
    main()

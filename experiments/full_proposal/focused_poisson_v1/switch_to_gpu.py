"""Archive incomplete CPU attempts and restart all formal arms on GPUs."""

import json
import os
import shutil
import signal
import sqlite3
import time
from pathlib import Path

from grid import ROOT


def active_pid(path):
    if not path.exists():
        return None
    pid = int(path.read_text().strip())
    cmd = Path(f"/proc/{pid}/cmdline")
    if not cmd.exists():
        return None
    raw = cmd.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return pid if "campaign.py" in raw else None


def main():
    db = sqlite3.connect(ROOT / "queue.sqlite", timeout=120)
    rows = db.execute("SELECT id,config,message FROM jobs WHERE status='running'").fetchall()
    pids = [active_pid(ROOT / f"worker_{i}.pid") for i in (0, 1)]
    if len(rows) != 2 or any(pid is None for pid in pids):
        raise RuntimeError(f"Expected two live CPU workers and jobs: {pids}, {rows}")
    audit = {"reason": "Move focused study to now-idle GPUs; preserve CPU prefixes and restart all methods on the same device class",
             "time_unix": time.time(), "cpu_workers": pids, "archived_attempts": []}
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline and any(active_pid(ROOT / f"worker_{i}.pid") for i in (0, 1)):
        time.sleep(0.2)
    if any(active_pid(ROOT / f"worker_{i}.pid") for i in (0, 1)):
        raise RuntimeError("CPU workers did not stop")
    for rid, raw, owner in rows:
        source = ROOT / "runs" / rid
        if (source / "result.json").exists():
            raise RuntimeError(f"Run completed during switch: {rid}")
        target = ROOT / "cpu_preempted" / "runs" / rid
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Archive destination already exists: {target}")
        trajectory = source / "trajectory.jsonl"
        lines = trajectory.read_text().splitlines() if trajectory.exists() else []
        checkpoints = [int(p.stem.split("_")[-1]) for p in source.glob("checkpoint_*.pt")]
        audit["archived_attempts"].append({"run_id": rid, "config": json.loads(raw),
            "owner": owner, "last_epoch": json.loads(lines[-1])["epoch"] if lines else 0,
            "last_checkpoint_epoch": max(checkpoints) if checkpoints else None,
            "archived_path": str(target)})
        shutil.move(str(source), str(target))
    with db:
        db.executemany("UPDATE jobs SET status='pending',message=?,updated=? WHERE id=?",
                       [("Restart from epoch zero on GPU; CPU prefix archived", time.time(), rid) for rid, _, _ in rows])
    audit["queue_after_switch"] = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
    (ROOT / "gpu_switch_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"archived": len(rows), "queue": audit["queue_after_switch"]}))


if __name__ == "__main__":
    main()

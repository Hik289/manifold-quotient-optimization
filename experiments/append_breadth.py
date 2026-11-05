#!/usr/bin/env python3
"""append_breadth.py — 把 breadth scan (INR/PINN/GCN × 3 arm, 2000ep) 补进 results.json (幂等)."""
import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
TS = time.strftime("%Y-%m-%dT%H:%M:%S+0900")
ENV = dict(gpu="none (CPU torch)", gpu_count=0, python_version="3.11",
           key_packages={"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.10.0+cu128"})


def main():
    d = json.load(open(os.path.join(HERE, 'breadth_scan_summary.json')))
    recs = []
    for tname, td in d['tasks'].items():
        ev = td['evaluation']
        for method, r in td['results'].items():
            eid = f"BR_{tname}_{method}"
            recs.append(dict(
                project_id="manifoldquotient", experiment_id=eid, timestamp=TS,
                method=method,
                description=f"Breadth scan {tname} {method} 2000ep (main-claim gauge robustness). "
                            f"task verdict={ev['verdict']} (C1 kfac-lock={ev['c1_kfac_lock']} "
                            f"C2 gauge-robust={ev['c2_gauge_robust']}). PREREG_breadth_scan.md",
                config=dict(model=tname, dataset=tname, learning_rate=1e-3, epochs=2000,
                            seeds=[42], rank=16, init="lora", sylvester_eps=1e-6,
                            task_verdict=ev['verdict']),
                environment=ENV,
                results=dict(
                    main_metric=dict(name="gauge_product_disagreement_max",
                                     values_by_seed={"42": r['gauge_disag']},
                                     mean=r['gauge_disag'], std=None),
                    other_metrics={
                        "kfac_over_sqrtkappa_final": {"mean": r['kfac_over_sqrtk'], "std": None},
                        "kfac_final": {"mean": r['kfac_final'], "std": None},
                        "sqrt_kappa": {"mean": r['sqrt_kappa'], "std": None},
                        "final_loss": {"mean": r['final_loss'], "std": None},
                        "final_psnr": {"mean": r.get('final_psnr'), "std": None},
                        "final_acc": {"mean": r.get('final_acc'), "std": None},
                        "task_metric": {"mean": r.get('metric'), "std": None},
                    }),
                trajectory_file=f"logs/breadth_{tname}_{method}_kfac.jsonl",
                training_time_seconds=None,
                notes=f"single seed 42, no mean/std/CI. breadth task verdict={ev['verdict']}. "
                      f"C1(kappa-drift) task-dependent: only matrix sensing drifts >3x; INR/GCN/PINN AdamW ~1.2x.",
            ))
    resfile = os.path.join(HERE, 'results.json')
    existing = json.load(open(resfile))
    ids = {r['experiment_id'] for r in recs}
    existing = [r for r in existing if r['experiment_id'] not in ids]
    existing.extend(recs)
    json.dump(existing, open(resfile, 'w'), indent=2, default=float)
    import collections
    print(f"total {len(existing)} records; breadth added {len(recs)}")
    print("by prefix:", dict(collections.Counter(r['experiment_id'].split('_')[0] for r in existing)))


if __name__ == '__main__':
    main()

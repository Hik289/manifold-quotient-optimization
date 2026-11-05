#!/usr/bin/env python3
"""append_phase3_a2.py — 把 A2 决定性对照 (Full/A1/A2 × cond) 补进 results.json (幂等)."""
import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
TS = time.strftime("%Y-%m-%dT%H:%M:%S+0900")
ENV = dict(gpu="none (CPU analytic)", gpu_count=0, python_version="3.11",
           key_packages={"numpy": "2.4.6", "scipy": "1.17.1"})
NARR = {'Full': 'reference', 'A1': 'balance_workhorse', 'A2': 'balance_workhorse'}
DESC = {'Full': 'Full QuotientOpt', 'A1': 'balance-only (no horizontal projection)',
        'A2': 'projection-only (no balance retraction) — DECISIVE balance-workhorse test'}


def main():
    d = json.load(open(os.path.join(HERE, 'phase3_a2_summary.json')))
    recs = []
    for key, abs_ in d['results'].items():
        cond = int(key.replace('cond', ''))
        for ab_id, r in abs_.items():
            s = r['train']; g = r['gauge']
            recs.append(dict(
                project_id="manifoldquotient",
                experiment_id=f"P3_MS_{ab_id}_cond{cond}_r16", timestamp=TS,
                method=f"quotientopt_ablation_{ab_id}",
                description=f"Phase3 ablation {ab_id} ({DESC[ab_id]}) matrix sensing cond={cond} rank16, "
                           f"1000ep + gauge robustness (500ep). narrative={NARR[ab_id]}",
                config=dict(model="matrix_factorization", dataset="matrix_sensing",
                            learning_rate=1e-2, epochs=1000, seeds=[42], cond=cond,
                            m=256, d=256, r=16, init="lora", sylvester_eps=1e-6,
                            ablation=ab_id, narrative_tag=NARR[ab_id]),
                environment=ENV,
                results=dict(
                    main_metric=dict(name="gauge_product_disagreement_max",
                                     values_by_seed={"42": g['disagreement_max']},
                                     mean=g['disagreement_max'], std=None),
                    other_metrics={
                        "gauge_disagreement_mean": {"mean": g['disagreement_mean'], "std": None},
                        "rho_upd_mean": {"mean": s['rho_upd_mean'], "std": None},
                        "kfac_final": {"mean": s['kfac_final'], "std": None},
                        "kfac_over_sqrtkappa": {"mean": s['kfac_over_sqrtkappa'], "std": None},
                        "final_rel": {"mean": s['final_rel'], "std": None},
                        "best_rel": {"mean": s['best_rel'], "std": None},
                        "loss_auc": {"mean": s['loss_auc'], "std": None},
                        "epoch_to_target": {"mean": s['epoch_to_target'], "std": None},
                    }),
                trajectory_file=s['trajectory_file'], training_time_seconds=None,
                notes="single seed 42, no mean/std/CI per project protocol",
            ))
    resfile = os.path.join(HERE, 'results.json')
    existing = json.load(open(resfile))
    ids = {r['experiment_id'] for r in recs}
    existing = [r for r in existing if r['experiment_id'] not in ids]
    existing.extend(recs)
    json.dump(existing, open(resfile, 'w'), indent=2, default=float)
    print(f"total {len(existing)} records, added/updated {len(recs)} P3 A2 records")


if __name__ == '__main__':
    main()

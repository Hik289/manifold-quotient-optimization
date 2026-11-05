#!/usr/bin/env python3
"""append_phase3_full.py — 把完整消融矩阵 A1-A14 补进 results.json (幂等).
覆盖之前只有 A2 的 P3_MS 记录, 用完整版替换."""
import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
TS = time.strftime("%Y-%m-%dT%H:%M:%S+0900")
ENV = dict(gpu="none (CPU analytic)", gpu_count=0, python_version="3.11",
           key_packages={"numpy": "2.4.6", "scipy": "1.17.1"})


def main():
    d = json.load(open(os.path.join(HERE, 'phase3_full_summary.json')))
    recs = []
    for s in d['summaries']:
        ab = s['ablation']; cond = int(s['cond']); rank = s['rank']
        eid = s['exp_id']
        recs.append(dict(
            project_id="manifoldquotient", experiment_id=eid, timestamp=TS,
            method=f"quotientopt_ablation_{ab}",
            description=f"Phase3 ablation {ab} ({s['desc']}) {s['task']} cond={cond} rank={rank}, "
                        f"1000ep + gauge robustness. narrative={s['narrative']}",
            config=dict(model="matrix_factorization", dataset="matrix_sensing",
                        learning_rate=1e-2, epochs=1000, seeds=[42], cond=cond,
                        m=256, d=256, r=rank, init=("balanced" if ab == "A12" else "lora"),
                        sylvester_eps=(0.0 if ab == "A8" else 1e-6),
                        ablation=ab, narrative_tag=s['narrative']),
            environment=ENV,
            results=dict(
                main_metric=dict(name="gauge_product_disagreement_max",
                                 values_by_seed={"42": s['gauge_disag_max']},
                                 mean=s['gauge_disag_max'], std=None),
                other_metrics={
                    "gauge_disagreement_mean": {"mean": s['gauge_disag_mean'], "std": None},
                    "rho_upd_mean": {"mean": s['rho_upd_mean'], "std": None},
                    "kfac_final": {"mean": s['kfac_final'], "std": None},
                    "kfac_over_sqrtkappa": {"mean": s['kfac_over_sqrtkappa'], "std": None},
                    "balance_err_final": {"mean": s['balance_err_final'], "std": None},
                    "final_rel": {"mean": s['final_rel'], "std": None},
                    "best_rel": {"mean": s['best_rel'], "std": None},
                    "loss_auc": {"mean": s['loss_auc'], "std": None},
                    "epoch_to_target": {"mean": s['epoch_to_target'], "std": None},
                }),
            trajectory_file=s['trajectory_file'], training_time_seconds=None,
            notes=f"single seed 42, no mean/std/CI. ablation={ab} narrative={s['narrative']}",
        ))
    resfile = os.path.join(HERE, 'results.json')
    existing = json.load(open(resfile))
    # 移除所有旧 P3_MS (A2-only 版) 用完整版替换
    existing = [r for r in existing if not r['experiment_id'].startswith('P3_MS')]
    existing.extend(recs)
    json.dump(existing, open(resfile, 'w'), indent=2, default=float)
    import collections
    cnt = collections.Counter(r['experiment_id'].split('_')[0] for r in existing)
    print(f"total {len(existing)} records; P3 ablation records added: {len(recs)}")
    print("by prefix:", dict(cnt))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
append_f2_to_results.py — 数据 hygiene: 把 4 个 F2 决定性实验补进 canonical results.json.

不跑新实验, 只从 *_summary.json 读核心数字, 按 AGENTS.md 标准格式追加.
单 seed 约定: 不填 mean/std/CI (values_by_seed 只有 "42", std=None).

4 个实验:
  EXP-GS   gauge_stress_summary.json      (P3, F2 INTERVENE)
  EXP-F2BF f2_balancefreq_summary.json    (H0.method_balance_vs_full.1.1)
  EXP-F2S  f2_stochastic_summary.json     (H0.method_balance_vs_full.1.2 维度1+3)
  EXP-F2SG f2_stoch_gauge_summary.json    (H0.method_balance_vs_full.1.2 维度2)

Author: ML Engineer (the project team)
"""
import json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
TS = time.strftime("%Y-%m-%dT%H:%M:%S+0900")
ENV = dict(gpu="none (CPU analytic)", gpu_count=0, python_version="3.11",
           key_packages={"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.10.0+cu128"})
NOTE = "single seed 42, no mean/std/CI per project protocol"


def L(x):
    return json.load(open(os.path.join(HERE, x)))


def rec(exp_id, method, desc, config, main_name, main_val, other, traj=None):
    return dict(
        project_id="manifoldquotient", experiment_id=exp_id, timestamp=TS, method=method,
        description=desc,
        config=dict(seeds=[42], **config),
        environment=ENV,
        results=dict(
            main_metric=dict(name=main_name, values_by_seed={"42": main_val}, mean=main_val, std=None),
            other_metrics={k: {"mean": v, "std": None} for k, v in other.items()},
        ),
        trajectory_file=traj, training_time_seconds=None, notes=NOTE,
    )


def build():
    records = []

    # ---- EXP-GS: gauge stress (F2 INTERVENE) ----
    gs = L('gauge_stress_summary.json')
    for key, methods in gs['results'].items():          # key = matrix_factorization_condXXX
        cond = int(key.split('cond')[1])
        for method, r in methods.items():
            records.append(rec(
                exp_id=f"EXP-GS_{method}_cond{cond}", method=method,
                desc=f"Gauge stress test (P3, F2 INTERVENE) matrix sensing cond={cond} rank16, "
                     f"scalar gauge{{1e-3..1e3}}+dense cond{{1..1e4}}, 1000ep. "
                     f"product trajectory disagreement vs gauge=I reference.",
                config=dict(model="matrix_factorization", dataset="matrix_sensing",
                            learning_rate=1e-2, epochs=1000, cond=cond, m=256, d=256, r=16,
                            init="lora", sylvester_eps=1e-6,
                            gauges="scalar{1e-3,1e-2,1e-1,1e1,1e2,1e3}+dense_cond{1,10,1e2,1e3,1e4}"),
                main_name="product_trajectory_disagreement_max_over_gauges",
                main_val=r['disagreement_max_over_gauges'],
                other=dict(disagreement_mean_over_gauges=r['disagreement_mean_over_gauges'],
                           n_diverged=r['n_diverged']),
            ))

    # ---- EXP-F2BF: balance frequency (H0.method_balance_vs_full.1.1) ----
    bf = L('f2_balancefreq_summary.json')
    for key, tags in bf['results'].items():
        cond = int(key.split('cond')[1])
        for tag, r in tags.items():                     # tag = method_ivN
            records.append(rec(
                exp_id=f"EXP-F2BF_{tag}_cond{cond}", method=r['method'],
                desc=f"F2 probe1 balance-frequency (H0.method_balance_vs_full.1.1, INTERVENE=A6/A7) "
                     f"matrix sensing cond={cond} rank16 balance_interval={r['interval']}, 500ep+gauge subset.",
                config=dict(model="matrix_factorization", dataset="matrix_sensing",
                            learning_rate=1e-2, epochs=500, cond=cond, m=256, d=256, r=16,
                            init="lora", sylvester_eps=1e-6, balance_interval=r['interval'],
                            gauge_subset="scalar{1e-3,1e3}+dense_cond{1e2,1e4}"),
                main_name="product_trajectory_disagreement_max",
                main_val=r['disagreement_max'],
                other=dict(disagreement_mean=r['disagreement_mean'],
                           rho_upd_mean=r['rho_upd_mean'], final_rel=r['final_rel'],
                           balance_interval=r['interval']),
            ))

    # ---- EXP-F2S: stochastic gradient dims 1+3 (H0.method_balance_vs_full.1.2) ----
    st = L('f2_stochastic_summary.json')
    for s in st['summaries']:
        cond = int(s['cond'])
        records.append(rec(
            exp_id=s['exp_id'].replace('F2S_', 'EXP-F2S_'), method=s['method'],
            desc=f"F2 probe2 stochastic-gradient dims1+3 (H0.method_balance_vs_full.1.2, INTERVENE) "
                 f"matrix sensing cond={cond} rank16 rel-grad-noise sigma={s['sigma']}, 1000ep.",
            config=dict(model="matrix_factorization", dataset="matrix_sensing",
                        learning_rate=1e-2, epochs=1000, cond=cond, m=256, d=256, r=16,
                        init="lora", sylvester_eps=1e-6, grad_noise_sigma=s['sigma']),
            main_name="steady_state_rel_error_mean_last200ep",
            main_val=s['steady_rel_mean'],
            other=dict(steady_rel_std=s['steady_rel_std'], best_rel=s['best_rel'],
                       final_rel=s['final_rel'], rho_upd_mean=s['rho_upd_mean'],
                       loss_auc=s['loss_auc'], kfac_final=s['kfac_final']),
            traj=s['trajectory_file'],
        ))

    # ---- EXP-F2SG: stochastic + gauge (dim 2) ----
    sg = L('f2_stoch_gauge_summary.json')
    for key, methods in sg['results'].items():          # key = condXXX_sigY
        cond = int(key.split('cond')[1].split('_')[0])
        sigma = float(key.split('sig')[1])
        for method, r in methods.items():
            records.append(rec(
                exp_id=f"EXP-F2SG_{method}_cond{cond}_sig{sigma}", method=method,
                desc=f"F2 probe2 dim2 stochastic+gauge robustness (H0.method_balance_vs_full.1.2, "
                     f"most-decisive causal) matrix sensing cond={cond} rank16 sigma={sigma}, "
                     f"same noise-seed isolates gauge vs noise, 1000ep.",
                config=dict(model="matrix_factorization", dataset="matrix_sensing",
                            learning_rate=1e-2, epochs=1000, cond=cond, m=256, d=256, r=16,
                            init="lora", sylvester_eps=1e-6, grad_noise_sigma=sigma,
                            noise_seed=777,
                            gauge_subset="scalar{1e-3,1e3}+dense_cond{1e2,1e4}"),
                main_name="product_trajectory_disagreement_max",
                main_val=r['disagreement_max'],
                other=dict(disagreement_mean=r['disagreement_mean']),
            ))
    return records


def main():
    resfile = os.path.join(HERE, 'results.json')
    existing = json.load(open(resfile))
    # 去重: 移除任何已存在的 EXP-GS/F2BF/F2S/F2SG (幂等)
    prefixes = ('EXP-GS', 'EXP-F2BF', 'EXP-F2S', 'EXP-F2SG')
    existing = [r for r in existing if not r['experiment_id'].startswith(prefixes)]
    new = build()
    existing.extend(new)
    json.dump(existing, open(resfile, 'w'), indent=2, default=float)
    # report
    import collections
    cnt = collections.Counter(r['experiment_id'].split('_')[0] for r in existing)
    print(f"total records: {len(existing)}")
    print(f"added F2 records: {len(new)}")
    print("by prefix:", dict(cnt))
    return existing


if __name__ == '__main__':
    main()

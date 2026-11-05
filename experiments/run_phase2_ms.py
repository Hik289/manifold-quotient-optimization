#!/usr/bin/env python3
"""
run_phase2_ms.py — Phase 2 matrix sensing 长训练 (2000 epoch).

必跑: matrix sensing cond{100,1000} × rank16, methods {adamw, balance_only, quotientopt}.
重点观测 (短跑看不出, 长训练显现):
  P2: AdamW κ_factor 是否向 κ(X) 漂移 (QO 保持 √κX)
  P5: 高 cond 下 QO epoch-to-target vs AdamW (加速比 ≥2x), 优势 2000ep 保持 (S6/F7)
  F2 任务层: full QO vs balance-only final metric / 收敛速度逐维

完整 20 指标每 epoch 存 jsonl. checkpoint epoch 存指标快照.
输出: results.json(append) + logs/{exp}_trajectory.jsonl + phase2_ms_summary.json

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer
from metrics import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

EPOCHS = 2000
EPS = 1e-6
CKPT = set([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] + list(range(1100, EPOCHS + 1, 100)))
TARGET_REL = 1e-4   # epoch-to-target (P5): rel_error 首次 < 1e-4


def train_long(task, method, lr, exp_id, epochs=EPOCHS, init_mode='lora'):
    B, A = task.make_init(mode=init_mode)
    opt = make_optimizer(method, lr=lr)
    # κ(X): X* 秩为 r, 用 top-r 奇异值 (= 生成时的 cond)
    sv_X = np.linalg.svd(task.X.astype(np.float64), compute_uv=False)[:task.r]
    kappa_X = float(sv_X[0] / sv_X[-1]); sqrt_kappa_X = float(np.sqrt(kappa_X))
    traj = []
    epoch_to_target = None
    diverged = False
    fpath = os.path.join(LOGDIR, f"{exp_id}_trajectory.jsonl")
    with open(fpath, 'w') as fout:
        for t in range(epochs):
            t0 = time.time()
            Bn, An, info = opt.step(B, A, task.grad)
            if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
                diverged = True
                break
            loss = task.loss(Bn, An)
            rel = task.rel_error(Bn, An)
            if epoch_to_target is None and rel < TARGET_REL:
                epoch_to_target = t
            secB, secA = info['section']
            M = compute_metrics(secB, secA, info['gB'], info['gA'], info['uB'], info['uA'],
                                Xstar=task.X, loss=loss, lr=lr,
                                momentum=info.get('momentum'), dB=info['dB'], dA=info['dA'],
                                eps=EPS, wall=time.time() - t0, heavy=True)
            M['epoch'] = t; M['rel_error'] = rel
            M['sqrt_kappa_X'] = sqrt_kappa_X; M['kappa_X'] = kappa_X
            M['is_checkpoint'] = (t in CKPT)
            fout.write(json.dumps(M, default=float) + '\n')
            # 只在内存留轻量汇总 (省内存)
            traj.append(dict(epoch=t, rho_upd=M['rho_upd'], rho_grad=M['rho_grad'],
                             factor_condition=M['factor_condition'], kappa_X=kappa_X,
                             sqrt_kappa_X=sqrt_kappa_X, train_loss=loss, rel_error=rel,
                             balance_error=M['balance_error'], sec=M['wall_clock_sec']))
            B, A = Bn, An
    rho_upds = [m['rho_upd'] for m in traj]
    kfacs = [m['factor_condition'] for m in traj]
    losses = [m['train_loss'] for m in traj]
    # κ 漂移: 后期均值 / 前期均值
    kfac_early = float(np.mean(kfacs[:200])) if len(kfacs) >= 200 else None
    kfac_late = float(np.mean(kfacs[-200:])) if len(kfacs) >= 200 else None
    summary = dict(
        exp_id=exp_id, method=method, lr=lr, epochs=len(traj), diverged=diverged,
        rho_upd_mean=float(np.mean(rho_upds)), rho_upd_final=float(rho_upds[-1]),
        kfac_mean=float(np.mean(kfacs)), kfac_final=float(kfacs[-1]),
        kfac_early=kfac_early, kfac_late=kfac_late,
        kfac_drift_ratio=float(kfac_late / kfac_early) if kfac_early else None,
        kappa_X=kappa_X, sqrt_kappa_X=sqrt_kappa_X,
        kfac_over_sqrtkappa_final=float(kfacs[-1] / sqrt_kappa_X),
        final_loss=float(losses[-1]), final_rel_error=float(traj[-1]['rel_error']),
        best_loss=float(np.min(losses)),
        loss_auc=float(np.trapezoid(losses)),
        epoch_to_target=epoch_to_target,
        total_sec=float(np.sum([m['sec'] for m in traj])),
        trajectory_file=f"logs/{exp_id}_trajectory.jsonl",
    )
    return summary


def main():
    t0 = time.time()
    grid = [(cond, 16) for cond in [100.0, 1000.0]]
    methods = ['adamw', 'balance_only', 'quotientopt']
    lr = 1e-2
    summaries = []
    results_records = []
    for (cond, rank) in grid:
        for method in methods:
            task = make_task('matrix_factorization', seed=42, m=256, d=256, r=rank, cond=cond)
            exp_id = f"P2_MS_{method}_cond{int(cond)}_r{rank}"
            print(f"[{time.strftime('%H:%M:%S')}] start {exp_id}", flush=True)
            s = train_long(task, method, lr, exp_id)
            summaries.append(dict(cond=cond, rank=rank, **s))
            results_records.append(build_record(exp_id, cond, rank, method, lr, s))
            print(f"[{time.strftime('%H:%M:%S')}] done {exp_id}: "
                  f"ρ_upd={s['rho_upd_mean']:.2e} κfac_drift={s['kfac_drift_ratio']} "
                  f"ep2tgt={s['epoch_to_target']} final_rel={s['final_rel_error']:.2e}", flush=True)

    out = dict(elapsed_sec=time.time() - t0, epochs=EPOCHS, lr=lr, summaries=summaries)
    with open(os.path.join(HERE, 'phase2_ms_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    _append_results(results_records)
    print_report(out)
    print(f"\nPHASE2_MS_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


def build_record(exp_id, cond, rank, method, lr, s):
    return dict(
        project_id="manifoldquotient", experiment_id=exp_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), method=method,
        description=f"Phase2 long-training matrix sensing cond={cond} rank={rank} (2000ep)",
        config=dict(model="matrix_factorization", dataset="matrix_sensing",
                    learning_rate=lr, epochs=EPOCHS, seeds=[42], init="lora",
                    sylvester_eps=EPS, m=256, d=256, r=rank, cond=cond),
        environment=dict(gpu="none (CPU analytic)", gpu_count=0, python_version="3.11",
                         key_packages={"numpy": "2.4.6", "scipy": "1.17.1"}),
        results=dict(
            main_metric=dict(name="rho_upd", values_by_seed={"42": s['rho_upd_mean']},
                             mean=s['rho_upd_mean'], std=None),
            other_metrics=dict(
                factor_condition_final={"mean": s['kfac_final'], "std": None},
                kfac_drift_ratio={"mean": s['kfac_drift_ratio'], "std": None},
                kfac_over_sqrtkappa_final={"mean": s['kfac_over_sqrtkappa_final'], "std": None},
                epoch_to_target={"mean": s['epoch_to_target'], "std": None},
                final_rel_error={"mean": s['final_rel_error'], "std": None},
                best_loss={"mean": s['best_loss'], "std": None},
                loss_auc={"mean": s['loss_auc'], "std": None},
            )),
        trajectory_file=s['trajectory_file'], training_time_seconds=s['total_sec'],
        notes="single seed 42, no mean/std/CI per project protocol",
    )


def _append_results(records):
    resfile = os.path.join(HERE, 'results.json')
    existing = []
    if os.path.exists(resfile):
        try:
            existing = json.load(open(resfile))
        except Exception:
            existing = []
    existing.extend(records)
    with open(resfile, 'w') as f:
        json.dump(existing, f, indent=2, default=float)


def print_report(out):
    print(f"\n=== PHASE 2 MS 长训练 (epochs={out['epochs']}) ===")
    print(f"{'exp':<34}{'ρ_upd':>11}{'κfac_fin':>11}{'κfac/√κX':>11}{'κ_drift':>10}{'ep2tgt':>8}{'final_rel':>11}")
    for s in out['summaries']:
        print(f"{s['exp_id']:<34}{s['rho_upd_mean']:>11.2e}{s['kfac_final']:>11.2e}"
              f"{s['kfac_over_sqrtkappa_final']:>11.3f}{(s['kfac_drift_ratio'] or 0):>10.3f}"
              f"{str(s['epoch_to_target']):>8}{s['final_rel_error']:>11.2e}")


if __name__ == '__main__':
    main()

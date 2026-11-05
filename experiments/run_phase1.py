#!/usr/bin/env python3
"""
run_phase1.py — Phase 1 机制短跑筛选 (300 epoch).

三个核心机制问题:
  P1: 垂直能量可测且 QO 能压掉 (AdamW ρ_upd vs QO ρ_upd; 全方法 ρ_grad<1e-6)
  P2: 因子条件数改善 (QO κ_factor ~√κ(X); AdamW 漂移向 κ(X))
  P4: F2 关键失败对照 (balance-only ρ_upd 是否仍 >0.01; full QO <1e-3) —— ISOLATE

强信号配方: AdamW + LoRA init + cond{100,1000}. matrix sensing + deep linear.
输出: results.json (标准格式) + logs/{exp}_trajectory.jsonl + phase1_summary.json

不碰长训练 / gauge stress (Director 指示).

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer
from metrics import compute_metrics
from geometry import factor_condition_number

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

EPOCHS = 300
EPS = 1e-6


def train_run(task, method, lr, init_mode='lora', epochs=EPOCHS, exp_id='', heavy_every=1,
              opt_kwargs=None):
    """跑一个短训练, 记录每 epoch 20 指标. 返回 (summary, trajectory)."""
    opt_kwargs = opt_kwargs or {}
    B, A = task.make_init(mode=init_mode)
    opt = make_optimizer(method, lr=lr, **opt_kwargs)
    traj = []
    # √κ(X) 参考线 (target)
    sv_X = np.linalg.svd(task.X.astype(np.float64), compute_uv=False)[:task.r]
    kappa_X = float(sv_X[0] / sv_X[-1])
    sqrt_kappa_X = float(np.sqrt(kappa_X))
    diverged = False
    for t in range(epochs):
        t0 = time.time()
        Bn, An, info = opt.step(B, A, task.grad)
        if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
            diverged = True
            break
        loss = task.loss(Bn, An)
        heavy = (t % heavy_every == 0) or (t == epochs - 1)
        # ⚠️ 几何量在 optimizer 实际操作的 section 上算 (QO=post-balance),
        #    梯度也来自 info (同一 section), 保证梯度点/更新点/测量点自洽.
        secB, secA = info['section']
        M = compute_metrics(secB, secA, info['gB'], info['gA'], info['uB'], info['uA'],
                            Xstar=task.X, loss=loss, lr=lr,
                            momentum=info.get('momentum'), dB=info['dB'], dA=info['dA'],
                            eps=EPS, wall=time.time() - t0, heavy=heavy)
        M['epoch'] = t
        M['rel_error'] = task.rel_error(Bn, An)
        M['sqrt_kappa_X'] = sqrt_kappa_X
        M['kappa_X'] = kappa_X
        traj.append(M)
        B, A = Bn, An
    # summary
    rho_upds = [m['rho_upd'] for m in traj if m['rho_upd'] is not None]
    rho_grads = [m['rho_grad'] for m in traj if m['rho_grad'] is not None]
    kfacs = [m['factor_condition'] for m in traj if m['factor_condition'] is not None]
    losses = [m['train_loss'] for m in traj]
    summary = dict(
        exp_id=exp_id, method=method, lr=lr, init=init_mode, epochs=len(traj),
        diverged=diverged,
        rho_upd_mean=float(np.mean(rho_upds)) if rho_upds else None,
        rho_upd_final=float(rho_upds[-1]) if rho_upds else None,
        rho_upd_max=float(np.max(rho_upds)) if rho_upds else None,
        rho_grad_mean=float(np.mean(rho_grads)) if rho_grads else None,
        rho_grad_max=float(np.max(rho_grads)) if rho_grads else None,
        kfac_mean=float(np.mean(kfacs)) if kfacs else None,
        kfac_final=float(kfacs[-1]) if kfacs else None,
        kfac_over_sqrtkappa=float(np.mean(kfacs) / sqrt_kappa_X) if kfacs else None,
        sqrt_kappa_X=sqrt_kappa_X, kappa_X=kappa_X,
        final_loss=float(losses[-1]) if losses else None,
        final_rel_error=float(traj[-1]['rel_error']) if traj else None,
        loss_auc=float(np.trapezoid(losses)) if losses else None,
        balance_err_final=float(traj[-1]['balance_error']) if traj else None,
    )
    return summary, traj


def promotion_decision(base_sum, qo_sum):
    """晋级判据 (EXPERIMENT_PLAN):
    nec1: baseline ρ_upd mean > 0.01
    nec2: QO short-run final loss <= baseline*1.05
    suf : QO ρ_upd < baseline/100 OR QO κ_factor << baseline (ratio>2)"""
    nec1 = (base_sum['rho_upd_mean'] or 0) > 0.01
    bl = base_sum['final_loss'] or float('inf')
    ql = qo_sum['final_loss'] or float('inf')
    nec2 = ql <= bl * 1.05 if np.isfinite(bl) else True
    suf_rho = (qo_sum['rho_upd_mean'] or 1) < (base_sum['rho_upd_mean'] or 0) / 100
    suf_kfac = False
    if qo_sum['kfac_mean'] and base_sum['kfac_mean']:
        suf_kfac = base_sum['kfac_mean'] / qo_sum['kfac_mean'] > 2
    promoted = nec1 and nec2 and (suf_rho or suf_kfac)
    return dict(promoted=bool(promoted), nec1_baseline_signal=bool(nec1),
                nec2_qo_not_worse=bool(nec2), suf_rho_advantage=bool(suf_rho),
                suf_kfac_advantage=bool(suf_kfac))


def main():
    t0 = time.time()
    # 强信号配方: AdamW + LoRA init + cond{100,1000}
    grid = []
    # matrix sensing: {adamw,sgdm,balance_only,quotientopt} x cond{100,1000} x rank{8,16}
    for cond in [100.0, 1000.0]:
        for rank in [8, 16]:
            grid.append(('matrix_factorization', dict(m=256, d=256, r=rank, cond=cond), 'MS'))
    # deep linear: {adamw,balance_only,quotientopt} x depth{4,8} x cond{100,1000}
    for cond in [100.0, 1000.0]:
        for depth in [4, 8]:
            grid.append(('deep_linear', dict(m=128, d=128, r=8, depth=depth, cond=cond), 'DL'))

    methods = ['adamw', 'sgdm', 'balance_only', 'quotientopt']
    lr = 1e-2  # 强信号统一 lr (adam-scale); baseline 收敛已在 Phase0 确认
    results_records = []   # 标准 results.json
    summaries = []
    promotions = []

    for (tname, tkw, tag) in grid:
        cfg_summaries = {}
        for method in methods:
            if tname == 'deep_linear' and method == 'sgdm':
                continue  # deep linear 只跑 adamw/balance_only/quotientopt (PLAN)
            task = make_task(tname, seed=42, **tkw)
            exp_id = f"P1_{tag}_{method}_cond{int(tkw['cond'])}_" + \
                     (f"r{tkw.get('r')}" if tag == 'MS' else f"d{tkw.get('depth')}")
            s, traj = train_run(task, method, lr, exp_id=exp_id)
            cfg_summaries[method] = s
            summaries.append(dict(task=tname, tag=tag, **tkw, **s))
            # 写轨迹 jsonl
            with open(os.path.join(LOGDIR, f"{exp_id}_trajectory.jsonl"), 'w') as f:
                for m in traj:
                    f.write(json.dumps(m, default=float) + '\n')
            # 标准 results.json 记录
            results_records.append(build_result_record(exp_id, tname, tkw, method, lr, s, traj))
        # 晋级 + F2 对照 (以 adamw 为 baseline)
        base = cfg_summaries.get('adamw')
        qo = cfg_summaries.get('quotientopt')
        bal = cfg_summaries.get('balance_only')
        if base and qo:
            dec = promotion_decision(base, qo)
            # F2 早期警报: balance-only ρ_upd vs full QO ρ_upd 分离
            f2 = dict(
                balance_only_rho_upd_mean=bal['rho_upd_mean'] if bal else None,
                full_qo_rho_upd_mean=qo['rho_upd_mean'],
                rho_upd_separated=bool((bal['rho_upd_mean'] or 0) > 0.01 and (qo['rho_upd_mean'] or 1) < 1e-3) if bal else None,
                balance_only_final_loss=bal['final_loss'] if bal else None,
                full_qo_final_loss=qo['final_loss'],
            )
            promotions.append(dict(task=tname, tag=tag, cond=tkw['cond'],
                                   rank=tkw.get('r'), depth=tkw.get('depth'),
                                   **dec, f2_check=f2))

    out = dict(elapsed_sec=time.time() - t0, epochs=EPOCHS, lr=lr,
               summaries=summaries, promotions=promotions)
    with open(os.path.join(HERE, 'phase1_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    # append 标准 results.json
    resfile = os.path.join(HERE, 'results.json')
    existing = []
    if os.path.exists(resfile):
        try:
            existing = json.load(open(resfile))
        except Exception:
            existing = []
    existing.extend(results_records)
    with open(resfile, 'w') as f:
        json.dump(existing, f, indent=2, default=float)

    print_report(out)
    return out


def build_result_record(exp_id, tname, tkw, method, lr, s, traj):
    return dict(
        project_id="manifoldquotient", experiment_id=exp_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        method=method,
        description=f"Phase1 short-run screening {tname} cond={tkw['cond']} (strong-signal recipe)",
        config=dict(model=tname, dataset=tname, learning_rate=lr, batch_size=None,
                    epochs=EPOCHS, seeds=[42], init="lora", sylvester_eps=EPS, **tkw),
        environment=dict(gpu="none (CPU analytic)", gpu_count=0, python_version="3.11",
                         key_packages={"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.10.0+cu128"}),
        results=dict(
            main_metric=dict(name="rho_upd (update-direction vertical energy)",
                             values_by_seed={"42": s['rho_upd_mean']},
                             mean=s['rho_upd_mean'], std=None),
            other_metrics=dict(
                rho_upd_final={"mean": s['rho_upd_final'], "std": None},
                rho_grad_max={"mean": s['rho_grad_max'], "std": None},
                factor_condition_mean={"mean": s['kfac_mean'], "std": None},
                sqrt_kappa_X={"mean": s['sqrt_kappa_X'], "std": None},
                final_loss={"mean": s['final_loss'], "std": None},
                final_rel_error={"mean": s['final_rel_error'], "std": None},
                loss_auc={"mean": s['loss_auc'], "std": None},
            )),
        trajectory_file=f"logs/{exp_id}_trajectory.jsonl",
        training_time_seconds=None,
        notes="single seed 42, no mean/std/CI per project protocol",
    )


def print_report(out):
    print(f"\n=== PHASE 1 短跑筛选 (epochs={out['epochs']}, elapsed {out['elapsed_sec']:.1f}s) ===\n")
    print("--- 机制问题 1+2: ρ_upd / ρ_grad / κ_factor ---")
    print(f"{'exp':<34}{'ρ_upd_mean':>12}{'ρ_grad_max':>12}{'κfac/√κX':>11}{'final_loss':>12}")
    for s in out['summaries']:
        eid = s['exp_id']
        print(f"{eid:<34}{_f(s['rho_upd_mean']):>12}{_f(s['rho_grad_max']):>12}"
              f"{_f(s['kfac_over_sqrtkappa']):>11}{_f(s['final_loss']):>12}")
    print("\n--- 机制问题 3: F2 关键失败对照 (balance-only vs full QO ρ_upd 分离) + 晋级 ---")
    for p in out['promotions']:
        f2 = p['f2_check']
        cfg = f"{p['tag']} cond{int(p['cond'])} " + (f"r{p['rank']}" if p['rank'] else f"d{p['depth']}")
        print(f"[{cfg}] promoted={p['promoted']} | "
              f"bal_ρ_upd={_f(f2['balance_only_rho_upd_mean'])} qo_ρ_upd={_f(f2['full_qo_rho_upd_mean'])} "
              f"separated={f2['rho_upd_separated']} | nec1={p['nec1_baseline_signal']} "
              f"nec2={p['nec2_qo_not_worse']} suf_rho={p['suf_rho_advantage']} suf_kfac={p['suf_kfac_advantage']}")


def _f(x):
    if x is None:
        return "  n/a"
    if abs(x) < 1e-3 or abs(x) > 1e4:
        return f"{x:.2e}"
    return f"{x:.4f}"


if __name__ == '__main__':
    main()

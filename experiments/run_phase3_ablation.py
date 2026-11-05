#!/usr/bin/env python3
"""
run_phase3_ablation.py — Phase 3 消融矩阵 A1-A14 (路 A 诚实 pivot: which components matter).

新叙事: balance retraction 是 workhorse; horizontal projection/momentum 降级理论完备性.
每个消融标注支撑叙事哪点: balance_workhorse / projection_redundant / momentum_redundant / numerical.

主任务: matrix sensing(256,rank16,cond{100,1000}) + 真实 deep linear(depth8,cond{100,1000}).
seed42, 1000ep, 完整指标. 补 canonical results.json.

⚠️ A2 最高优先: projection-only (do_balance=False) 是否崩 gauge 不变性 = balance 承载组件决定性对照.
   加 gauge robustness 测试(不只 final metric).

用法: python run_phase3_ablation.py [A2only|all]

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer, QuotientOpt
from metrics import compute_metrics
from geometry import F32, fnorm
from data import make_rng

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)
EPOCHS = 1000
EPS = 1e-6
LR = 1e-2
GAUGE_SUBSET = [('scalar', 1e-3), ('scalar', 1e3), ('dense', 1e2), ('dense', 1e4)]

# 消融定义: id -> (QuotientOpt kwargs 覆盖, 叙事标签, 描述)
ABLATIONS = {
    'Full':  (dict(), 'reference', 'Full QuotientOpt (balance+hproj+hmomentum)'),
    'A1':    (dict(do_horizontal_grad=False, reproject_momentum=False), 'balance_workhorse',
              'balance-only (no horizontal projection)'),
    'A2':    (dict(do_balance=False), 'balance_workhorse',
              'projection-only (no balance retraction) — DECISIVE: does gauge invariance collapse?'),
    'A3':    (dict(use_momentum=False), 'momentum_redundant', 'no momentum'),
    'A4':    (dict(transport_momentum=False), 'momentum_redundant', 'no momentum transport'),
    'A5':    (dict(reproject_momentum=False), 'momentum_redundant', 'no momentum reprojection'),
    'A6':    (dict(balance_interval=5), 'balance_freq', 'balance every 5 steps'),
    'A7':    (dict(balance_interval=25), 'balance_freq', 'balance every 25 steps'),
    'A8':    (dict(sylvester_eps=0.0), 'numerical', 'no Sylvester damping (eps=0)'),
    'A9':    (dict(gauge_mode='scalar'), 'gauge_subgroup', 'scalar gauge correction only'),
    'A10':   (dict(gauge_mode='diagonal'), 'gauge_subgroup', 'diagonal gauge correction only'),
    'A11':   (dict(factor_weight_decay=1e-2), 'signal_source', 'factor weight decay (non-invariant)'),
    'A12':   (dict(), 'signal_source', 'fixed balanced init (init=balanced, online correction off-test)'),
    # A14 rank sweep handled separately
}


def make_opt(ab_id):
    kw, _, _ = ABLATIONS[ab_id]
    return QuotientOpt(lr=LR, **kw)


def init_mode_for(ab_id):
    return 'balanced' if ab_id == 'A12' else 'lora'


def train_ablation(task, ab_id, exp_id, epochs=EPOCHS):
    opt = make_opt(ab_id)
    B, A = task.make_init(mode=init_mode_for(ab_id))
    sv_X = np.linalg.svd(task.X.astype(np.float64), compute_uv=False)[:task.r]
    sqrt_kappa_X = float(np.sqrt(sv_X[0] / sv_X[-1]))
    fpath = os.path.join(LOGDIR, f"{exp_id}_trajectory.jsonl")
    rhos = []; kfacs = []; losses = []; rels = []; bals = []
    diverged = False
    ep2tgt = None
    with open(fpath, 'w') as fout:
        for t in range(epochs):
            t0 = time.time()
            Bn, An, info = opt.step(B, A, task.grad)
            if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
                diverged = True; break
            B, A = Bn, An
            loss = task.loss(B, A); rel = task.rel_error(B, A)
            if ep2tgt is None and rel < 1e-4:
                ep2tgt = t
            secB, secA = info['section']
            heavy = (t % 5 == 0 or t == epochs - 1)
            M = compute_metrics(secB, secA, info['gB'], info['gA'], info['uB'], info['uA'],
                                Xstar=task.X, loss=loss, lr=LR,
                                momentum=info.get('momentum'), dB=info['dB'], dA=info['dA'],
                                eps=EPS, heavy=heavy)
            M['epoch'] = t; M['rel_error'] = rel
            fout.write(json.dumps(M, default=float) + '\n')
            rhos.append(M['rho_upd']); losses.append(loss); rels.append(rel)
            bals.append(M['balance_error'])
            if M['factor_condition'] is not None:
                kfacs.append(M['factor_condition'])
    return dict(
        exp_id=exp_id, ablation=ab_id, diverged=diverged, epochs=len(losses),
        rho_upd_mean=float(np.mean(rhos)) if rhos else None,
        rho_upd_final=float(rhos[-1]) if rhos else None,
        kfac_final=float(kfacs[-1]) if kfacs else None,
        kfac_over_sqrtkappa=float(kfacs[-1]/sqrt_kappa_X) if kfacs else None,
        sqrt_kappa_X=sqrt_kappa_X,
        balance_err_final=float(bals[-1]) if bals else None,
        final_loss=float(losses[-1]) if losses else None,
        best_loss=float(np.min(losses)) if losses else None,
        final_rel=float(rels[-1]) if rels else None,
        best_rel=float(np.min(rels)) if rels else None,
        loss_auc=float(np.trapezoid(losses)) if losses else None,
        epoch_to_target=ep2tgt,
        trajectory_file=f"logs/{exp_id}_trajectory.jsonl",
    )


def make_gauge(r, kind, val, rng):
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    U = np.linalg.qr(rng.standard_normal((r, r)))[0]
    V = np.linalg.qr(rng.standard_normal((r, r)))[0]
    s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
    return (U @ np.diag(s) @ V.T).astype(F32)


def gauge_robustness(task_name, tkw, ab_id, epochs=500):
    """A2 决定性测试: 从注入 gauge 出发的 product disagreement.
    projection-only(A2) 若 disagreement 爆炸 → balance 是 gauge 不变性承载组件."""
    task_ref = make_task(task_name, seed=42, **tkw)
    B0, A0 = task_ref.make_init(mode=init_mode_for(ab_id))
    r = tkw['r']
    Ieye = np.eye(r).astype(F32)

    def run(Q):
        B = (B0 @ Q).astype(F32)
        Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
        A = (Qinv @ A0).astype(F32)
        task = make_task(task_name, seed=42, **tkw)
        opt = make_opt(ab_id)
        prods = []
        for t in range(epochs):
            Bn, An, info = opt.step(B, A, task.grad)
            if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
                break
            B, A = Bn, An
            prods.append((B @ A).astype(F32).copy())
        return prods
    ref = run(Ieye)
    T = len(ref)
    disags = []
    for (kind, val) in GAUGE_SUBSET:
        Q = make_gauge(r, kind, val, make_rng(456))
        gp = run(Q)
        Tg = min(len(gp), T)
        if Tg == 0:
            disags.append(float('inf')); continue
        disags.append(float(fnorm(gp[Tg-1] - ref[Tg-1])))
    finals = [d for d in disags if np.isfinite(d)]
    return dict(disagreement_max=float(np.max(finals)) if finals else float('inf'),
                disagreement_mean=float(np.mean(finals)) if finals else float('inf'),
                per_gauge=disags, diverged=(T < epochs))


def run_a2_decisive():
    """A2 决定性对照: projection-only vs balance-only vs full, gauge robustness + final metric."""
    print("=== A2 决定性对照: projection-only 是否崩 gauge 不变性 ===", flush=True)
    out = {}
    for cond in [100.0, 1000.0]:
        tkw = dict(m=256, d=256, r=16, cond=cond)
        key = f"cond{int(cond)}"
        out[key] = {}
        print(f"[{time.strftime('%H:%M:%S')}] matrix sensing {key}", flush=True)
        for ab_id in ['Full', 'A1', 'A2']:
            exp_id = f"P3_MS_{ab_id}_cond{int(cond)}_r16"
            s = train_ablation(make_task('matrix_factorization', seed=42, **tkw), ab_id, exp_id)
            gr = gauge_robustness('matrix_factorization', tkw, ab_id)
            out[key][ab_id] = dict(train=s, gauge=gr)
            print(f"  {ab_id:5} ({ABLATIONS[ab_id][2][:38]:38}): "
                  f"gauge_disag_max={gr['disagreement_max']:.2e} "
                  f"final_rel={s['final_rel']:.2e} ρ_upd={s['rho_upd_mean']:.2e} "
                  f"κfac={s['kfac_final']:.1f} {'DIVERGED' if gr['diverged'] or s['diverged'] else ''}", flush=True)
        # A2 verdict
        a2 = out[key]['A2']['gauge']['disagreement_max']
        a1 = out[key]['A1']['gauge']['disagreement_max']
        full = out[key]['Full']['gauge']['disagreement_max']
        print(f"  >>> A2 verdict [{key}]: projection-only disag={a2:.2e} vs balance系(A1={a1:.2e},Full={full:.2e})", flush=True)
        if a2 > 100 * max(a1, full):
            print(f"      → projection-only gauge 不变性崩塌 (>100x balance系) → balance 是承载组件 ✅", flush=True)
        elif a2 > 10 * max(a1, full):
            print(f"      → projection-only 明显更差 (>10x) → balance 承载, projection 不足", flush=True)
        else:
            print(f"      → projection-only 未明显崩 → 需进一步分析", flush=True)
    return out


def run_full_matrix():
    """完整 A1-A14 消融矩阵. matrix sensing cond{100,1000}×rank16 全消融 + rank sweep(A14).
    每个消融含 gauge robustness (支撑 balance workhorse / projection redundant 叙事)."""
    summaries = []
    # A1-A12 + Full on matrix sensing cond{100,1000}
    ab_list = ['Full', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12']
    for cond in [100.0, 1000.0]:
        tkw = dict(m=256, d=256, r=16, cond=cond)
        for ab_id in ab_list:
            exp_id = f"P3_MS_{ab_id}_cond{int(cond)}_r16"
            print(f"[{time.strftime('%H:%M:%S')}] {exp_id} ({ABLATIONS[ab_id][1]})", flush=True)
            s = train_ablation(make_task('matrix_factorization', seed=42, **tkw), ab_id, exp_id)
            gr = gauge_robustness('matrix_factorization', tkw, ab_id)
            summaries.append({**s, 'task': 'matrix_sensing', 'cond': cond, 'rank': 16,
                              'narrative': ABLATIONS[ab_id][1], 'desc': ABLATIONS[ab_id][2],
                              'gauge_disag_max': gr['disagreement_max'],
                              'gauge_disag_mean': gr['disagreement_mean'],
                              'gauge_per': gr['per_gauge']})
            print(f"    disag={gr['disagreement_max']:.2e} rho_upd={s['rho_upd_mean']:.2e} "
                  f"kfac={s['kfac_final']} final_rel={s['final_rel']:.2e} "
                  f"{'DIV' if s['diverged'] or gr['diverged'] else ''}", flush=True)
    # A14 rank sweep (Full) on cond100
    for rank in [4, 8, 32]:   # 16 已在上面
        tkw = dict(m=256, d=256, r=rank, cond=100.0)
        exp_id = f"P3_MS_A14_Full_cond100_r{rank}"
        print(f"[{time.strftime('%H:%M:%S')}] {exp_id} (rank sweep)", flush=True)
        s = train_ablation(make_task('matrix_factorization', seed=42, **tkw), 'Full', exp_id)
        gr = gauge_robustness('matrix_factorization', tkw, 'Full')
        summaries.append({**s, 'ablation': 'A14', 'task': 'matrix_sensing', 'cond': 100.0, 'rank': rank,
                          'narrative': 'rank_sensitivity', 'desc': f'rank={rank} sensitivity (Full)',
                          'gauge_disag_max': gr['disagreement_max'],
                          'gauge_disag_mean': gr['disagreement_mean'],
                          'gauge_per': gr['per_gauge']})
        print(f"    rank{rank}: disag={gr['disagreement_max']:.2e} kfac={s['kfac_final']} "
              f"final_rel={s['final_rel']:.2e}", flush=True)
    return summaries


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'A2only'
    t0 = time.time()
    if mode == 'A2only':
        out = run_a2_decisive()
        with open(os.path.join(HERE, 'phase3_a2_summary.json'), 'w') as f:
            json.dump(dict(elapsed_sec=time.time()-t0, results=out), f, indent=2, default=float)
        print(f"\nPHASE3_A2_DONE elapsed={time.time()-t0:.1f}s")
    else:
        summaries = run_full_matrix()
        out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, summaries=summaries)
        with open(os.path.join(HERE, 'phase3_full_summary.json'), 'w') as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nPHASE3_FULL_DONE elapsed={time.time()-t0:.1f}s  ({len(summaries)} ablation runs)")
    return out


if __name__ == '__main__':
    main()

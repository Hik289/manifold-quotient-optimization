#!/usr/bin/env python3
"""
run_f2_stochastic.py — F2 探针2: 随机梯度 H0.method_balance_vs_full.1.2 (INTERVENE).

Director 采纳的 candidate (b). 逻辑: 确定性 full-batch 下 momentum 作用小, 水平 momentum
价值被掩盖; 随机梯度下 momentum 主导, balance-only 的垂直动量累积应更严重.
判据: 随机梯度下 full QO 优于 balance-only → 水平 momentum 独立价值显现.

实现: matrix sensing 加相对梯度噪声 g_stoch = g + σ·‖g‖_scale·N(0,1) (模拟 SGD 噪声;
  256×256 full measurement 存不下, 梯度噪声是干净标准替代, 直接隔离 momentum 效应).
  noise σ ∈ {0.1, 0.3, 1.0} × {adamw, balance_only, quotientopt}.
  seed 42 固定 (RNG 可复现). 1000ep (随机梯度需更长看稳态).

观测: final/best rel error (稳态噪声地板), loss AUC, ρ_upd, gauge disagreement 子集.
输出: f2_stochastic_summary.json + logs/F2S_*_trajectory.jsonl

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer
from metrics import compute_metrics
from geometry import F32, fnorm
from data import make_rng

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

EPOCHS = 1000
EPS = 1e-6
LR = 1e-2


def make_noisy_grad(task, sigma, rng):
    """包装 task.grad 加相对梯度噪声: g + sigma * ‖g‖_F/sqrt(size) * N(0,1)."""
    def grad_fn(B, A):
        gB, gA = task.grad(B, A)
        if sigma > 0:
            # 相对噪声: 尺度 ∝ 每个因子梯度的 RMS
            sB = sigma * fnorm(gB) / np.sqrt(gB.size)
            sA = sigma * fnorm(gA) / np.sqrt(gA.size)
            gB = (gB + sB * rng.standard_normal(gB.shape)).astype(F32)
            gA = (gA + sA * rng.standard_normal(gA.shape)).astype(F32)
        return gB, gA
    return grad_fn


def train_stoch(task, method, sigma, exp_id, epochs=EPOCHS):
    rng = make_rng(42)   # 固定 seed, 随机梯度可复现
    grad_fn = make_noisy_grad(task, sigma, rng)
    B, A = task.make_init(mode='lora')
    opt = make_optimizer(method, lr=LR)
    fpath = os.path.join(LOGDIR, f"{exp_id}_trajectory.jsonl")
    rhos = []; rels = []; losses = []; kfacs = []
    with open(fpath, 'w') as fout:
        for t in range(epochs):
            t0 = time.time()
            Bn, An, info = opt.step(B, A, grad_fn)
            if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
                break
            B, A = Bn, An
            loss = task.loss(B, A); rel = task.rel_error(B, A)
            secB, secA = info['section']
            M = compute_metrics(secB, secA, info['gB'], info['gA'], info['uB'], info['uA'],
                                Xstar=task.X, loss=loss, lr=LR,
                                momentum=info.get('momentum'), dB=info['dB'], dA=info['dA'],
                                eps=EPS, heavy=(t % 5 == 0 or t == epochs-1))
            M['epoch'] = t; M['rel_error'] = rel
            fout.write(json.dumps(M, default=float) + '\n')
            rhos.append(M['rho_upd']); rels.append(rel); losses.append(loss)
            if M['factor_condition'] is not None:
                kfacs.append(M['factor_condition'])
    # 稳态: 后 200 epoch 均值 (随机梯度地板)
    tail = min(200, len(rels))
    return dict(
        exp_id=exp_id, method=method, sigma=sigma, epochs=len(rels),
        rho_upd_mean=float(np.mean(rhos)),
        final_rel=float(rels[-1]), best_rel=float(np.min(rels)),
        steady_rel_mean=float(np.mean(rels[-tail:])),   # 稳态噪声地板
        steady_rel_std=float(np.std(rels[-tail:])),
        final_loss=float(losses[-1]), best_loss=float(np.min(losses)),
        loss_auc=float(np.trapezoid(losses)),
        kfac_final=float(kfacs[-1]) if kfacs else None,
        trajectory_file=f"logs/{exp_id}_trajectory.jsonl",
    )


def main():
    t0 = time.time()
    methods = ['adamw', 'balance_only', 'quotientopt']
    sigmas = [0.1, 0.3, 1.0]
    summaries = []
    for cond in [100.0, 1000.0]:
        for sigma in sigmas:
            for method in methods:
                task = make_task('matrix_factorization', seed=42, m=256, d=256, r=16, cond=cond)
                exp_id = f"F2S_{method}_cond{int(cond)}_sig{sigma}"
                print(f"[{time.strftime('%H:%M:%S')}] {exp_id}", flush=True)
                s = train_stoch(task, method, sigma, exp_id)
                s['cond'] = cond
                summaries.append(s)
                print(f"  ρ_upd={s['rho_upd_mean']:.2e} steady_rel={s['steady_rel_mean']:.3e}"
                      f"±{s['steady_rel_std']:.1e} best_rel={s['best_rel']:.3e} auc={s['loss_auc']:.2e}", flush=True)
    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, sigmas=sigmas, summaries=summaries)
    with open(os.path.join(HERE, 'f2_stochastic_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    # 决定性对比: 每 (cond,sigma) 下 full QO vs balance-only steady rel
    print("\n=== F2 探针2 裁决: 随机梯度下 full QO vs balance-only (steady_rel_mean) ===")
    by = {(s['method'], int(s['cond']), s['sigma']): s for s in summaries}
    for cond in [100, 1000]:
        for sigma in sigmas:
            a = by[('adamw', cond, sigma)]; b = by[('balance_only', cond, sigma)]; q = by[('quotientopt', cond, sigma)]
            ratio = b['steady_rel_mean'] / q['steady_rel_mean'] if q['steady_rel_mean'] > 0 else float('inf')
            verdict = "QO优" if ratio > 1.2 else ("bal优" if ratio < 0.83 else "相当")
            print(f"cond{cond} σ={sigma}: adamw={a['steady_rel_mean']:.2e} | bal={b['steady_rel_mean']:.2e} | "
                  f"QO={q['steady_rel_mean']:.2e} | bal/QO={ratio:.2f} [{verdict}]", flush=True)
    print(f"\nF2_STOCHASTIC_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

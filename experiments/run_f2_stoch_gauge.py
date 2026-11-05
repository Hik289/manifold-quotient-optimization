#!/usr/bin/env python3
"""
run_f2_stoch_gauge.py — F2 探针2 维度2 (Director 强调的最关键因果测试):
  随机梯度 + gauge 扰动下, balance-only(残余 ρ_upd) 是否终于漂移而 full QO 不漂?

逻辑: 确定性下 balance retraction 每步冲掉 drift → balance-only gauge-robust (gauge stress 已见)。
  但随机梯度下 momentum 累积的垂直分量持续注入噪声, balance-only 可能终于漂移, full QO 不漂
  (水平 momentum 保持 clean)。这是 horizontal momentum 独立价值最干净的因果维度。

设计: matrix sensing cond{100,1000}×rank16, 相对梯度噪声 σ∈{0.3,1.0}, 从同一 quotient point
  注入 gauge 子集(scalar{1e-3,1e3}+dense{1e2,1e4}), 三方 {adamw,balance_only,quotientopt}
  各训练 1000ep, 测 product trajectory disagreement (相对 gauge=I 参考, 同噪声 seed).
  ⚠️ 关键: 参考轨迹与 gauge 轨迹用**相同噪声序列** (同 seed), 隔离 gauge 效应 vs 噪声效应.

判据: 随机梯度下 balance-only disagreement >> full QO → horizontal momentum 独立价值确立(F2 排除)。
  仍 bal ≈ QO → 该维度也 falsify。

输出: f2_stoch_gauge_summary.json

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer
from geometry import F32, fnorm
from data import make_rng

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 1000
EPS = 1e-6
LR = 1e-2
GAUGE_SUBSET = [('scalar', 1e-3), ('scalar', 1e3), ('dense', 1e2), ('dense', 1e4)]


def make_gauge(r, kind, val, rng):
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    U = np.linalg.qr(rng.standard_normal((r, r)))[0]
    V = np.linalg.qr(rng.standard_normal((r, r)))[0]
    s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
    return (U @ np.diag(s) @ V.T).astype(F32)


def noisy_grad(task, sigma, noise_seed):
    """相对梯度噪声, 用固定 noise_seed 使参考轨迹与 gauge 轨迹噪声序列一致."""
    rng = make_rng(noise_seed)
    def grad_fn(B, A):
        gB, gA = task.grad(B, A)
        if sigma > 0:
            sB = sigma * fnorm(gB) / np.sqrt(gB.size)
            sA = sigma * fnorm(gA) / np.sqrt(gA.size)
            gB = (gB + sB * rng.standard_normal(gB.shape)).astype(F32)
            gA = (gA + sA * rng.standard_normal(gA.shape)).astype(F32)
        return gB, gA
    return grad_fn


def train(task, method, B0, A0, Q, sigma, noise_seed, epochs=EPOCHS):
    r = B0.shape[1]
    B = (B0 @ Q).astype(F32)
    Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
    A = (Qinv @ A0).astype(F32)
    opt = make_optimizer(method, lr=LR)
    grad_fn = noisy_grad(task, sigma, noise_seed)
    prods = []
    for t in range(epochs):
        Bn, An, info = opt.step(B, A, grad_fn)
        if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
            break
        B, A = Bn, An
        prods.append((B @ A).astype(F32).copy())
    return prods


def run_config(tname, tkw, sigma):
    r = tkw['r']
    task_ref = make_task(tname, seed=42, **tkw)
    B0, A0 = task_ref.make_init(mode='lora')
    Ieye = np.eye(r).astype(F32)
    out = {}
    for method in ['adamw', 'balance_only', 'quotientopt']:
        # 参考轨迹 gauge=I, noise_seed=777
        task = make_task(tname, seed=42, **tkw)
        ref = train(task, method, B0, A0, Ieye, sigma, noise_seed=777)
        T = len(ref)
        disags = []
        for (kind, val) in GAUGE_SUBSET:
            Q = make_gauge(r, kind, val, make_rng(456))
            task = make_task(tname, seed=42, **tkw)
            # 关键: 同 noise_seed=777, 隔离 gauge 效应
            gp = train(task, method, B0, A0, Q, sigma, noise_seed=777)
            Tg = min(len(gp), T)
            if Tg == 0:
                disags.append(float('inf')); continue
            disags.append(float(fnorm(gp[Tg-1] - ref[Tg-1])))
        finals = [d for d in disags if np.isfinite(d)]
        out[method] = dict(
            disagreement_max=float(np.max(finals)) if finals else float('inf'),
            disagreement_mean=float(np.mean(finals)) if finals else float('inf'),
            per_gauge=disags,
        )
    return out


def main():
    t0 = time.time()
    configs = [('matrix_factorization', dict(m=256, d=256, r=16, cond=cond)) for cond in [100.0, 1000.0]]
    all_out = {}
    for (tname, tkw) in configs:
        for sigma in [0.3, 1.0]:
            key = f"cond{int(tkw['cond'])}_sig{sigma}"
            print(f"[{time.strftime('%H:%M:%S')}] {key}", flush=True)
            res = run_config(tname, tkw, sigma)
            all_out[key] = res
            a = res['adamw']['disagreement_max']
            b = res['balance_only']['disagreement_max']
            q = res['quotientopt']['disagreement_max']
            ratio = b/q if q > 0 else float('inf')
            print(f"  adamw={a:.2e} bal={b:.2e} QO={q:.2e} | bal/QO={ratio:.2f} "
                  f"[{'QO更robust' if ratio>3 else 'bal≈QO'}]", flush=True)
    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, gauge_subset=GAUGE_SUBSET, results=all_out)
    with open(os.path.join(HERE, 'f2_stoch_gauge_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nF2_STOCH_GAUGE_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

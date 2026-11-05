#!/usr/bin/env python3
"""
run_f2_balancefreq.py — F2 最终判定: H0.method_balance_vs_full.1.1
  水平投影独立价值 ∝ 1/balance_frequency  (debug_move: INTERVENE, = 消融 A6/A7)

核心逻辑 (Director): balance_interval=1 时 balance 每步冲掉 gauge drift → 水平投影冗余(已观测)。
  interval=5/25 时动量 gauge drift 在两次 balance 之间累积, 只有 full QO 的水平 momentum
  能保持轨迹 gauge-clean, balance-only 应漂移。

设计: balance_interval ∈ {1,5,25} × {balance_only, quotientopt} (+ adamw 参照),
  matrix sensing cond{100,1000} rank16, 短跑 500ep + gauge stress 子集(代表性 gauge).
观测: 每档 interval 下 balance-only vs full QO 的
  (product trajectory disagreement / ρ_upd 累积 / final metric).

决定性判据:
  分离随 interval 单调放大 (interval=25 时 QO 明显优于 bal, interval=1 相等) → 水平投影独立价值确立, F2 排除
  各 interval 都 bal≈QO → F2 完全触发, 停下报 Director

输出: f2_balancefreq_summary.json + logs/F2_*_trajectory.jsonl

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

EPOCHS = 500
EPS = 1e-6
LR = 1e-2
# gauge stress 子集: 代表性 (最能放大漂移的 scalar 极端 + dense 高 cond)
GAUGE_SUBSET = [('scalar', 1e-3), ('scalar', 1e3), ('dense', 1e2), ('dense', 1e4)]


def make_gauge(r, kind, val, rng):
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    U = np.linalg.qr(rng.standard_normal((r, r)))[0]
    V = np.linalg.qr(rng.standard_normal((r, r)))[0]
    s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
    return (U @ np.diag(s) @ V.T).astype(F32)


def train(task, method, B0, A0, Q, interval, epochs=EPOCHS, record_rho=False):
    """从 gauge Q 出发训练, 返回 product trajectory (+可选 ρ_upd 轨迹)."""
    r = B0.shape[1]
    B = (B0 @ Q).astype(F32)
    Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
    A = (Qinv @ A0).astype(F32)
    kw = {}
    if method in ('balance_only', 'quotientopt'):
        kw['balance_interval'] = interval
    opt = make_optimizer(method, lr=LR, **kw)
    prods = []; rhos = []
    for t in range(epochs):
        Bn, An, info = opt.step(B, A, task.grad)
        if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
            break
        B, A = Bn, An
        prods.append((B @ A).astype(F32).copy())
        if record_rho:
            secB, secA = info['section']
            M = compute_metrics(secB, secA, info['gB'], info['gA'],
                                info['uB'], info['uA'], eps=EPS, heavy=False)
            rhos.append(M['rho_upd'])
    return prods, rhos


def run_config(tname, tkw):
    r = tkw['r']
    task_ref = make_task(tname, seed=42, **tkw)
    B0, A0 = task_ref.make_init(mode='lora')
    Ieye = np.eye(r).astype(F32)
    out = {}
    # adamw 参照 (无 interval 概念)
    for method, intervals in [('adamw', [1]), ('balance_only', [1, 5, 25]), ('quotientopt', [1, 5, 25])]:
        for interval in intervals:
            tag = f"{method}_iv{interval}"
            # 参考轨迹 gauge=I
            task = make_task(tname, seed=42, **tkw)
            ref_prods, ref_rhos = train(task, method, B0, A0, Ieye, interval, record_rho=True)
            T = len(ref_prods)
            # gauge stress 子集
            disags = []
            for (kind, val) in GAUGE_SUBSET:
                Q = make_gauge(r, kind, val, make_rng(456))
                task = make_task(tname, seed=42, **tkw)
                g_prods, _ = train(task, method, B0, A0, Q, interval)
                Tg = min(len(g_prods), T)
                if Tg == 0:
                    disags.append(float('inf')); continue
                dfin = fnorm(g_prods[Tg-1] - ref_prods[Tg-1])
                disags.append(float(dfin))
            finals = [d for d in disags if np.isfinite(d)]
            # final metric (gauge=I 轨迹)
            final_rel = float(np.linalg.norm(ref_prods[-1] - task_ref.X, 'fro') /
                              (np.linalg.norm(task_ref.X, 'fro') + 1e-12)) if T > 0 else None
            out[tag] = dict(
                method=method, interval=interval,
                disagreement_max=float(np.max(finals)) if finals else float('inf'),
                disagreement_mean=float(np.mean(finals)) if finals else float('inf'),
                per_gauge_disag=disags,
                rho_upd_mean=float(np.mean(ref_rhos)) if ref_rhos else None,
                final_rel=final_rel,
            )
    return out


def main():
    t0 = time.time()
    configs = [('matrix_factorization', dict(m=256, d=256, r=16, cond=cond)) for cond in [100.0, 1000.0]]
    all_out = {}
    for (tname, tkw) in configs:
        key = f"{tname}_cond{int(tkw['cond'])}"
        print(f"[{time.strftime('%H:%M:%S')}] {key}", flush=True)
        res = run_config(tname, tkw)
        all_out[key] = res
        # 打印决定性对比: 每档 interval 下 bal vs QO disagreement
        print(f"  {'tag':<20}{'disag_max':>12}{'disag_mean':>12}{'ρ_upd':>11}{'final_rel':>11}", flush=True)
        for tag in res:
            r = res[tag]
            print(f"  {tag:<20}{r['disagreement_max']:>12.3e}{r['disagreement_mean']:>12.3e}"
                  f"{(r['rho_upd_mean'] or 0):>11.2e}{(r['final_rel'] or 0):>11.2e}", flush=True)
        # F2 裁决: 分离比随 interval
        print(f"  >>> F2 separation (balance-only disag / full QO disag) by interval:", flush=True)
        for iv in [1, 5, 25]:
            b = res.get(f'balance_only_iv{iv}'); q = res.get(f'quotientopt_iv{iv}')
            if b and q and q['disagreement_max'] > 0:
                ratio = b['disagreement_max'] / q['disagreement_max']
                print(f"      interval={iv}: bal/QO = {ratio:.2f}  "
                      f"(bal={b['disagreement_max']:.2e} QO={q['disagreement_max']:.2e})", flush=True)
    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, gauge_subset=GAUGE_SUBSET, results=all_out)
    with open(os.path.join(HERE, 'f2_balancefreq_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nF2_BALANCEFREQ_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
run_gauge_stress.py — Gauge Stress Test (P3) = F2 的 INTERVENE 决定实验.

Director 裁决: ρ_upd 分离是相关性证据, gauge stress 升级为因果.
核心: 从同一 quotient point 出发, 注入不同 gauge Q, 三方对比 product trajectory disagreement.
  disagreement = ‖π(B_T^Q, A_T^Q) − π(B_T, A_T)‖_F   (相对参考 gauge=I)

决定性预期 (Corollary 3):
  full QO 跨 gauge 一致 (O(Tεμ⁻¹)≈1e-2)
  balance-only 因残余 ρ_upd 应明显漂移 (介于 QO 和 AdamW 之间)
  AdamW 漂移最大 (Ω(ηT·0.04))

判据:
  balance-only disagreement >> full QO (≥1 数量级) → F2 在 gauge robustness 维度被排除 (水平投影独立因果贡献确立)
  balance-only ≈ full QO (都 gauge-robust) → F2 在此维度也触发, 报 Director

设计: 同一 X* 与同一初始 quotient point [B0,A0], 对每个 gauge Q 生成 (B0 Q, Q^{-1} A0),
      各方法独立训练 T epoch, 记录 product trajectory, 与 gauge=I 参考比 disagreement.
输出: gauge_stress_summary.json + logs/GS_*_trajectory.jsonl

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tasks import make_task
from optimizers import make_optimizer
from geometry import F32, fnorm
from data import make_rng, init_factors

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

EPOCHS = 1000
EPS = 1e-6
LR = 1e-2


def make_gauge(r, kind, val, rng):
    """kind='scalar': Q=val*I; kind='dense': random Q with cond(Q)=val."""
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    else:  # dense with prescribed condition number
        U = np.linalg.qr(rng.standard_normal((r, r)))[0]
        V = np.linalg.qr(rng.standard_normal((r, r)))[0]
        s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
        return (U @ np.diag(s) @ V.T).astype(F32)


def train_with_gauge(task, method, B0, A0, Q, epochs=EPOCHS):
    """从 gauge-transformed 初始点训练, 返回 product trajectory (每 epoch 的 BA)."""
    B = (B0 @ Q).astype(F32)
    Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
    A = (Qinv @ A0).astype(F32)
    opt = make_optimizer(method, lr=LR)
    prods = []
    for t in range(epochs):
        Bn, An, info = opt.step(B, A, task.grad)
        if not (np.all(np.isfinite(Bn)) and np.all(np.isfinite(An))):
            prods.append(None); break
        B, A = Bn, An
        # 记录 product (低秩, 存 Frobenius 与参考比即可, 但需存矩阵做 disagreement)
        prods.append((B @ A).astype(F32).copy())
    return prods


def run_config(task_name, tkw, methods, gauges):
    """一个 (task, cond) 下, 各方法跨 gauge 的 disagreement."""
    task_ref = make_task(task_name, seed=42, **tkw)
    # 固定初始 quotient point (balanced-ish LoRA init), 所有 gauge 从它出发
    B0, A0 = task_ref.make_init(mode='lora')
    r = tkw['r']
    rng = make_rng(123)
    results = {}
    for method in methods:
        # 参考轨迹: gauge = I
        task = make_task(task_name, seed=42, **tkw)
        ref_prods = train_with_gauge(task, method, B0, A0, np.eye(r).astype(F32))
        T = len([p for p in ref_prods if p is not None])
        method_res = {'ref_final_norm': fnorm(ref_prods[T-1]) if T > 0 else None,
                      'gauges': []}
        for (kind, val) in gauges:
            Q = make_gauge(r, kind, val, make_rng(456))
            task = make_task(task_name, seed=42, **tkw)
            g_prods = train_with_gauge(task, method, B0, A0, Q)
            Tg = min(len([p for p in g_prods if p is not None]), T)
            if Tg == 0:
                disag_final = float('inf'); disag_traj = []
            else:
                # trajectory disagreement 相对参考
                disag_traj = [fnorm(g_prods[t] - ref_prods[t]) for t in range(Tg)
                              if g_prods[t] is not None and ref_prods[t] is not None]
                disag_final = float(disag_traj[-1]) if disag_traj else float('inf')
            method_res['gauges'].append(dict(
                gauge_kind=kind, gauge_val=val,
                cond_Q=float(np.linalg.cond(Q)),
                disagreement_final=disag_final,
                disagreement_max=float(np.max(disag_traj)) if disag_traj else float('inf'),
                diverged=(Tg < T),
                Tg=Tg,
            ))
        # 汇总: 跨所有 gauge 的 max/mean disagreement
        finals = [g['disagreement_final'] for g in method_res['gauges'] if np.isfinite(g['disagreement_final'])]
        method_res['disagreement_max_over_gauges'] = float(np.max(finals)) if finals else float('inf')
        method_res['disagreement_mean_over_gauges'] = float(np.mean(finals)) if finals else float('inf')
        method_res['n_diverged'] = sum(1 for g in method_res['gauges'] if g['diverged'])
        results[method] = method_res
    return results


def main():
    t0 = time.time()
    methods = ['adamw', 'balance_only', 'quotientopt']
    # scalar gauge {1e-3..1e3} + dense gauge cond{1,10,1e2,1e3,1e4}
    gauges = [('scalar', v) for v in [1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3]] + \
             [('dense', v) for v in [1.0, 10.0, 1e2, 1e3, 1e4]]
    configs = [('matrix_factorization', dict(m=256, d=256, r=16, cond=cond)) for cond in [100.0, 1000.0]]

    all_out = {}
    for (tname, tkw) in configs:
        key = f"{tname}_cond{int(tkw['cond'])}"
        print(f"[{time.strftime('%H:%M:%S')}] gauge stress {key}", flush=True)
        res = run_config(tname, tkw, methods, gauges)
        all_out[key] = res
        # F2 决定性对比打印
        for m in methods:
            r = res[m]
            print(f"  {m:14} disag_max={r['disagreement_max_over_gauges']:.3e} "
                  f"disag_mean={r['disagreement_mean_over_gauges']:.3e} n_div={r['n_diverged']}", flush=True)
        # F2 裁决
        qo = res['quotientopt']['disagreement_max_over_gauges']
        bal = res['balance_only']['disagreement_max_over_gauges']
        adam = res['adamw']['disagreement_max_over_gauges']
        ratio_bal_qo = bal / qo if qo > 0 else float('inf')
        print(f"  >>> F2 verdict [{key}]: bal/QO ratio={ratio_bal_qo:.1f} "
              f"(≥10 → F2 排除, 水平投影独立因果贡献; ~1 → F2 触发)", flush=True)

    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, lr=LR,
               gauges=[[k, v] for (k, v) in gauges], results=all_out)
    with open(os.path.join(HERE, 'gauge_stress_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nGAUGE_STRESS_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

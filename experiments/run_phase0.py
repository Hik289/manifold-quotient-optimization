#!/usr/bin/env python3
"""
run_phase0.py — Phase 0 anchor gate (H0.anchor_1/2/3/4).

四个 anchor:
  anchor_1 (EXP-000): 12 项数值单元测试 (ε=1e-6 自适应阈值 + ε=0 严格)
  anchor_3 (EXP-001): 合成数据生成自检
  anchor_4 (EXP-002): 几何量 pipeline toy-case (rho_grad=0 / rho_upd=解析 / kappa=sqrt(cond))
  anchor_2 (EXP-003): baseline 收敛锚点 (AdamW/SGD-M/Balance-only 收敛到解析最优附近)

全绿才准进 Phase 1. 输出 phase0_results.json.

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tests.test_numerics import run_all as numeric_tests
from data import self_check as data_self_check
from metrics import toy_case_verify
from tasks import make_task
from optimizers import make_optimizer


def _run_one(task, method, lr, init_mode, epochs=1200):
    B, A = task.make_init(mode=init_mode)
    opt = make_optimizer(method, lr=lr)
    last = float('inf')
    for t in range(epochs):
        B, A, info = opt.step(B, A, task.grad)
        if not (np.all(np.isfinite(B)) and np.all(np.isfinite(A))):
            return float('inf'), float('inf')
    return task.loss(B, A), task.rel_error(B, A)


def anchor_2_baseline_convergence():
    """M-ADAMW/M-SGDM/M-BAL 在 matrix fact + deep linear 收敛到解析最优附近.
    对每个方法做小 LR 扫 (框架 sanity + 顺带记录 SGD 的窄稳定区间, 关联 S7).
    判据: 该方法在 best LR 下 final rel_error < 0.05."""
    results = []
    ok = True
    configs = [
        ('matrix_factorization', dict(m=256, d=256, r=16, cond=100.0)),
        ('deep_linear', dict(m=128, d=128, r=8, depth=8, cond=100.0)),
    ]
    lr_grid = {
        'adamw': [1e-2, 3e-2],
        'balance_only': [1e-2, 3e-2],
        'sgdm': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],  # 扫稳定区间
    }
    for tname, tkw in configs:
        for method in ['adamw', 'sgdm', 'balance_only']:
            best = dict(rel=float('inf'), lr=None, loss=float('inf'))
            stable_lrs = []
            for lr in lr_grid[method]:
                task = make_task(tname, seed=42, **tkw)  # fresh task/rng per run
                loss, rel = _run_one(task, method, lr, 'lora')
                if np.isfinite(rel):
                    stable_lrs.append(lr)
                if rel < best['rel']:
                    best = dict(rel=float(rel), lr=lr, loss=float(loss))
            conv = best['rel'] < 0.05
            row = dict(task=tname, method=method, best_lr=best['lr'],
                       final_loss=best['loss'], final_rel_error=best['rel'],
                       n_stable_lr=len(stable_lrs), converged=bool(conv))
            results.append(row)
            if not conv:
                ok = False
    return ok, results


def main():
    t0 = time.time()
    report = {}

    print("=== anchor_1: 12 numerical tests (EXP-000) ===")
    n_eps = numeric_tests(eps=1e-6)
    n_zero = numeric_tests(eps=0.0)
    anchor_1_ok = n_eps['all_pass'] and n_zero['all_pass']
    report['anchor_1'] = dict(ok=anchor_1_ok, eps_1e6=n_eps, eps_0=n_zero)
    print(f"  eps=1e-6 all_pass={n_eps['all_pass']}  eps=0 all_pass={n_zero['all_pass']}")
    if not anchor_1_ok:
        print("  FAILED items (eps=1e-6):", {k: v for k, v in n_eps['passed'].items() if not v})
        print("  FAILED items (eps=0):", {k: v for k, v in n_zero['passed'].items() if not v})

    print("=== anchor_3: data generation self-check (EXP-001) ===")
    d_ok, d_res = data_self_check(verbose=False)
    report['anchor_3'] = dict(ok=d_ok, n_configs=len(d_res))
    print(f"  all_pass={d_ok}")

    print("=== anchor_4: geometry pipeline toy-case (EXP-002) ===")
    t_ok, t_res = toy_case_verify(verbose=True)
    report['anchor_4'] = dict(ok=t_ok, checks=t_res)

    print("=== anchor_2: baseline convergence anchor (EXP-003) ===")
    b_ok, b_res = anchor_2_baseline_convergence()
    report['anchor_2'] = dict(ok=b_ok, results=b_res)
    for r in b_res:
        print(f"  {r['task']:>22} {r['method']:>13}: best_lr={r['best_lr']} rel_err={r['final_rel_error']:.2e} "
              f"n_stable_lr={r['n_stable_lr']} conv={r['converged']}")

    all_green = anchor_1_ok and d_ok and t_ok and b_ok
    report['phase0_gate'] = dict(all_green=all_green,
                                 anchor_1=anchor_1_ok, anchor_2=b_ok,
                                 anchor_3=d_ok, anchor_4=t_ok)
    report['elapsed_sec'] = time.time() - t0
    print(f"\n=== PHASE 0 GATE: all_green={all_green} (elapsed {report['elapsed_sec']:.1f}s) ===")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase0_results.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    return all_green


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)

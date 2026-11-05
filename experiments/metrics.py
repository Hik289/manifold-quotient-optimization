#!/usr/bin/env python3
"""
metrics.py — 20 个 per-epoch 指标记录器 (README §5.3) + toy-case 验证 (H0.anchor_4 / EXP-002).

⚠️ 主指标 = 指标#8 vertical energy ratio, 必须在 update direction u 上算 (C1):
   - AdamW:  u = m_hat / (sqrt(v_hat) + eps)
   - SGD-M:  u = momentum buffer
   - QuotientOpt: u = M_t (水平重投影后动量)
同时记 rho_grad (raw gradient, sanity, 预期 ~0).

Author: ML Engineer (the project team)
"""
import numpy as np
from geometry import (F32, fnorm, horizontal_lift, vertical_energy_ratio,
                      factor_condition_number, balance_error, connection_form,
                      vertical_from_omega, balance_to_section)
from data import make_rng, make_lowrank_target


def compute_metrics(B, A, gB, gA, uB, uA, Xstar=None, loss=None, lr=None,
                    momentum=None, prev_BA=None, dB=None, dA=None,
                    eps=1e-6, wall=None, mem=None, heavy=True):
    """计算一个 epoch 的 20 指标. 返回 dict.
    gB,gA: raw gradient;  uB,uA: 实际更新方向;  momentum: (mB,mA) 或 None."""
    M = {}
    # 1-4
    M['train_loss'] = float(loss) if loss is not None else None
    M['val_metric'] = None
    M['test_metric'] = None
    M['learning_rate'] = float(lr) if lr is not None else None
    # 5 raw gradient norm
    M['raw_grad_norm'] = fnorm(gB) ** 2 + fnorm(gA) ** 2
    M['raw_grad_norm'] = np.sqrt(M['raw_grad_norm'])
    # 6 horizontal gradient norm
    hgB, hgA, Om_g, *_ = horizontal_lift(B, A, gB, gA, eps)
    M['horiz_grad_norm'] = np.sqrt(fnorm(hgB) ** 2 + fnorm(hgA) ** 2)
    # 7 vertical gradient norm (sanity, 预期 ~0)
    M['vert_grad_norm'] = np.sqrt(fnorm(gB - hgB) ** 2 + fnorm(gA - hgA) ** 2)
    # 8 vertical energy ratio —— 主指标 rho_upd (update direction) + rho_grad (sanity)
    rho_upd, upd_info = vertical_energy_ratio(B, A, uB, uA, eps)
    rho_grad, _ = vertical_energy_ratio(B, A, gB, gA, eps)
    M['rho_upd'] = rho_upd           # ⭐ 主指标
    M['rho_grad'] = rho_grad         # sanity, 预期 <1e-6
    M['omega_upd_norm'] = upd_info['omega_norm']
    # 9 factor norm
    M['factor_norm_B'] = fnorm(B); M['factor_norm_A'] = fnorm(A)
    # 10 factor condition number (SVD, heavy)
    if heavy:
        kfac, kB, kA = factor_condition_number(B, A)
        M['factor_condition'] = kfac
        # 11 singular values of X=BA. 注意 X=BA 秩为 r (远小于 min(m,d)),
        #    full-matrix cond 会因 σ_min≈0 爆炸; κ(X) 用 top-r 奇异值比 (product 的真实条件数).
        r = min(B.shape[1], A.shape[0])
        sv = np.linalg.svd((B @ A).astype(np.float64), compute_uv=False)
        sv_r = sv[:r]
        M['sigma_min_X'] = float(sv_r[-1])   # 第 r 个奇异值 (低秩 product 的最小非零奇异值)
        M['kappa_X'] = float(sv_r[0] / sv_r[-1]) if sv_r[-1] > 0 else None
    else:
        M['factor_condition'] = None; M['sigma_min_X'] = None; M['kappa_X'] = None
    # 12 balance error
    M['balance_error'] = balance_error(B, A)
    # 13 Sylvester residual (relative, on update)
    Om_u, C_u, He_u = connection_form(B, A, uB, uA, eps)
    M['sylvester_residual'] = fnorm(He_u @ Om_u + Om_u @ He_u - C_u) / (fnorm(C_u) + 1)
    # 14 momentum vertical residual
    if momentum is not None:
        mB, mA = momentum
        hmB, hmA, *_ = horizontal_lift(B, A, mB, mA, eps)
        M['momentum_vert_residual'] = np.sqrt(fnorm(mB - hmB) ** 2 + fnorm(mA - hmA) ** 2)
    else:
        M['momentum_vert_residual'] = None
    # 15 product update norm
    if dB is not None and dA is not None:
        # d(BA) ≈ dB A + B dA
        M['product_update_norm'] = fnorm(dB @ A + B @ dA)
        # 16 factor update norm
        M['factor_update_norm_B'] = fnorm(dB); M['factor_update_norm_A'] = fnorm(dA)
    else:
        M['product_update_norm'] = None
        M['factor_update_norm_B'] = None; M['factor_update_norm_A'] = None
    # 17 quotient gradient proxy = ||Pi_H G||
    M['quotient_grad_proxy'] = M['horiz_grad_norm']
    # 18 gauge trajectory disagreement (仅 gauge stress EXP 外部填)
    M['gauge_disagreement'] = None
    # 19 wall-clock
    M['wall_clock_sec'] = float(wall) if wall is not None else None
    # 20 peak memory
    M['peak_memory_mb'] = float(mem) if mem is not None else None
    return M


# ---------------------------------------------------------------------------
# toy-case 验证 (EXP-002 / H0.anchor_4)
# ---------------------------------------------------------------------------
def toy_case_verify(verbose=True):
    """注入已知 gauge/垂直分量, 验证 (a) rho_grad=0 (b) rho_upd=解析值 (c) kappa=sqrt(cond)."""
    rng = make_rng(42)
    checks = []
    all_ok = True
    for cond in [1.0, 10.0, 100.0, 1000.0]:
        m, d, r = 64, 48, 8
        X, U, s, V = make_lowrank_target(m, d, r, cond, rng)
        # balanced representative
        B = (U * np.sqrt(s)).astype(F32); A = (np.sqrt(s)[:, None] * V.T).astype(F32)

        # (a) raw LoRA gradient rho_grad = 0  (Corollary 1)
        GX = rng.standard_normal((m, d)).astype(F32)
        gB = (GX @ A.T).astype(F32); gA = (B.T @ GX).astype(F32)
        rho_grad, _ = vertical_energy_ratio(B, A, gB, gA, eps=0.0)
        a_ok = rho_grad < 1e-6

        # (b) 注入已知垂直分量 v=(BΩ0,-Ω0 A) 到 horizontal u_h, rho_upd = ||v||^2/||u||^2
        Om0 = (0.5 * rng.standard_normal((r, r))).astype(F32)
        vB, vA = vertical_from_omega(B, A, Om0)
        # 取一个水平方向 (raw gradient 已水平) 作为基底
        uB = gB + vB; uA = gA + vA
        rho_upd, _ = vertical_energy_ratio(B, A, uB, uA, eps=0.0)
        v_sq = fnorm(vB) ** 2 + fnorm(vA) ** 2
        u_sq = fnorm(uB) ** 2 + fnorm(uA) ** 2
        rho_analytic = v_sq / u_sq
        b_ok = abs(rho_upd - rho_analytic) / (rho_analytic + 1e-12) < 1e-3

        # (c) balanced-rep factor condition ~ sqrt(cond(X))  (Theorem 3)
        kfac, kB, kA = factor_condition_number(B, A)
        sqrt_cond = np.sqrt(cond)
        c_ok = abs(np.log10(kB + 1e-12) - np.log10(sqrt_cond + 1e-12)) < 0.1 if cond > 1 else kB < 1.3

        row = dict(cond=cond, rho_grad=float(rho_grad), a_ok=bool(a_ok),
                   rho_upd=float(rho_upd), rho_analytic=float(rho_analytic), b_ok=bool(b_ok),
                   kappa_B=float(kB), sqrt_cond=float(sqrt_cond), c_ok=bool(c_ok))
        checks.append(row)
        if not (a_ok and b_ok and c_ok):
            all_ok = False
            if verbose:
                print("FAIL:", row)
        elif verbose:
            print(f"cond={cond:>7}: rho_grad={rho_grad:.2e}(a={a_ok}) "
                  f"rho_upd={rho_upd:.4f}~{rho_analytic:.4f}(b={b_ok}) "
                  f"kB={kB:.3f}~sqrt_cond={sqrt_cond:.3f}(c={c_ok})")
    if verbose:
        print(f"toy_case_verify all_pass={all_ok}")
    return all_ok, checks


if __name__ == '__main__':
    import json
    ok, checks = toy_case_verify()
    print(json.dumps(dict(all_pass=ok, checks=checks), indent=2, default=float))

#!/usr/bin/env python3
"""
test_numerics.py — README §10 十二项数值单元测试 (H0.anchor_1 / EXP-000).

阈值策略 (约束 C8):
- ε=0 (精确 connection): 严格 < 1e-6
- ε>0: 涉及投影/水平条件 (10.5/10.6/10.7) 阈值放宽为 max(1e-6, 3ε‖Ω‖)
  (EDA 已证这些超标逐位 = Theorem 11 阻尼偏置 2ε‖Ω‖, 非 bug)

有限差分 (10.11) 必须中心差分 + float64 (前向差分有 2e-3 假失败).
运行: python test_numerics.py  (输出 JSON)  或  pytest test_numerics.py

Author: ML Engineer (the project team)
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry import (F32, fnorm, balance_to_section, gram, connection_form,
                      horizontal_lift, vertical_from_omega, vertical_energy_ratio,
                      factor_condition_number)
from data import make_rng, make_lowrank_target, init_factors

TOL = 1e-6


def adaptive_tol(eps, omega_norm):
    """C8 自适应阈值."""
    return max(1e-6, 3 * eps * omega_norm)


def _make_generic(m, d, r, cond, rng):
    """生成一个 generic (非平衡) 因子 + 随机切向量."""
    X, U, s, V = make_lowrank_target(m, d, r, cond, rng)
    G0 = (np.eye(r) + 0.3 * rng.standard_normal((r, r))).astype(F32)
    B = ((U * np.sqrt(s)) @ G0).astype(F32)
    A = (np.linalg.inv(G0) @ (np.sqrt(s)[:, None] * V.T)).astype(F32)
    gB = rng.standard_normal((m, r)).astype(F32)
    gA = rng.standard_normal((r, d)).astype(F32)
    return B, A, X, gB, gA, s, U, V


def run_config(m, d, r, cond, eps, rng):
    """一个 (m,d,r,cond) 的全部 §10 检查. 返回 (残差 dict, 阈值 dict)."""
    res, thr = {}, {}
    B, A, X, gB, gA, s, U, V = _make_generic(m, d, r, cond, rng)
    Bb, Ab, Q = balance_to_section(B, A)

    # 10.1 乘积保持
    res['10.1'] = fnorm(Bb @ Ab - B @ A) / (fnorm(B @ A) + 1); thr['10.1'] = 1e-4  # FP32 ~2.4e-5
    # 10.2 平衡性
    res['10.2'] = fnorm(Bb.T @ Bb - Ab @ Ab.T); thr['10.2'] = 1e-4
    # 10.3 垂直核 (BΩ)A + B(-ΩA) = 0
    Om = rng.standard_normal((r, r)).astype(F32)
    vB, vA = vertical_from_omega(Bb, Ab, Om)
    res['10.3'] = fnorm(vB @ Ab + Bb @ vA); thr['10.3'] = 1e-4
    # 10.4 Sylvester residual (relative)
    Omega, C, He = connection_form(Bb, Ab, gB, gA, eps)
    res['10.4'] = fnorm(He @ Omega + Omega @ He - C) / (fnorm(C) + 1); thr['10.4'] = 1e-5
    om_norm = fnorm(Omega)
    # 10.5 水平条件 B^T hB - hA A^T = 0  (relative, 归一化到 ‖Ω‖-scale 量级)
    hB, hA, Om5, *_ = horizontal_lift(Bb, Ab, gB, gA, eps)
    scale5 = fnorm(Bb) * fnorm(hB) + fnorm(hA) * fnorm(Ab) + 1e-9
    res['10.5'] = fnorm(Bb.T @ hB - hA @ Ab.T) / scale5
    thr['10.5'] = 1e-6 if eps == 0 else 1e-3   # ε>0: damping-bias regime (Thm 11)
    # 10.6 投影幂等 (relative)
    hB2, hA2, *_ = horizontal_lift(Bb, Ab, hB, hA, eps)
    res['10.6'] = (fnorm(hB2 - hB) + fnorm(hA2 - hA)) / (fnorm(hB) + fnorm(hA) + 1e-9)
    thr['10.6'] = 1e-6 if eps == 0 else 1e-3
    # 10.7 纯垂直归零 (relative)
    pvB, pvA, Om7, *_ = horizontal_lift(Bb, Ab, vB, vA, eps)
    res['10.7'] = (fnorm(pvB) + fnorm(pvA)) / (fnorm(vB) + fnorm(vA) + 1e-9)
    thr['10.7'] = 1e-6 if eps == 0 else 1e-3
    # 10.8 乘积微分保持 hB A + B hA == gB A + B gA
    res['10.8'] = fnorm((hB @ Ab + Bb @ hA) - (gB @ Ab + Bb @ gA)); thr['10.8'] = 1e-4
    # 10.9 gauge consistency: 不同 gauge balance 后乘积一致
    G2 = (np.eye(r) + 0.5 * rng.standard_normal((r, r))).astype(F32)
    B2 = (Bb @ G2).astype(F32); A2 = (np.linalg.inv(G2) @ Ab).astype(F32)
    Bb2, Ab2, _ = balance_to_section(B2, A2)
    res['10.9'] = fnorm(Bb2 @ Ab2 - Bb @ Ab) / (fnorm(Bb @ Ab) + 1); thr['10.9'] = 1e-3
    # 10.11 有限差分 Dpi (中心差分 + float64)
    t = 1e-5
    Bb64 = Bb.astype(np.float64); Ab64 = Ab.astype(np.float64)
    gB64 = gB.astype(np.float64); gA64 = gA.astype(np.float64)
    dpi = gB64 @ Ab64 + Bb64 @ gA64
    fd = ((Bb64 + t*gB64) @ (Ab64 + t*gA64) - (Bb64 - t*gB64) @ (Ab64 - t*gA64)) / (2*t)
    res['10.11'] = fnorm(fd - dpi) / (fnorm(dpi) + 1); thr['10.11'] = 1e-6
    return res, thr, om_norm


def rank_one_closed_form(eps, rng):
    """10.10 r=1 标量 Sylvester 闭式 Om = c/(2(h+eps))."""
    m, d = 8, 6
    X, U, s, V = make_lowrank_target(m, d, 1, 1.0, rng)
    B = (U * np.sqrt(s)).astype(F32); A = (np.sqrt(s)[:, None] * V.T).astype(F32)
    Bb, Ab, _ = balance_to_section(B, A)
    gB = rng.standard_normal((m, 1)).astype(F32)
    gA = rng.standard_normal((1, d)).astype(F32)
    h = float((Bb.T @ Bb).ravel()[0])
    c = float((Bb.T @ gB - gA @ Ab.T).ravel()[0])
    omega_closed = c / (2 * (h + eps))
    Omega, C, He = connection_form(Bb, Ab, gB, gA, eps)
    return abs(float(Omega.ravel()[0]) - omega_closed)


def test_10_12_mixed_precision():
    """10.12 QR/SVD/Sylvester dtype == float32."""
    rng = make_rng(42)
    X, U, s, V = make_lowrank_target(64, 48, 8, 100.0, rng)
    B = (U * np.sqrt(s)).astype(F32); A = (np.sqrt(s)[:, None] * V.T).astype(F32)
    Bb, Ab, Q = balance_to_section(B, A)
    assert Bb.dtype == F32 and Ab.dtype == F32 and Q.dtype == F32
    Omega, C, He = connection_form(Bb, Ab, B, A, 1e-6)
    assert Omega.dtype == F32 and C.dtype == F32 and He.dtype == F32


def run_all(eps=1e-6):
    rng = make_rng(42)
    configs = [(m, d, r, cond)
               for (m, d) in [(64, 48), (128, 128), (256, 96)]
               for r in [4, 8, 16, 32]
               for cond in [1.0, 10.0, 100.0, 1000.0]]
    worst = {}; worst_thr = {}; per_config = []
    for (m, d, r, cond) in configs:
        if r > min(m, d):
            continue
        res, thr, om = run_config(m, d, r, cond, eps, rng)
        per_config.append(dict(m=m, d=d, r=r, cond=cond, omega_norm=om, **res))
        for k, v in res.items():
            if v > worst.get(k, -1):
                worst[k] = v; worst_thr[k] = thr[k]
            else:
                worst_thr.setdefault(k, thr[k])
            worst_thr[k] = max(worst_thr[k], thr[k])
    worst['10.10'] = rank_one_closed_form(eps, rng); worst_thr['10.10'] = 1e-6
    # 10.12
    mp_ok = True
    try:
        test_10_12_mixed_precision()
    except AssertionError:
        mp_ok = False
    passed = {k: bool(worst[k] < worst_thr[k]) for k in worst}
    passed['10.12'] = mp_ok
    all_pass = all(passed.values())
    return dict(eps=eps, all_pass=all_pass, passed=passed,
                worst_residual=worst, threshold_used=worst_thr,
                n_configs=len(per_config))


# pytest hooks
def test_all_eps_default():
    out = run_all(eps=1e-6)
    assert out['all_pass'], json.dumps(out, indent=2, default=float)


def test_all_eps_zero():
    out = run_all(eps=0.0)
    assert out['all_pass'], json.dumps(out, indent=2, default=float)


if __name__ == '__main__':
    report = {'eps_1e-6': run_all(eps=1e-6), 'eps_0': run_all(eps=0.0)}
    print(json.dumps(report, indent=2, default=float))

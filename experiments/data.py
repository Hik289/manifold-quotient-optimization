#!/usr/bin/env python3
"""
data.py — 合成数据生成器 + 生成自检 (H0.anchor_3 / EXP-001).

复用 EDA 已验证的 make_factors 谱结构 (sigma = logspace(0, -log10 cond)).
seed 42 固定 (项目约定 C3).

Author: ML Engineer (the project team)
"""
import numpy as np

F32 = np.float32


def make_rng(seed=42):
    return np.random.default_rng(seed)


def make_lowrank_target(m, d, r, cond=1.0, rng=None):
    """低秩目标 X* = U diag(s) V^T, cond(X*) = cond.
    返回 (X*, U, s, V)."""
    if rng is None:
        rng = make_rng()
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((d, r)))[0]
    if r == 1:
        s = np.array([1.0])
    else:
        s = np.logspace(0, -np.log10(cond), r)   # sigma_max=1 ... sigma_min=1/cond
    X = ((U * s) @ V.T).astype(F32)
    return X, U.astype(F32), s.astype(np.float64), V.astype(F32)


def init_factors(m, d, r, U=None, s=None, V=None, mode='balanced', cond=1.0, rng=None):
    """初始化 LoRA 因子 (B, A).
    mode='balanced': B = U s^{1/2}, A = s^{1/2} V^T (平衡代表, 需 U/s/V)
    mode='lora'    : B ~ 0 (small), A ~ N/sqrt(d)  (经典 LoRA 非对称 init, 最大失衡)
    mode='balanced_perturbed': 平衡代表 + 小噪声
    """
    if rng is None:
        rng = make_rng()
    if mode == 'lora':
        B = (0.01 * rng.standard_normal((m, r))).astype(F32)
        A = (rng.standard_normal((r, d)) / np.sqrt(d)).astype(F32)
    elif mode == 'balanced':
        assert U is not None and s is not None and V is not None
        B = (U * np.sqrt(s)).astype(F32)
        A = (np.sqrt(s)[:, None] * V.T).astype(F32)
    elif mode == 'balanced_perturbed':
        assert U is not None and s is not None and V is not None
        B = (U * np.sqrt(s) + 0.05 * rng.standard_normal((m, r))).astype(F32)
        A = (np.sqrt(s)[:, None] * V.T + 0.05 * rng.standard_normal((r, d))).astype(F32)
    else:
        raise ValueError(mode)
    return B, A


def make_matrix_sensing(m, d, r, cond=1.0, n_meas=None, rng=None):
    """Matrix sensing: y_i = <A_i, X*>, A_i Gaussian.
    返回 dict(Xstar, meas_ops, y, U, s, V). n_meas 默认 ~5*dof."""
    if rng is None:
        rng = make_rng()
    X, U, s, V = make_lowrank_target(m, d, r, cond, rng)
    dof = r * (m + d - r)
    if n_meas is None:
        n_meas = int(5 * dof)
    # 存 measurement 为 (n_meas, m, d) 可能大; 对 P0 用 full-observation 版本
    # (matrix factorization f(BA)=0.5||BA-X*||^2) 更省内存, 见 tasks.py.
    return dict(Xstar=X, U=U, s=s, V=V, n_meas=n_meas, dof=dof, m=m, d=d, r=r, cond=cond)


def make_deep_linear(m, d, r, depth, cond=1.0, rng=None):
    """Deep linear target: 学习一个低秩线性映射 X* (rank r), 通过 depth 层因子.
    P0 用 end-to-end product = 低秩矩阵回归代理. 返回 target spec."""
    if rng is None:
        rng = make_rng()
    X, U, s, V = make_lowrank_target(m, d, r, cond, rng)
    return dict(Xstar=X, U=U, s=s, V=V, depth=depth, m=m, d=d, r=r, cond=cond)


# ---------------------------------------------------------------------------
# 生成自检 (EXP-001 / H0.anchor_3)
# ---------------------------------------------------------------------------
def self_check(verbose=True):
    """断言生成矩阵的 rank/cond/size 与 config 一致."""
    rng = make_rng(42)
    results = []
    ok = True
    for (m, d) in [(64, 48), (128, 128), (256, 96)]:
        for r in [4, 8, 16]:
            for cond in [1.0, 10.0, 100.0, 1000.0]:
                if r > min(m, d):
                    continue
                X, U, s, V = make_lowrank_target(m, d, r, cond, rng)
                # rank
                sv = np.linalg.svd(X, compute_uv=False)
                rank_ok = int(np.sum(sv > 1e-5 * sv[0])) == r
                # cond
                cond_meas = float(sv[0] / sv[r - 1]) if r > 1 else 1.0
                cond_ok = abs(np.log10(cond_meas + 1e-12) - np.log10(cond)) < 0.15 if cond > 1 else cond_meas < 1.2
                # size
                size_ok = X.shape == (m, d)
                row = dict(m=m, d=d, r=r, cond=cond, rank_ok=bool(rank_ok),
                           cond_meas=cond_meas, cond_ok=bool(cond_ok), size_ok=bool(size_ok))
                results.append(row)
                if not (rank_ok and cond_ok and size_ok):
                    ok = False
                    if verbose:
                        print("FAIL:", row)
    if verbose:
        print(f"data self_check: {len(results)} configs, all_pass={ok}")
    return ok, results


if __name__ == '__main__':
    import json
    ok, res = self_check()
    print(json.dumps(dict(all_pass=ok, n=len(res)), indent=2))

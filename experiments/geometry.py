#!/usr/bin/env python3
"""
geometry.py — QuotientOpt 核心几何算子 (FP32, README §1/§3/§10).

设计: Phase 0-3 的 P0 任务 (matrix sensing / deep linear) 梯度解析可算,
全部几何用 numpy/scipy FP32 实现, 与 data_scientist EDA 已验证代码 bit-compatible.
Sylvester 采用对称特征分解闭式解 (H 对称 PD), 并在测试中对照 scipy.solve_sylvester.

所有几何运算 FP32 (README §10.12 / 约束 C7).
有限差分验证用中心差分 + float64 (EDA §2.3).

Author: ML Engineer (the project team)
"""
import numpy as np
from scipy.linalg import solve_sylvester

F32 = np.float32


def fnorm(M):
    return float(np.linalg.norm(np.asarray(M), 'fro'))


# ---------------------------------------------------------------------------
# Balance retraction to canonical section  B^T B = A A^T  (README §3.2)
# ---------------------------------------------------------------------------
def balance_to_section(B, A):
    """平衡收缩到 canonical section. 返回 (Bbar, Abar, Q):
    Bbar Abar = B A, Bbar^T Bbar = Abar Abar^T, Bbar = B Q, Abar = Q^{-1} A."""
    B = B.astype(F32); A = A.astype(F32)
    QB, RB = np.linalg.qr(B)              # B = QB RB
    QA, RA = np.linalg.qr(A.T)            # A^T = QA RA
    M = (RB @ RA.T).astype(F32)
    U, S, Vt = np.linalg.svd(M)
    S12 = np.sqrt(S).astype(F32)
    Bbar = (QB @ U @ np.diag(S12)).astype(F32)
    Abar = (np.diag(S12) @ Vt @ QA.T).astype(F32)
    Q = (np.linalg.pinv(B) @ Bbar).astype(F32)
    return Bbar, Abar, Q


def gram(B, A):
    """对称化 H = 0.5(B^T B + A A^T) (section 上 B^T B = A A^T)."""
    return (0.5 * (B.T @ B + A @ A.T)).astype(F32)


# ---------------------------------------------------------------------------
# Connection form via Sylvester  (H+eI)Om + Om(H+eI) = B^T xiB - xiA A^T
#   README §1.9/§3.4;  对称 H 用特征分解闭式解.
# ---------------------------------------------------------------------------
def _sylvester_sym(He, C):
    """解 He Om + Om He = C, He 对称. 闭式: Om = Q ((Q^T C Q)/(l_i+l_j)) Q^T."""
    He = He.astype(np.float64); C = C.astype(np.float64)
    lam, Qm = np.linalg.eigh(He)
    Ct = Qm.T @ C @ Qm
    denom = lam[:, None] + lam[None, :]
    Om = Qm @ (Ct / denom) @ Qm.T
    return Om


def connection_form(B, A, xiB, xiA, eps=1e-6, method='eigh'):
    """求 connection coefficient Omega. 返回 (Omega, C, He) 全 FP32.
    method='eigh' 对称闭式 (默认, GPU 友好); 'scipy' 用 Bartels-Stewart (对照)."""
    H = gram(B, A).astype(np.float64)
    He = H + eps * np.eye(H.shape[0])
    C = (B.T @ xiB - xiA @ A.T).astype(np.float64)
    if method == 'scipy':
        Omega = solve_sylvester(He, He, C)
    else:
        Omega = _sylvester_sym(He, C)
    return Omega.astype(F32), C.astype(F32), He.astype(F32)


def vertical_from_omega(B, A, Omega):
    """Fundamental vector field (B Om, -Om A) (README §1.6)."""
    return (B @ Omega).astype(F32), (-Omega @ A).astype(F32)


def horizontal_lift(B, A, xiB, xiA, eps=1e-6, method='eigh'):
    """水平投影 Pi_H(xi). 返回 (hB, hA, Omega, C, He) (README §3.4)."""
    Omega, C, He = connection_form(B, A, xiB, xiA, eps, method)
    hB = (xiB - B @ Omega).astype(F32)
    hA = (xiA + Omega @ A).astype(F32)
    return hB, hA, Omega, C, He


def vertical_energy_ratio(B, A, gB, gA, eps=1e-6, method='eigh'):
    """rho = ||Pi_V g||^2 / ||g||^2, Pi_V g = g - Pi_H g.
    ⚠️ 主指标 rho_upd 必须传入 *更新方向* u (Adam m/sqrt(v) 或 SGD momentum),
       传入 raw gradient 得到 rho_grad (sanity, 预期 ~0)."""
    hB, hA, Omega, *_ = horizontal_lift(B, A, gB, gA, eps, method)
    vB, vA = gB - hB, gA - hA
    num = fnorm(vB) ** 2 + fnorm(vA) ** 2
    den = fnorm(gB) ** 2 + fnorm(gA) ** 2
    ratio = num / den if den > 0 else 0.0
    return ratio, dict(vert_sq=num, tot_sq=den, omega_norm=fnorm(Omega))


# ---------------------------------------------------------------------------
# Gauge transport of momentum under section change  Bbar = B Q, Abar = Q^{-1} A
#   D Phi_Q [(xiB, xiA)] = (xiB Q, Q^{-1} xiA)  (README §3.3)
# ---------------------------------------------------------------------------
def transport_tangent(xiB, xiA, Q):
    """把切向量沿 gauge 变化 Q transport 到新 section."""
    Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
    return (xiB @ Q).astype(F32), (Qinv @ xiA).astype(F32)


def factor_condition_number(B, A):
    """factor condition number 几何平均 kappa(B)*kappa(A) 的 sqrt (几何平均)."""
    kB = float(np.linalg.cond(B))
    kA = float(np.linalg.cond(A))
    return np.sqrt(kB * kA), kB, kA


def balance_error(B, A):
    """||B^T B - A A^T||_F / (||B^T B||_F)."""
    return fnorm(B.T @ B - A @ A.T) / (fnorm(B.T @ B) + 1e-12)

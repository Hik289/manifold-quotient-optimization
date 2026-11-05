#!/usr/bin/env python3
"""
QuotientOpt EDA — Geometric-diagnostic measurement-protocol validation.

Implements every geometric quantity used as core evidence in the project and
verifies each against the analytically-known answer / self-check threshold from
QuotientOpt_README_manifold_deep.md §10 (Numerical Tests).

All geometry runs in FP32 (§10.12). We use numpy (double for reference, single
for the protocol-under-test) + scipy.linalg.solve_sylvester.

Author: Data Scientist (the project team)
"""
import numpy as np
from scipy.linalg import solve_sylvester
import json, time, sys

RNG = np.random.default_rng(42)          # seed 42 (project convention)
F32 = np.float32
TOL = 1e-6                                # §10 threshold

def fnorm(M):
    return float(np.linalg.norm(M, 'fro'))

# ----------------------------------------------------------------------------
# Core geometry (FP32) -- exactly the operators the optimizer will use.
# ----------------------------------------------------------------------------
def balance_to_section(B, A):
    """Balance retraction to canonical section B^T B = A A^T (README §3.2).
    Returns (Bbar, Abar, Q) with Bbar Abar = B A and Bbar^T Bbar = Abar Abar^T.
    Q is the gauge s.t. Bbar = B Q, Abar = Q^{-1} A."""
    B = B.astype(F32); A = A.astype(F32)
    QB, RB = np.linalg.qr(B)                 # B = QB RB, QB: m x r
    QA, RA = np.linalg.qr(A.T)               # A^T = QA RA, QA: d x r
    M = (RB @ RA.T).astype(F32)              # r x r
    U, S, Vt = np.linalg.svd(M)
    S12 = np.sqrt(S).astype(F32)
    Bbar = (QB @ U @ np.diag(S12)).astype(F32)
    Abar = (np.diag(S12) @ Vt @ QA.T).astype(F32)
    # Gauge Q s.t. Bbar = B Q  => Q = pinv(B) Bbar
    Q = (np.linalg.pinv(B) @ Bbar).astype(F32)
    return Bbar, Abar, Q

def gram(B, A):
    """Symmetrized H = B^T B ( = A A^T on the section)."""
    H = 0.5 * (B.T @ B + A @ A.T)
    return H.astype(F32)

def connection_form(B, A, xiB, xiA, eps):
    """Solve (H+eI) Omega + Omega (H+eI) = B^T xiB - xiA A^T  (README §1.9/§3.4)."""
    H = gram(B, A).astype(np.float64)
    He = H + eps * np.eye(H.shape[0])
    C = (B.T @ xiB - xiA @ A.T).astype(np.float64)
    Omega = solve_sylvester(He, He, C)       # He Omega + Omega He = C
    return Omega.astype(F32), C.astype(F32), He.astype(F32)

def horizontal_lift(B, A, xiB, xiA, eps):
    """Return horizontal component (README §3.4)."""
    Omega, C, He = connection_form(B, A, xiB, xiA, eps)
    hB = (xiB - B @ Omega).astype(F32)
    hA = (xiA + Omega @ A).astype(F32)
    return hB, hA, Omega, C, He

def vertical_from_omega(B, A, Omega):
    """Fundamental vector field (B Omega, -Omega A)  (README §1.6)."""
    return (B @ Omega).astype(F32), (-Omega @ A).astype(F32)

def vertical_energy_ratio(B, A, gB, gA, eps):
    """||Pi_V G||^2 / ||G||^2  where Pi_V G = G - Pi_H G."""
    hB, hA, *_ = horizontal_lift(B, A, gB, gA, eps)
    vB, vA = gB - hB, gA - hA
    num = fnorm(vB)**2 + fnorm(vA)**2
    den = fnorm(gB)**2 + fnorm(gA)**2
    return num / den, (num, den)

def make_factors(m, d, r, cond=1.0, balanced=False, gauge=None):
    """Build (B,A) with target X=BA having prescribed condition number.
    If balanced, start from the balanced representative of X."""
    U = np.linalg.qr(RNG.standard_normal((m, r)))[0]
    V = np.linalg.qr(RNG.standard_normal((d, r)))[0]
    if r == 1:
        s = np.array([1.0])
    else:
        s = np.logspace(0, -np.log10(cond), r)   # sigma_max=1 ... sigma_min=1/cond
    X = (U * s) @ V.T
    B = (U * np.sqrt(s)).astype(F32)             # balanced rep
    A = (np.sqrt(s)[:, None] * V.T).astype(F32)
    if gauge is not None:
        B = (B @ gauge).astype(F32)
        A = (np.linalg.inv(gauge) @ A).astype(F32)
    return B.astype(F32), A.astype(F32), X.astype(F32), s

# ----------------------------------------------------------------------------
# §10 self-checks
# ----------------------------------------------------------------------------
def run_checks(m, d, r, cond, eps):
    """Run every §10 numerical test for one (m,d,r,cond) configuration."""
    res = {}
    # generic factors (perturb balanced rep by random gauge to make it generic)
    G0 = (np.eye(r) + 0.3 * RNG.standard_normal((r, r))).astype(F32)
    B, A, X, s = make_factors(m, d, r, cond=cond, gauge=G0)
    # a random tangent / gradient direction
    gB = RNG.standard_normal((m, r)).astype(F32)
    gA = RNG.standard_normal((r, d)).astype(F32)

    # 10.1 product preservation of balance
    Bb, Ab, Q = balance_to_section(B, A)
    res['10.1_product_preservation'] = fnorm(Bb @ Ab - B @ A) / (fnorm(B @ A) + 1)
    # 10.2 balance
    res['10.2_balance_error'] = fnorm(Bb.T @ Bb - Ab @ Ab.T)
    # 10.3 vertical kernel: (B Om)A + B(-Om A) = 0
    Om = RNG.standard_normal((r, r)).astype(F32)
    vB, vA = vertical_from_omega(Bb, Ab, Om)
    res['10.3_vertical_kernel'] = fnorm(vB @ Ab + Bb @ vA)
    # 10.4 Sylvester residual (on balanced rep)
    Omega, C, He = connection_form(Bb, Ab, gB, gA, eps)
    res['10.4_sylvester_residual'] = fnorm(He @ Omega + Omega @ He - C) / (fnorm(C) + 1)
    # 10.5 horizontal condition: B^T hB - hA A^T = 0
    hB, hA, *_ = horizontal_lift(Bb, Ab, gB, gA, eps)
    res['10.5_horizontal_condition'] = fnorm(Bb.T @ hB - hA @ Ab.T)
    # 10.6 projector idempotence
    hB2, hA2, *_ = horizontal_lift(Bb, Ab, hB, hA, eps)
    res['10.6_idempotence'] = (fnorm(hB2 - hB) + fnorm(hA2 - hA))
    # 10.7 pure vertical -> zero
    pvB, pvA, *_ = horizontal_lift(Bb, Ab, vB, vA, eps)
    res['10.7_pure_vertical_zero'] = fnorm(pvB) + fnorm(pvA)
    # 10.8 product differential preservation: hB A + B hA == gB A + B gA
    lhs = hB @ Ab + Bb @ hA
    rhs = gB @ Ab + Bb @ gA
    res['10.8_product_diff_preservation'] = fnorm(lhs - rhs)
    # 10.11 finite difference of Dpi
    t = 1e-4
    dpi = gB @ Ab + Bb @ gA
    fd = ((Bb + t*gB) @ (Ab + t*gA) - Bb @ Ab) / t
    res['10.11_finite_difference'] = fnorm(fd - dpi) / (fnorm(dpi) + 1)
    # extra: vertical energy ratio of a *generic* gradient (should be >0)
    ver, _ = vertical_energy_ratio(Bb, Ab, gB, gA, eps)
    res['aux_vertical_energy_ratio_generic'] = ver
    # extra: vertical energy of the Euclidean LoRA gradient  gB=G_X A^T, gA=B^T G_X
    GX = RNG.standard_normal((m, d)).astype(F32)
    loraB = (GX @ Ab.T).astype(F32); loraA = (Bb.T @ GX).astype(F32)
    ver_lora, _ = vertical_energy_ratio(Bb, Ab, loraB, loraA, eps)
    res['aux_vertical_energy_ratio_loragrad'] = ver_lora
    # extra: factor condition number of balanced rep (should ~ sqrt(cond(X)))
    res['aux_kappa_Bbar'] = float(np.linalg.cond(Bb))
    res['aux_sqrt_cond_target'] = float(np.sqrt(cond)) if cond > 0 else 1.0
    # extra: gauge product disagreement -- balance two gauges of same X, compare products
    G2 = (np.eye(r) + 0.5 * RNG.standard_normal((r, r))).astype(F32)
    B2 = (Bb @ G2).astype(F32); A2 = (np.linalg.inv(G2) @ Ab).astype(F32)
    Bb2, Ab2, _ = balance_to_section(B2, A2)
    res['10.9_gauge_consistency'] = fnorm(Bb2 @ Ab2 - Bb @ Ab) / (fnorm(Bb @ Ab) + 1)
    return res

def rank_one_closed_form(eps):
    """10.10 r=1 scalar Sylvester closed form."""
    m, d = 8, 6
    B, A, X, s = make_factors(m, d, 1, cond=1.0)
    gB = RNG.standard_normal((m, 1)).astype(F32)
    gA = RNG.standard_normal((1, d)).astype(F32)
    Bb, Ab, _ = balance_to_section(B, A)
    h = float((Bb.T @ Bb).ravel()[0])          # scalar H
    c = float((Bb.T @ gB - gA @ Ab.T).ravel()[0])
    omega_closed = c / (2*(h+eps))             # (h+e)w + w(h+e) = c
    Omega, C, He = connection_form(Bb, Ab, gB, gA, eps)
    return abs(float(Omega.ravel()[0]) - omega_closed)

def main():
    configs = [(m, d, r, cond)
               for (m, d) in [(64, 48), (128, 128), (256, 96)]
               for r in [1, 4, 8, 16, 32]
               for cond in [1.0, 10.0, 100.0, 1000.0]]
    eps = 1e-6
    all_rows = []
    worst = {}
    t0 = time.time()
    for (m, d, r, cond) in configs:
        if r > min(m, d):
            continue
        row = run_checks(m, d, r, cond, eps)
        row.update(dict(m=m, d=d, r=r, cond=cond))
        all_rows.append(row)
        for k, v in row.items():
            if k.startswith('10.') and isinstance(v, float):
                worst[k] = max(worst.get(k, 0.0), v)
    worst['10.10_rank_one_closed_form'] = rank_one_closed_form(eps)

    # damping sweep: connection error vs eps at fixed near-rank-deficient config
    damp_rows = []
    for cond in [1.0, 1e2, 1e4, 1e6]:
        for eps in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            B, A, X, s = make_factors(64, 48, 8, cond=cond,
                                      gauge=(np.eye(8)+0.3*RNG.standard_normal((8,8))).astype(F32))
            Bb, Ab, _ = balance_to_section(B, A)
            gB = RNG.standard_normal((64, 8)).astype(F32)
            gA = RNG.standard_normal((8, 48)).astype(F32)
            Omega, C, He = connection_form(Bb, Ab, gB, gA, eps)
            syl_res = fnorm(He@Omega+Omega@He-C)/(fnorm(C)+1)
            sig_min = float(np.min(np.linalg.svd(X, compute_uv=False)))
            damp_rows.append(dict(cond=cond, eps=eps, sylvester_residual=syl_res,
                                  sigma_min_X=sig_min, omega_norm=fnorm(Omega)))

    # vertical energy vs condition number & gauge magnitude (signal-strength map)
    signal_rows = []
    for cond in [1.0, 10.0, 100.0, 1000.0]:
        for gmag in [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]:
            gauge = (gmag * np.eye(8)).astype(F32)   # scalar gauge
            B, A, X, s = make_factors(64, 48, 8, cond=cond, gauge=gauge)
            Bb, Ab, _ = balance_to_section(B, A)  # balance removes scalar gauge
            # measure vertical energy on the RAW (un-balanced) factors -- what a
            # gauge-blind optimizer like AdamW actually sees.
            GX = RNG.standard_normal((64, 48)).astype(F32)
            rawB = (GX @ A.T).astype(F32); rawA = (B.T @ GX).astype(F32)
            ver_raw, _ = vertical_energy_ratio(B, A, rawB, rawA, 1e-6)
            balB = (GX @ Ab.T).astype(F32); balA = (Bb.T @ GX).astype(F32)
            ver_bal, _ = vertical_energy_ratio(Bb, Ab, balB, balA, 1e-6)
            signal_rows.append(dict(cond=cond, gauge_mag=gmag,
                                    vert_ratio_raw=ver_raw, vert_ratio_balanced=ver_bal))

    out = dict(elapsed_sec=time.time()-t0,
               n_configs=len(all_rows),
               worst_case_selfcheck=worst,
               threshold=TOL,
               all_pass=all(v < TOL for k, v in worst.items()),
               per_config=all_rows,
               damping_sweep=damp_rows,
               signal_map=signal_rows)
    print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    main()

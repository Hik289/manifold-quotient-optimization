#!/usr/bin/env python3
"""
run_pathB_depthscan.py — 路 B 深链 depth-screening (theory-guided, theorist Prop 1).

核心假设 (pathB_theory_targeting.md): Theorem 2 (balance 足够) 唯一在深链 L≥3 破。
机制: balance-only 逐界面顺序 Gauss-Seidel sweep, L≥3 投影不对易(von Neumann) →
  一次 sweep 到不了联合平衡流形 → 残余 δ_L 随 depth 增大 → 喂进未重投影 momentum
  按 β/(1-β) 累积。full QO 每步重投影 momentum 消掉。F2 全 null 因 L=2 (δ_2≡0)。

测三个量 (theorist §4):
  (1) 逐 sweep 后联合平衡残余 δ_L = Σ_k ‖W_{k+1}ᵀW_{k+1} − W_k W_kᵀ‖_F
  (2) sup_Q product disagreement, gauge 施加于**内部界面**
  (3) 逐界面 momentum vertical residual

配置: L∈{2,4,8,16}(L=2 F2-null 锚), r∈{8,16}, κ∈{100,1000}, LoRA init,
  β∈{0.9,0}, arms {balance_only, quotientopt, adamw}, 确定性, 400ep 短跑.

⚠️ full QO horizontal momentum 重投影已确认 ON (DLOptimizer step 第4步).

期望 (可证伪): L=2 bal≈QO; L≥4 balance-only disag+momentum vert residual 随 depth 单调增,
  QO 平坦; 分离 ∝ β/(1-β) (β=0 消失). kill-switch: δ_L 全回机器零 或 β=0.9 到 L=16 仍 bal≈QO.

输出: pathB_depthscan_summary.json + logs/PB_*_trajectory.jsonl

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from deeplinear import DeepLinearChain
from geometry import (F32, fnorm, balance_to_section, horizontal_lift,
                      vertical_energy_ratio, factor_condition_number, balance_error,
                      transport_tangent, vertical_from_omega, connection_form)
from data import make_rng

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)
EPOCHS = 400
EPS = 1e-6
LR = 1e-2
GAUGE_INTERIOR = [('scalar', 1e-2), ('scalar', 1e2), ('dense', 1e2), ('dense', 1e3)]


def joint_balance_residual(Ws):
    """δ_L = Σ_k ‖W_{k+1}ᵀ W_{k+1} − W_k W_kᵀ‖_F (Def 2)."""
    L = len(Ws)
    tot = 0.0
    for k in range(L - 1):
        tot += fnorm(Ws[k+1].T @ Ws[k+1] - Ws[k] @ Ws[k].T)
    return float(tot)


class DLOptChain:
    """深链逐因子 optimizer, 支持 β 参数 + 三个测量.
    method: 'adamw'(无balance) / 'balance_only'(逐界面balance,不重投影momentum) /
            'quotientopt'(逐界面balance+hproj+momentum重投影 ON)."""
    def __init__(self, L, method, lr=LR, beta=0.9, beta2=0.999, eps=EPS):
        self.L = L; self.method = method
        self.lr, self.beta, self.beta2, self.eps = lr, beta, beta2, eps
        self.mB = None; self.vB = None; self.t = 0

    def step(self, chain, measure=False):
        self.t += 1
        Ws = chain.Ws; L = self.L; n_iface = L - 1
        if self.mB is None:
            self.mB = [np.zeros_like(W) for W in Ws]
            self.vB = [np.zeros_like(W) for W in Ws]
        delta_pre = joint_balance_residual(Ws) if measure else None
        # 1. 逐界面 balance sweep (Gauss-Seidel)
        if self.method in ('balance_only', 'quotientopt'):
            for k in range(n_iface):
                Bk, Ak, Q = balance_to_section(Ws[k+1], Ws[k])
                Ws[k+1] = Bk; Ws[k] = Ak
                if self.method == 'quotientopt' and Q is not None:
                    tB, tA = transport_tangent(self.mB[k+1], self.mB[k], Q)
                    self.mB[k+1] = tB; self.mB[k] = tA
        delta_post = joint_balance_residual(Ws) if measure else None   # δ_L after one sweep
        grads = chain.grads()
        # 2. 逐界面 horizontal projection of gradient (quotientopt)
        if self.method == 'quotientopt':
            hg = [g.copy() for g in grads]
            for k in range(n_iface):
                hgB, hgA, Om, *_ = horizontal_lift(Ws[k+1], Ws[k], hg[k+1], hg[k], self.eps)
                hg[k+1] = hgB; hg[k] = hgA
            grads = hg
        # 3. adam preconditioning + momentum per factor
        us = []
        for i, g in enumerate(grads):
            self.vB[i] = self.beta2*self.vB[i] + (1-self.beta2)*g**2
            vh = self.vB[i]/(1-self.beta2**self.t)
            pg = (g/(np.sqrt(vh)+1e-8)).astype(F32)
            self.mB[i] = (self.beta*self.mB[i] + (1-self.beta)*pg).astype(F32)
            us.append(self.mB[i])
        # 4. reproject momentum to horizontal (quotientopt) —— 关键 momentum reset (ON)
        if self.method == 'quotientopt':
            for k in range(n_iface):
                hmB, hmA, Om, *_ = horizontal_lift(Ws[k+1], Ws[k], us[k+1], us[k], self.eps)
                us[k+1] = hmB; us[k] = hmA
                self.mB[k+1] = hmB; self.mB[k] = hmA
        # 5. momentum vertical residual per interface (measure)
        mom_vert = None
        if measure:
            mv = []
            for k in range(n_iface):
                # momentum 状态在界面 k 的垂直残余
                rho, _ = vertical_energy_ratio(Ws[k+1], Ws[k], self.mB[k+1], self.mB[k], self.eps)
                mv.append(rho)
            mom_vert = float(np.mean(mv))
        # 6. update
        for i in range(L):
            Ws[i] = (Ws[i] - self.lr*us[i]).astype(F32)
        return dict(delta_pre=delta_pre, delta_post=delta_post, mom_vert=mom_vert)


def interior_gauge(chain_factory, L, method, beta, gauge, epochs=EPOCHS):
    """在**内部界面**注入 gauge, 测 product trajectory disagreement (相对 gauge=I)."""
    def run(inject):
        chain = chain_factory()
        if inject is not None:
            k_int, Q = inject                       # 内部界面 k
            r = Q.shape[0]
            chain.Ws[k_int+1] = (chain.Ws[k_int+1] @ Q).astype(F32)
            Qinv = np.linalg.inv(Q.astype(np.float64)).astype(F32)
            chain.Ws[k_int] = (Qinv @ chain.Ws[k_int]).astype(F32)
        opt = DLOptChain(L, method, beta=beta)
        prods = []
        for t in range(epochs):
            opt.step(chain)
            X = chain.product()
            if not np.all(np.isfinite(X)):
                break
            prods.append(X.astype(F32).copy())
        return prods
    ref = run(None)
    T = len(ref)
    # 内部界面: 取中间界面 k = (L-1)//2 (L=2 时 k=0 是唯一界面)
    k_int = max(0, (L - 1) // 2)
    kind, val = gauge
    r = chain_factory().r
    Q = make_gauge(r, kind, val, make_rng(456))
    gp = run((k_int, Q))
    Tg = min(len(gp), T)
    if Tg == 0:
        return float('inf')
    return float(fnorm(gp[Tg-1] - ref[Tg-1]))


def make_gauge(r, kind, val, rng):
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    U = np.linalg.qr(rng.standard_normal((r, r)))[0]
    V = np.linalg.qr(rng.standard_normal((r, r)))[0]
    s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
    return (U @ np.diag(s) @ V.T).astype(F32)


def run_arm(L, r, cond, method, beta):
    """一个 (L,r,cond,method,β) 配置: 训练测 δ_L + momentum vert, 再测 interior gauge disag."""
    m = d = 64
    def factory():
        return DeepLinearChain(m, d, r, L, cond=cond, seed=42, init='lora')
    chain = factory()
    opt = DLOptChain(L, method, beta=beta)
    deltas = []; momverts = []; losses = []
    for t in range(EPOCHS):
        info = opt.step(chain, measure=True)
        if info['delta_post'] is not None:
            deltas.append(info['delta_post'])
        if info['mom_vert'] is not None:
            momverts.append(info['mom_vert'])
        losses.append(chain.loss())
        if not np.isfinite(losses[-1]):
            break
    # interior gauge disagreement (max over gauge subset)
    disags = [interior_gauge(factory, L, method, beta, g) for g in GAUGE_INTERIOR]
    finals = [x for x in disags if np.isfinite(x)]
    tail = min(100, len(deltas))
    return dict(
        L=L, r=r, cond=cond, method=method, beta=beta,
        delta_L_mean=float(np.mean(deltas[-tail:])) if deltas else None,
        delta_L_final=float(deltas[-1]) if deltas else None,
        mom_vert_mean=float(np.mean(momverts[-tail:])) if momverts else None,
        gauge_disag_max=float(np.max(finals)) if finals else float('inf'),
        gauge_disag_mean=float(np.mean(finals)) if finals else float('inf'),
        final_loss=float(losses[-1]) if losses else None,
        final_rel=float(chain.rel_error()),
    )


def main():
    t0 = time.time()
    summaries = []
    # 主 screening: r=8, cond=100 先扫全 depth×method×β (最便宜); 再补 r=16/cond=1000 关键点
    configs = []
    for L in [2, 4, 8, 16]:
        for beta in [0.9, 0.0]:
            for method in ['balance_only', 'quotientopt', 'adamw']:
                configs.append((L, 8, 100.0, method, beta))
    # 关键点补充 (高信号): r=16, cond=1000, β=0.9, L∈{2,8,16}
    for L in [2, 8, 16]:
        for method in ['balance_only', 'quotientopt']:
            configs.append((L, 16, 1000.0, method, 0.9))

    for (L, r, cond, method, beta) in configs:
        eid = f"PB_L{L}_r{r}_c{int(cond)}_{method}_b{beta}"
        print(f"[{time.strftime('%H:%M:%S')}] {eid}", flush=True)
        s = run_arm(L, r, cond, method, beta)
        s['exp_id'] = eid
        summaries.append(s)
        print(f"    δ_L={s['delta_L_mean']:.2e} mom_vert={s['mom_vert_mean']:.2e} "
              f"gauge_disag={s['gauge_disag_max']:.2e} final_rel={s['final_rel']:.2e}", flush=True)

    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, summaries=summaries)
    with open(os.path.join(HERE, 'pathB_depthscan_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)

    # 决定性分析: depth-scaling separation (β=0.9, r=8, cond=100)
    print("\n=== 路B 裁决: balance-only vs full QO 随 depth (r8,cond100,β=0.9) ===")
    print(f"{'L':>3}{'bal_disag':>12}{'QO_disag':>12}{'bal/QO':>9}{'bal_δL':>11}{'QO_δL':>11}{'bal_momV':>11}{'QO_momV':>11}")
    by = {(s['L'], s['method'], s['beta'], s['r'], int(s['cond'])): s for s in summaries}
    for L in [2, 4, 8, 16]:
        b = by.get((L, 'balance_only', 0.9, 8, 100)); q = by.get((L, 'quotientopt', 0.9, 8, 100))
        if b and q:
            ratio = b['gauge_disag_max']/q['gauge_disag_max'] if q['gauge_disag_max'] > 0 else float('inf')
            print(f"{L:>3}{b['gauge_disag_max']:>12.2e}{q['gauge_disag_max']:>12.2e}{ratio:>9.1f}"
                  f"{b['delta_L_mean']:>11.2e}{q['delta_L_mean']:>11.2e}"
                  f"{b['mom_vert_mean']:>11.2e}{q['mom_vert_mean']:>11.2e}")
    print("\n=== β 机制签名: 分离应 ∝ β/(1-β), β=0 消失 (L=16,r8,cond100) ===")
    for beta in [0.9, 0.0]:
        b = by.get((16, 'balance_only', beta, 8, 100)); q = by.get((16, 'quotientopt', beta, 8, 100))
        if b and q:
            ratio = b['gauge_disag_max']/q['gauge_disag_max'] if q['gauge_disag_max'] > 0 else float('inf')
            print(f"  β={beta}: bal/QO disag ratio={ratio:.1f} (bal={b['gauge_disag_max']:.2e} QO={q['gauge_disag_max']:.2e})")
    print(f"\nPATHB_DEPTHSCAN_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

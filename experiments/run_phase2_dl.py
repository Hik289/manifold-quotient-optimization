#!/usr/bin/env python3
"""
run_phase2_dl.py — Phase 2 deep linear 长训练 (≥2000 epoch).

Director 裁决: deep linear 是 analysis task (展示 depth 放大 gauge drift), 核心 setting 仍是 two-factor.
配置: cond{100,1000} × depth{8,16}, methods {adamw, balance_only, quotientopt_perinterface}, ≥2000ep.
观测: 任务层 full QO vs balance-only 差距是否随 depth 拉大 (多界面下 balance 是否不足).

⚠️ CAVEAT (已向 Director 说明, 待确认): deep linear 上的 "QO" 是**逐界面 heuristic**——
   对每个相邻界面 (W_{i+1},W_i) 独立应用现有 two-factor balance+horizontal projection+momentum,
   顺序应用。复用现有 two-factor 算子、无需新理论, 但忽略界面间耦合, 是 heuristic 近似,
   **不是严格多界面 QO** (严格版需 (GL(r))^{L-1} 联合 connection = theorist 的活)。
   balance_only 同理是逐界面 balance。结果标注此 caveat。

输出: results.json(append) + logs/P2_DL_*_trajectory.jsonl + phase2_dl_summary.json

Author: ML Engineer (the project team)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from deeplinear import DeepLinearChain
from geometry import (F32, fnorm, balance_to_section, horizontal_lift,
                      vertical_energy_ratio, factor_condition_number, balance_error,
                      transport_tangent)

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

EPOCHS = 2000
EPS = 1e-6
LR = 1e-2
TARGET_REL = 1e-3   # deep linear 更难, target 放宽到 1e-3


class DLOptimizer:
    """deep linear 逐因子 optimizer. method:
       'adamw'         : 逐因子 AdamW, 无 balance
       'balance_only'  : 逐界面 balance + 逐因子 AdamW (不投影 update)
       'quotientopt'   : 逐界面 balance + horizontal projection + horizontal momentum (heuristic)
    """
    def __init__(self, n_factors, method, lr=LR, beta=0.9, beta2=0.999, eps=EPS):
        self.L = n_factors; self.method = method
        self.lr, self.beta, self.beta2, self.eps = lr, beta, beta2, eps
        self.mB = None; self.vB = None; self.t = 0

    def step(self, chain):
        self.t += 1
        Ws = chain.Ws
        L = self.L
        n_iface = L - 1
        if self.mB is None:
            self.mB = [np.zeros_like(W) for W in Ws]
            self.vB = [np.zeros_like(W) for W in Ws]
        # 1. 逐界面 balance (balance_only / quotientopt)
        if self.method in ('balance_only', 'quotientopt'):
            for k in range(n_iface):
                Bk, Ak, Q = balance_to_section(Ws[k+1], Ws[k])
                Ws[k+1] = Bk; Ws[k] = Ak
                # transport momentum across this interface (quotientopt)
                if self.method == 'quotientopt' and Q is not None:
                    tB, tA = transport_tangent(self.mB[k+1], self.mB[k], Q)
                    self.mB[k+1] = tB; self.mB[k] = tA
        grads = chain.grads()
        # 2. 逐界面 horizontal projection of gradient (quotientopt only)
        if self.method == 'quotientopt':
            hg = [g.copy() for g in grads]
            for k in range(n_iface):
                hgB, hgA, Om, *_ = horizontal_lift(Ws[k+1], Ws[k], hg[k+1], hg[k], self.eps)
                hg[k+1] = hgB; hg[k] = hgA
            grads = hg
        # 3. adam preconditioning per factor
        us = []
        for i, g in enumerate(grads):
            self.vB[i] = self.beta2*self.vB[i] + (1-self.beta2)*g**2
            vh = self.vB[i]/(1-self.beta2**self.t)
            pg = (g/(np.sqrt(vh)+1e-8)).astype(F32)
            # momentum
            self.mB[i] = (self.beta*self.mB[i] + (1-self.beta)*pg).astype(F32)
            us.append(self.mB[i])
        # 4. reproject momentum to horizontal (quotientopt)
        if self.method == 'quotientopt':
            for k in range(n_iface):
                hmB, hmA, Om, *_ = horizontal_lift(Ws[k+1], Ws[k], us[k+1], us[k], self.eps)
                us[k+1] = hmB; us[k] = hmA
                self.mB[k+1] = hmB; self.mB[k] = hmA
        # 5. update
        for i in range(L):
            Ws[i] = (Ws[i] - self.lr*us[i]).astype(F32)
        # measure: mean over interfaces of update vertical energy + balance err + factor cond
        vert = []; bal = []
        for k in range(n_iface):
            rho, _ = vertical_energy_ratio(Ws[k+1], Ws[k], us[k+1], us[k], self.eps)
            vert.append(rho); bal.append(balance_error(Ws[k+1], Ws[k]))
        kfac, _, _ = factor_condition_number(Ws[-1], Ws[0])
        return dict(rho_upd_mean=float(np.mean(vert)), rho_upd_max=float(np.max(vert)),
                    balance_mean=float(np.mean(bal)), kfac=float(kfac),
                    us=us)


def train_dl(m, d, r, depth, cond, method, epochs=EPOCHS):
    chain = DeepLinearChain(m, d, r, depth, cond=cond, seed=42, init='lora')
    opt = DLOptimizer(depth, method)
    exp_id = f"P2_DL_{method}_cond{int(cond)}_d{depth}"
    fpath = os.path.join(LOGDIR, f"{exp_id}_trajectory.jsonl")
    rhos = []; kfacs = []; losses = []; rels = []
    ep2tgt = None
    with open(fpath, 'w') as fout:
        for t in range(epochs):
            t0 = time.time()
            info = opt.step(chain)
            loss = chain.loss(); rel = chain.rel_error()
            if not np.isfinite(loss):
                break
            if ep2tgt is None and rel < TARGET_REL:
                ep2tgt = t
            M = dict(epoch=t, rho_upd_mean=info['rho_upd_mean'], rho_upd_max=info['rho_upd_max'],
                     balance_mean=info['balance_mean'], factor_condition=info['kfac'],
                     train_loss=loss, rel_error=rel, wall=time.time()-t0)
            fout.write(json.dumps(M, default=float) + '\n')
            rhos.append(info['rho_upd_mean']); kfacs.append(info['kfac'])
            losses.append(loss); rels.append(rel)
    summary = dict(
        exp_id=exp_id, method=method, cond=cond, depth=depth, epochs=len(losses),
        rho_upd_mean=float(np.mean(rhos)), rho_upd_final=float(rhos[-1]),
        kfac_final=float(kfacs[-1]), kfac_mean=float(np.mean(kfacs)),
        final_loss=float(losses[-1]), best_loss=float(np.min(losses)),
        final_rel=float(rels[-1]), best_rel=float(np.min(rels)),
        loss_auc=float(np.trapezoid(losses)), epoch_to_target=ep2tgt,
        trajectory_file=f"logs/{exp_id}_trajectory.jsonl",
    )
    return summary


def main():
    t0 = time.time()
    methods = ['adamw', 'balance_only', 'quotientopt']
    summaries = []
    for cond in [100.0, 1000.0]:
        for depth in [8, 16]:
            for method in methods:
                print(f"[{time.strftime('%H:%M:%S')}] start P2_DL_{method}_cond{int(cond)}_d{depth}", flush=True)
                s = train_dl(128, 128, 8, depth, cond, method)
                summaries.append(s)
                print(f"  done: ρ_upd={s['rho_upd_mean']:.2e} kfac={s['kfac_final']:.1f} "
                      f"final_rel={s['final_rel']:.2e} best_rel={s['best_rel']:.2e} ep2tgt={s['epoch_to_target']}", flush=True)
    out = dict(elapsed_sec=time.time()-t0, epochs=EPOCHS, summaries=summaries,
               caveat="deep linear QO = per-interface heuristic (reuses two-factor operators, "
                      "ignores inter-interface coupling, NOT strict multi-interface QO)")
    with open(os.path.join(HERE, 'phase2_dl_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    # F2 depth-regime print
    print("\n=== F2 depth-regime: full QO vs balance-only vs AdamW (final_rel / best_rel) ===")
    by = {(s['method'], int(s['cond']), s['depth']): s for s in summaries}
    for cond in [100, 1000]:
        for depth in [8, 16]:
            a = by[('adamw', cond, depth)]; b = by[('balance_only', cond, depth)]; q = by[('quotientopt', cond, depth)]
            print(f"cond{cond} d{depth}: adamw rel={a['final_rel']:.2e}/{a['best_rel']:.2e} | "
                  f"bal={b['final_rel']:.2e}/{b['best_rel']:.2e} | QO={q['final_rel']:.2e}/{q['best_rel']:.2e} | "
                  f"QO/bal ρ_upd={q['rho_upd_mean']:.1e}/{b['rho_upd_mean']:.1e}")
    print(f"\nPHASE2_DL_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

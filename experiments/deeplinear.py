#!/usr/bin/env python3
"""
deeplinear.py — 真实 depth-L 线性链 (裁决 1: DeepLinear 重构为真实深链).

链: X = W_L @ W_{L-1} @ ... @ W_1,  W_1: r×d, W_L: m×r, 中间 W_i: r×r.
目标: 拟合低秩 X* (rank r).  f = 0.5 ||X - X*||^2.
gauge: 每个相邻界面 (W_{i+1}, W_i) 有 GL(r) 冗余 (W_{i+1} G, G^{-1} W_i), 共 (L-1) 个界面.

depth 现在**真正进入动力学**: 梯度经 L 层链式传播, 失衡在各界面累积.
诊断: 逐因子 AdamW (baseline) / 逐界面 balance (balance-only) 下, 测各界面的
       vertical energy / balance error / 条件数, 看 depth{4,8,16} 是否单调可区分.

Author: ML Engineer (the project team)
"""
import numpy as np
from geometry import (F32, fnorm, balance_to_section, horizontal_lift,
                      vertical_energy_ratio, factor_condition_number, balance_error,
                      transport_tangent)
from data import make_rng, make_lowrank_target


class DeepLinearChain:
    """depth-L 线性链, 拟合低秩目标. 因子 [W_1(r×d), W_2..W_{L-1}(r×r), W_L(m×r)]."""
    def __init__(self, m, d, r, depth, cond=1.0, seed=42, init='lora'):
        assert depth >= 2
        self.rng = make_rng(seed)
        self.X, self.U, self.s, self.V = make_lowrank_target(m, d, r, cond, self.rng)
        self.m, self.d, self.r, self.depth, self.cond = m, d, r, depth, cond
        self.Ws = self._init_weights(init)

    def _init_weights(self, mode):
        r, m, d, L = self.r, self.m, self.d, self.depth
        Ws = []
        # W_1: r×d
        # W_2..W_{L-1}: r×r
        # W_L: m×r
        shapes = [(r, d)] + [(r, r)] * (L - 2) + [(m, r)]
        for i, sh in enumerate(shapes):
            if mode == 'lora':
                # 非对称: 最后一层近零 (类 LoRA B≈0), 其余 ~N/sqrt(fanin)
                if i == L - 1:
                    W = 0.01 * self.rng.standard_normal(sh)
                else:
                    W = self.rng.standard_normal(sh) / np.sqrt(sh[1])
            else:  # balanced-ish small
                W = self.rng.standard_normal(sh) / np.sqrt(sh[1])
            Ws.append(W.astype(F32))
        return Ws

    def product(self, Ws=None):
        Ws = Ws or self.Ws
        X = Ws[0]
        for W in Ws[1:]:
            X = W @ X
        return X

    def loss(self, Ws=None):
        return 0.5 * float(np.linalg.norm(self.product(Ws) - self.X, 'fro') ** 2)

    def rel_error(self, Ws=None):
        X = self.product(Ws)
        return float(np.linalg.norm(X - self.X, 'fro') / (np.linalg.norm(self.X, 'fro') + 1e-12))

    def grads(self, Ws=None):
        """链式法则: dL/dW_i.  X = W_L...W_1, G_X = X - X*.
        dL/dW_i = (W_L...W_{i+1})^T G_X (W_{i-1}...W_1)^T."""
        Ws = Ws or self.Ws
        L = self.depth
        # prefix[i] = W_{i-1}...W_1 (输出到 W_i 的输入), prefix[0]=I_d (概念上)
        # 计算 partial products
        # left[i] = W_L @ ... @ W_{i+1}  (i from 0..L-1; left[L-1]=I_m)
        # right[i] = W_{i-1} @ ... @ W_1 (right[0]=I_d)
        right = [None] * L
        acc = np.eye(self.d, dtype=F32)
        right[0] = acc
        for i in range(1, L):
            acc = Ws[i - 1] @ acc
            right[i] = acc
        left = [None] * L
        acc = np.eye(self.m, dtype=F32)
        left[L - 1] = acc
        for i in range(L - 2, -1, -1):
            acc = acc @ Ws[i + 1]
            left[i] = acc
        Gx = (self.product(Ws) - self.X).astype(F32)
        grads = []
        for i in range(L):
            g = (left[i].T @ Gx @ right[i].T).astype(F32)
            grads.append(g)
        return grads


def diagnose_depth(depths=(4, 8, 16), cond=100.0, method='adamw', epochs=300,
                   lr=1e-2, m=128, d=128, r=8, do_balance=False, eps=1e-6):
    """诊断 depth 是否携带信息: 逐因子 AdamW (+可选逐界面 balance),
    测各界面 vertical energy / balance error / 条件数随 depth 的变化."""
    out = {}
    for L in depths:
        chain = DeepLinearChain(m, d, r, L, cond=cond, seed=42, init='lora')
        # per-factor adam state
        mB = [np.zeros_like(W) for W in chain.Ws]
        vB = [np.zeros_like(W) for W in chain.Ws]
        b1, b2 = 0.9, 0.999
        vert_iface = []   # 每界面 vertical energy 轨迹 (mean over run)
        bal_iface = []
        losses = []
        n_iface = L - 1
        vert_acc = np.zeros(n_iface); bal_acc = np.zeros(n_iface); cnt = 0
        for t in range(1, epochs + 1):
            # optional per-interface balance (balance-only 诊断)
            if do_balance:
                for k in range(n_iface):
                    # 界面 k: (W_{k+1}, W_k) 之间 —— 视 W_k 为 A, W_{k+1} 为 B
                    Bk, Ak, Q = balance_to_section(chain.Ws[k + 1], chain.Ws[k])
                    chain.Ws[k + 1] = Bk; chain.Ws[k] = Ak
            grads = chain.grads()
            # measure per-interface vertical energy of the UPDATE direction
            # update = adam-preconditioned per factor
            us = []
            for i, g in enumerate(grads):
                mB[i] = b1 * mB[i] + (1 - b1) * g
                vB[i] = b2 * vB[i] + (1 - b2) * g ** 2
                mh = mB[i] / (1 - b1 ** t); vh = vB[i] / (1 - b2 ** t)
                u = (mh / (np.sqrt(vh) + 1e-8)).astype(F32)
                us.append(u)
            # per-interface rho_upd: 界面 k 用 (W_{k+1} as B, W_k as A), update (u_{k+1}, u_k)
            for k in range(n_iface):
                Bk, Ak = chain.Ws[k + 1], chain.Ws[k]
                uBk, uAk = us[k + 1], us[k]
                rho, _ = vertical_energy_ratio(Bk, Ak, uBk, uAk, eps)
                be = balance_error(Bk, Ak)
                if t > epochs // 2:   # 后半段稳定期均值
                    vert_acc[k] += rho; bal_acc[k] += be
            if t > epochs // 2:
                cnt += 1
            # update factors
            for i in range(len(chain.Ws)):
                chain.Ws[i] = (chain.Ws[i] - lr * us[i]).astype(F32)
            losses.append(chain.loss())
        vert_mean = (vert_acc / max(cnt, 1)).tolist()
        bal_mean = (bal_acc / max(cnt, 1)).tolist()
        # 全链等效因子条件数 (第一/最后界面代理)
        kfac, kB, kA = factor_condition_number(chain.Ws[-1], chain.Ws[0])
        out[L] = dict(
            depth=L, n_interface=n_iface,
            vert_per_interface=vert_mean,
            vert_iface_mean=float(np.mean(vert_mean)),
            vert_iface_max=float(np.max(vert_mean)),
            balance_per_interface=bal_mean,
            balance_iface_mean=float(np.mean(bal_mean)),
            final_loss=float(losses[-1]),
            final_rel=chain.rel_error(),
        )
    return out


if __name__ == '__main__':
    import json, sys
    method = sys.argv[1] if len(sys.argv) > 1 else 'adamw'
    do_bal = (method == 'balance')
    res = {}
    for cond in [100.0, 1000.0]:
        res[f'cond{int(cond)}'] = diagnose_depth(cond=cond, do_balance=do_bal)
    print(json.dumps(res, indent=2, default=float))

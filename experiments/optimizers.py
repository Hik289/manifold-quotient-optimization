#!/usr/bin/env python3
"""
optimizers.py — QuotientOpt + baselines (AdamW/SGD-M) + Balance-only + 14 消融 flag.

QuotientOpt 主体严格按 README §3.7/§3.8 伪代码:
  balance → transport(momentum) → 在平衡代表上算梯度 → grad horizontal lift
  → horizontal momentum → reproject momentum → additive retraction

⚠️ 接口: step(B, A, grad_fn) —— grad_fn(B,A)->(gB,gA).
   梯度在 optimizer 实际操作的 section 上计算 (QO=平衡代表), 保证梯度点/更新点/测量点自洽.
   info['section'] = (B,A) 是 u 所处 section, ρ_upd 必须在此测 (否则 section 不一致产生伪垂直能量).

消融通过 config flag 切换 (同一主体, 避免实现漂移). 见 IMPLEMENTATION_PLAN §2.

Author: ML Engineer (the project team)
"""
import numpy as np
from geometry import (F32, fnorm, balance_to_section, connection_form,
                      horizontal_lift, vertical_from_omega, transport_tangent)


class AdamW:
    """标准 AdamW (因子空间直接更新, 不 balance). 主对照 baseline."""
    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, wd
        self.mB = self.mA = self.vB = self.vA = None
        self.t = 0

    def step(self, B, A, grad_fn):
        if self.mB is None:
            self.mB = np.zeros_like(B); self.mA = np.zeros_like(A)
            self.vB = np.zeros_like(B); self.vA = np.zeros_like(A)
        self.t += 1
        if self.wd > 0:
            B = B * (1 - self.lr * self.wd); A = A * (1 - self.lr * self.wd)
        gB, gA = grad_fn(B, A)
        self.mB = self.b1*self.mB + (1-self.b1)*gB
        self.mA = self.b1*self.mA + (1-self.b1)*gA
        self.vB = self.b2*self.vB + (1-self.b2)*gB**2
        self.vA = self.b2*self.vA + (1-self.b2)*gA**2
        mhB = self.mB/(1-self.b1**self.t); mhA = self.mA/(1-self.b1**self.t)
        vhB = self.vB/(1-self.b2**self.t); vhA = self.vA/(1-self.b2**self.t)
        uB = (mhB/(np.sqrt(vhB)+self.eps)).astype(F32)
        uA = (mhA/(np.sqrt(vhA)+self.eps)).astype(F32)
        dB = -self.lr*uB; dA = -self.lr*uA
        Bn = (B+dB).astype(F32); An = (A+dA).astype(F32)
        return Bn, An, dict(uB=uB, uA=uA, gB=gB, gA=gA, dB=dB, dA=dA,
                            momentum=(self.mB, self.mA), section=(B, A))


class SGDM:
    """重球动量 SGD. 次对照 (不 balance)."""
    def __init__(self, lr=1e-2, momentum=0.9, wd=0.0):
        self.lr, self.mu, self.wd = lr, momentum, wd
        self.mB = self.mA = None

    def step(self, B, A, grad_fn):
        if self.mB is None:
            self.mB = np.zeros_like(B); self.mA = np.zeros_like(A)
        gB, gA = grad_fn(B, A)
        if self.wd > 0:
            gB = gB + self.wd*B; gA = gA + self.wd*A
        self.mB = self.mu*self.mB + gB
        self.mA = self.mu*self.mA + gA
        uB = self.mB.astype(F32); uA = self.mA.astype(F32)
        dB = -self.lr*uB; dA = -self.lr*uA
        Bn = (B+dB).astype(F32); An = (A+dA).astype(F32)
        return Bn, An, dict(uB=uB, uA=uA, gB=gB, gA=gA, dB=dB, dA=dA,
                            momentum=(self.mB, self.mA), section=(B, A))


def _scale_omega(Omega, mode):
    """gauge_mode: full=完整 Ω; diagonal=只留对角; scalar=只留 trace/r * I."""
    if mode == 'full':
        return Omega
    elif mode == 'diagonal':
        return np.diag(np.diag(Omega)).astype(F32)
    elif mode == 'scalar':
        r = Omega.shape[0]
        return (np.trace(Omega)/r*np.eye(r)).astype(F32)
    raise ValueError(mode)


class QuotientOpt:
    """QuotientOpt + 消融 flag. 底层 underlying optimizer 默认 AdamW-style 预条件.

    flags (默认 = Full QO):
      do_balance=True, balance_interval=1, do_horizontal_grad=True,
      use_momentum=True, transport_momentum=True, reproject_momentum=True,
      sylvester_eps=1e-6, gauge_mode='full', factor_weight_decay=0.0,
      underlying='adam' | 'sgd'
    Balance-only (A1) = do_horizontal_grad=False, reproject_momentum=False.
    """
    def __init__(self, lr=1e-2, beta=0.9, underlying='adam', beta2=0.999,
                 do_balance=True, balance_interval=1, do_horizontal_grad=True,
                 use_momentum=True, transport_momentum=True, reproject_momentum=True,
                 sylvester_eps=1e-6, gauge_mode='full', factor_weight_decay=0.0):
        self.lr, self.beta, self.beta2 = lr, beta, beta2
        self.underlying = underlying
        self.do_balance = do_balance
        self.balance_interval = balance_interval
        self.do_horizontal_grad = do_horizontal_grad
        self.use_momentum = use_momentum
        self.transport_momentum = transport_momentum
        self.reproject_momentum = reproject_momentum
        self.eps = sylvester_eps
        self.gauge_mode = gauge_mode
        self.fwd = factor_weight_decay
        self.mB = self.mA = None      # horizontal momentum
        self.vB = self.vA = None      # (adam) second moment
        self.t = 0

    def step(self, B, A, grad_fn):
        self.t += 1
        Q = None
        # 1. balance to canonical section
        if self.do_balance and (self.t - 1) % self.balance_interval == 0:
            B, A, Q = balance_to_section(B, A)
        if self.mB is None:
            self.mB = np.zeros_like(B); self.mA = np.zeros_like(A)
            self.vB = np.zeros_like(B); self.vA = np.zeros_like(A)
        # 2. gauge transport of momentum
        if Q is not None and self.transport_momentum and self.use_momentum:
            self.mB, self.mA = transport_tangent(self.mB, self.mA, Q)
        # 3. 在平衡代表上算梯度 (README §3.4)
        gB, gA = grad_fn(B, A)
        # factor weight decay (non-invariant, ablation A11)
        if self.fwd > 0:
            gB = gB + self.fwd*B; gA = gA + self.fwd*A
        # 3b. horizontal lift of gradient
        if self.do_horizontal_grad:
            hgB, hgA, Om_g, *_ = horizontal_lift(B, A, gB, gA, self.eps)
            if self.gauge_mode != 'full':
                Om_g = _scale_omega(Om_g, self.gauge_mode)
                hgB = (gB - B @ Om_g).astype(F32); hgA = (gA + Om_g @ A).astype(F32)
        else:
            hgB, hgA = gB, gA   # balance-only: 不投影梯度
        # underlying preconditioning (adam second moment) —— 产生 gauge 漂移的来源
        if self.underlying == 'adam':
            self.vB = self.beta2*self.vB + (1-self.beta2)*hgB**2
            self.vA = self.beta2*self.vA + (1-self.beta2)*hgA**2
            vhB = self.vB/(1-self.beta2**self.t); vhA = self.vA/(1-self.beta2**self.t)
            pgB = (hgB/(np.sqrt(vhB)+1e-8)).astype(F32)
            pgA = (hgA/(np.sqrt(vhA)+1e-8)).astype(F32)
        else:
            pgB, pgA = hgB, hgA
        # 4. horizontal momentum
        if self.use_momentum:
            self.mB = (self.beta*self.mB + (1-self.beta)*pgB).astype(F32)
            self.mA = (self.beta*self.mA + (1-self.beta)*pgA).astype(F32)
            uB_pre, uA_pre = self.mB, self.mA
        else:
            uB_pre, uA_pre = pgB, pgA
        # 5. reproject momentum to horizontal
        if self.do_horizontal_grad and self.reproject_momentum:
            hmB, hmA, Om_m, *_ = horizontal_lift(B, A, uB_pre, uA_pre, self.eps)
            if self.gauge_mode != 'full':
                Om_m = _scale_omega(Om_m, self.gauge_mode)
                hmB = (uB_pre - B @ Om_m).astype(F32); hmA = (uA_pre + Om_m @ A).astype(F32)
            if self.use_momentum:
                self.mB, self.mA = hmB, hmA
            uB, uA = hmB, hmA
        else:
            uB, uA = uB_pre, uA_pre
        # 6. additive retraction (下一步 balance 回 section)
        dB = -self.lr*uB; dA = -self.lr*uA
        Bn = (B+dB).astype(F32); An = (A+dA).astype(F32)
        # ⚠️ u 相对 balance 后 section (B,A) 水平, ρ_upd 必须在此 section 上测
        return Bn, An, dict(uB=np.asarray(uB, F32), uA=np.asarray(uA, F32),
                            gB=gB, gA=gA, dB=dB, dA=dA,
                            momentum=(self.mB, self.mA), gauge_Q=Q, section=(B, A))


def make_optimizer(method, lr=1e-2, **kw):
    """工厂: 'adamw','sgdm','balance_only','quotientopt' + 消融 flag."""
    if method == 'adamw':
        return AdamW(lr=lr, wd=kw.get('wd', 0.0))
    if method == 'sgdm':
        return SGDM(lr=lr, momentum=kw.get('momentum', 0.9), wd=kw.get('wd', 0.0))
    if method == 'balance_only':
        return QuotientOpt(lr=lr, do_horizontal_grad=False, reproject_momentum=False,
                           underlying=kw.get('underlying', 'adam'),
                           **{k: v for k, v in kw.items()
                              if k in ('beta', 'balance_interval', 'sylvester_eps')})
    if method == 'quotientopt':
        return QuotientOpt(lr=lr, **kw)
    raise ValueError(method)

#!/usr/bin/env python3
"""
tasks.py — 任务封装 (P0: 解析梯度).

matrix_factorization:  f(BA) = 0.5 ||BA - X*||_F^2,  grad_X = BA - X*
  -> grad_B = grad_X A^T, grad_A = B^T grad_X   (LoRA chain rule, 天然水平)
deep_linear:  end-to-end 低秩线性回归代理, 同 f(BA)=0.5||BA-X*||^2 但 depth 影响
  失衡累积 (通过初始化 scale). P0 用 product-form 代理.

Author: ML Engineer (the project team)
"""
import numpy as np
from geometry import F32
from data import make_lowrank_target, init_factors, make_rng


class MatrixFactorization:
    """f(BA) = 0.5||BA - X*||^2. 强凸 in X (matrix sensing full-observation 版)."""
    def __init__(self, m, d, r, cond=1.0, seed=42):
        self.rng = make_rng(seed)
        self.X, self.U, self.s, self.V = make_lowrank_target(m, d, r, cond, self.rng)
        self.m, self.d, self.r, self.cond = m, d, r, cond
        self.opt_loss = 0.0   # 解析最优 (rank-r 可精确拟合)

    def loss(self, B, A):
        return 0.5 * float(np.linalg.norm(B @ A - self.X, 'fro') ** 2)

    def grad(self, B, A):
        Gx = (B @ A - self.X).astype(F32)
        return (Gx @ A.T).astype(F32), (B.T @ Gx).astype(F32)

    def rel_error(self, B, A):
        return float(np.linalg.norm(B @ A - self.X, 'fro') / (np.linalg.norm(self.X, 'fro') + 1e-12))

    def make_init(self, mode='lora'):
        return init_factors(self.m, self.d, self.r, self.U, self.s, self.V, mode=mode, rng=self.rng)


class DeepLinear:
    """Deep linear 低秩回归. product-form 代理 f(BA)=0.5||BA-X*||^2,
    depth 通过初始化 scale 放大失衡 (Saxe-style). P0 阶段."""
    def __init__(self, m, d, r, depth=8, cond=1.0, seed=42):
        self.rng = make_rng(seed)
        self.X, self.U, self.s, self.V = make_lowrank_target(m, d, r, cond, self.rng)
        self.m, self.d, self.r, self.depth, self.cond = m, d, r, depth, cond
        self.opt_loss = 0.0

    def loss(self, B, A):
        return 0.5 * float(np.linalg.norm(B @ A - self.X, 'fro') ** 2)

    def grad(self, B, A):
        Gx = (B @ A - self.X).astype(F32)
        return (Gx @ A.T).astype(F32), (B.T @ Gx).astype(F32)

    def rel_error(self, B, A):
        return float(np.linalg.norm(B @ A - self.X, 'fro') / (np.linalg.norm(self.X, 'fro') + 1e-12))

    def make_init(self, mode='lora'):
        return init_factors(self.m, self.d, self.r, self.U, self.s, self.V, mode=mode, rng=self.rng)


def make_task(name, **kw):
    if name == 'matrix_factorization' or name == 'matrix_sensing':
        return MatrixFactorization(**{k: v for k, v in kw.items()
                                      if k in ('m', 'd', 'r', 'cond', 'seed')})
    if name == 'deep_linear':
        return DeepLinear(**{k: v for k, v in kw.items()
                            if k in ('m', 'd', 'r', 'depth', 'cond', 'seed')})
    raise ValueError(name)

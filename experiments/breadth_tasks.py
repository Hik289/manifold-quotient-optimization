#!/usr/bin/env python3
"""
breadth_tasks.py — INR + SciML 任务 (torch), 低秩因子插入 + numpy 几何 optimizer 桥接.

设计: 网络前向用 torch; 关键 hidden 层参数化为 W = W0 + (alpha) B@A (低秩 LoRA 更新);
  B,A 是被优化的 LoRA 因子。torch autograd 算 B/A 梯度 → 转 numpy → 现有 optimizers.py
  (AdamW/balance_only/quotientopt) 更新 → 写回 torch。这样 gauge 几何与主线一致。

任务:
  INRImageFit: SIREN-style MLP 拟合图像, 一个 hidden 层用低秩因子
  PINNBurgers: PINN 解 1D Burgers, 一个 hidden 层用低秩因子

Author: ML Engineer (the project team)
"""
import numpy as np
import torch
import torch.nn as nn

F32 = np.float32
torch.manual_seed(42)


def set_seed(s=42):
    torch.manual_seed(s); np.random.seed(s)


class LowRankLinear(nn.Module):
    """W = W0(frozen) + scale * B @ A. B,A 是 LoRA 因子 (被 numpy optimizer 更新)."""
    def __init__(self, in_dim, out_dim, rank, cond_init=100.0, scale=1.0, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W0 = nn.Parameter(torch.randn(out_dim, in_dim, generator=g) / np.sqrt(in_dim),
                               requires_grad=False)
        self.scale = scale
        self.rank = rank
        # LoRA 非对称 init: B≈0, A~N (放大失衡, 主线一致)
        self.B = nn.Parameter(0.01 * torch.randn(out_dim, rank, generator=g))
        self.A = nn.Parameter(torch.randn(rank, in_dim, generator=g) / np.sqrt(in_dim))

    def forward(self, x):
        W = self.W0 + self.scale * (self.B @ self.A)
        return x @ W.t()

    def get_factors(self):
        return (self.B.detach().cpu().numpy().astype(F32),
                self.A.detach().cpu().numpy().astype(F32))

    def set_factors(self, Bnp, Anp):
        with torch.no_grad():
            self.B.copy_(torch.from_numpy(Bnp.astype(np.float32)))
            self.A.copy_(torch.from_numpy(Anp.astype(np.float32)))

    def grad_factors(self):
        return (self.B.grad.detach().cpu().numpy().astype(F32),
                self.A.grad.detach().cpu().numpy().astype(F32))


class SIREN(nn.Module):
    """SIREN-style INR, 一个中间层低秩. 拟合 2D 坐标→亮度."""
    def __init__(self, hidden=128, rank=16, w0=30.0, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w0 = w0
        self.l1 = nn.Linear(2, hidden)
        with torch.no_grad():
            self.l1.weight.uniform_(-1/2, 1/2, generator=g)
        # 中间层低秩 (被 gauge 优化的目标)
        self.lr_layer = LowRankLinear(hidden, hidden, rank, seed=seed)
        self.l3 = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x):
        x = torch.sin(self.w0 * self.l1(x))
        x = torch.sin(self.w0 * self.lr_layer(x))
        return self.l3(x)


class PINNBurgers(nn.Module):
    """PINN 解 1D Burgers u_t + u u_x = nu u_xx. 一个 hidden 层低秩."""
    def __init__(self, hidden=64, rank=16, seed=42):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.lr_layer = LowRankLinear(hidden, hidden, rank, seed=seed)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, xt):
        h = self.act(self.l1(xt))
        h = self.act(self.lr_layer(h))
        return self.l3(h)


def make_image_target(n=64, seed=42):
    """合成高频图像 (强迫 INR 用高频, 高 cond 信号)."""
    g = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    img = (np.sin(8*np.pi*X) * np.cos(6*np.pi*Y) + 0.5*np.sin(14*np.pi*(X+Y)))
    coords = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)
    vals = img.ravel().astype(np.float32)[:, None]
    return coords, vals, img


class GCNLowRank(nn.Module):
    """2-layer GCN, 中间层低秩. 节点分类 (self-contained SBM graph)."""
    def __init__(self, in_dim, hidden, n_class, rank=16, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = nn.Linear(in_dim, hidden)
        self.lr_layer = LowRankLinear(hidden, hidden, rank, seed=seed)
        self.l3 = nn.Linear(hidden, n_class)
        self.act = nn.ReLU()

    def forward(self, x, Ahat):
        h = self.act(Ahat @ self.l1(x))
        h = self.act(Ahat @ self.lr_layer(h))
        return Ahat @ self.l3(h)


def make_sbm_graph(n_per=100, n_blocks=4, p_in=0.15, p_out=0.01, feat_dim=32, seed=42):
    """合成 stochastic block model 图 + 节点特征 + 标签. 返回 (X, Ahat, y, masks)."""
    g = np.random.default_rng(seed)
    N = n_per * n_blocks
    y = np.repeat(np.arange(n_blocks), n_per)
    # adjacency
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            p = p_in if y[i] == y[j] else p_out
            if g.random() < p:
                A[i, j] = A[j, i] = 1.0
    A += np.eye(N, dtype=np.float32)
    deg = A.sum(1)
    Dinv = np.diag(1.0/np.sqrt(deg + 1e-8)).astype(np.float32)
    Ahat = (Dinv @ A @ Dinv).astype(np.float32)   # normalized adjacency
    # features: block-correlated + noise
    centers = g.standard_normal((n_blocks, feat_dim)).astype(np.float32)
    X = (centers[y] + 0.8 * g.standard_normal((N, feat_dim))).astype(np.float32)
    # train/test masks
    idx = g.permutation(N)
    train_mask = np.zeros(N, bool); train_mask[idx[:N//2]] = True
    test_mask = ~train_mask
    return X, Ahat, y.astype(np.int64), train_mask, test_mask


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse <= 0:
        return 99.0
    rng = target.max() - target.min()
    return float(20 * np.log10(rng / (np.sqrt(mse) + 1e-12)))

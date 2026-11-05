#!/usr/bin/env python3
"""
run_breadth_scan.py — 任务池 breadth scan (路 A 主 claim 更广成立).

判据见 PREREG_breadth_scan.md (跑前写死). 主 claim = balance/QO vs AdamW 的
gauge 不变性 + κ_fac 锁 √κ + gauge robustness。不 claim horizontal projection 独立价值。

任务: INR (SIREN image fit) + SciML (PINN Burgers). 低秩层用 numpy optimizer 优化。
测: κ_fac/√κ 轨迹 + gauge stress product disagreement 三方 + final task metric。
全任务如实报 win/tie/weak → breadth_scan_summary.json.

Author: ML Engineer (the project team)
"""
import sys, os, json, time
# 限制线程数 (hpc 被其他项目 job 争用, 避免 oversubscription thrash)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
torch.set_num_threads(2)
from optimizers import make_optimizer
from geometry import F32, fnorm, factor_condition_number, balance_to_section
from breadth_tasks import (SIREN, PINNBurgers, GCNLowRank, make_image_target,
                           make_sbm_graph, psnr, set_seed)

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, 'logs')
os.makedirs(LOGDIR, exist_ok=True)
EPS = 1e-6


def sqrt_kappa_of_factors(B, A, rank):
    """√κ(BA) via top-r 奇异值."""
    X = (B @ A).astype(np.float64)
    sv = np.linalg.svd(X, compute_uv=False)[:rank]
    if sv[-1] <= 0:
        return None
    return float(np.sqrt(sv[0] / sv[-1]))


# ---- INR: SIREN image fitting ----
def train_inr(method, rank=16, epochs=500, lr=1e-3, gauge_Q=None, seed=42):
    set_seed(seed)
    dev = 'cpu'
    coords, vals, img = make_image_target(n=48, seed=seed)
    xc = torch.from_numpy(coords); yv = torch.from_numpy(vals)
    net = SIREN(hidden=96, rank=rank, seed=seed).to(dev)
    # 注入 gauge (gauge stress): B←BQ, A←Q^{-1}A
    if gauge_Q is not None:
        B, A = net.lr_layer.get_factors()
        Qi = np.linalg.inv(gauge_Q.astype(np.float64)).astype(F32)
        net.lr_layer.set_factors((B @ gauge_Q).astype(F32), (Qi @ A).astype(F32))
    # torch optimizer for non-LoRA params; numpy optimizer for LoRA factors
    other = [p for n, p in net.named_parameters() if 'lr_layer' not in n and p.requires_grad]
    topt = torch.optim.Adam(other, lr=lr)
    fopt = make_optimizer(method, lr=lr)
    kfacs = []; prods = []; losses = []
    for t in range(epochs):
        topt.zero_grad()
        if net.lr_layer.B.grad is not None:
            net.lr_layer.B.grad = None; net.lr_layer.A.grad = None
        pred = net(xc)
        loss = ((pred - yv) ** 2).mean()
        loss.backward()
        topt.step()
        # numpy optimizer step on LoRA factors
        B, A = net.lr_layer.get_factors()
        def grad_fn(Bc, Ac):
            return net.lr_layer.grad_factors()
        Bn, An, info = fopt.step(B, A, grad_fn)
        net.lr_layer.set_factors(Bn, An)
        secB, secA = info['section']
        kf, _, _ = factor_condition_number(secB, secA)
        kfacs.append(kf)
        prods.append((secB @ secA).astype(F32).copy())
        losses.append(float(loss.item()))
    # final metric
    with torch.no_grad():
        pred = net(xc).cpu().numpy()
    final_psnr = psnr(pred, vals)
    sk = sqrt_kappa_of_factors(*net.lr_layer.get_factors(), rank)
    return dict(kfac_final=float(kfacs[-1]), sqrt_kappa=sk,
                kfac_over_sqrtk=float(kfacs[-1]/sk) if sk else None,
                final_psnr=final_psnr, final_loss=losses[-1], prods=prods,
                kfac_traj=kfacs)


# ---- SciML: PINN Burgers ----
def train_pinn(method, rank=16, epochs=500, lr=1e-3, gauge_Q=None, seed=42, nu=0.01):
    set_seed(seed)
    net = PINNBurgers(hidden=64, rank=rank, seed=seed)
    if gauge_Q is not None:
        B, A = net.lr_layer.get_factors()
        Qi = np.linalg.inv(gauge_Q.astype(np.float64)).astype(F32)
        net.lr_layer.set_factors((B @ gauge_Q).astype(F32), (Qi @ A).astype(F32))
    other = [p for n, p in net.named_parameters() if 'lr_layer' not in n and p.requires_grad]
    topt = torch.optim.Adam(other, lr=lr)
    fopt = make_optimizer(method, lr=lr)
    # collocation points
    g = np.random.default_rng(seed)
    N = 1000
    x = torch.tensor(g.uniform(-1, 1, (N, 1)), dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(g.uniform(0, 1, (N, 1)), dtype=torch.float32, requires_grad=True)
    # IC: u(x,0) = -sin(pi x)
    xi = torch.tensor(np.linspace(-1, 1, 100)[:, None], dtype=torch.float32)
    ti = torch.zeros_like(xi)
    ui = -torch.sin(np.pi * xi)
    kfacs = []; prods = []; losses = []
    for it in range(epochs):
        topt.zero_grad()
        if net.lr_layer.B.grad is not None:
            net.lr_layer.B.grad = None; net.lr_layer.A.grad = None
        xt = torch.cat([x, tt], 1)
        u = net(xt)
        du = torch.autograd.grad(u.sum(), [x, tt], create_graph=True)
        u_x, u_t = du[0], du[1]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        f = u_t + u * u_x - nu * u_xx
        loss_pde = (f ** 2).mean()
        ic = net(torch.cat([xi, ti], 1))
        loss_ic = ((ic - ui) ** 2).mean()
        loss = loss_pde + loss_ic
        loss.backward()
        topt.step()
        B, A = net.lr_layer.get_factors()
        def grad_fn(Bc, Ac):
            return net.lr_layer.grad_factors()
        Bn, An, info = fopt.step(B, A, grad_fn)
        net.lr_layer.set_factors(Bn, An)
        secB, secA = info['section']
        kf, _, _ = factor_condition_number(secB, secA)
        kfacs.append(kf); prods.append((secB @ secA).astype(F32).copy())
        losses.append(float(loss.item()))
    sk = sqrt_kappa_of_factors(*net.lr_layer.get_factors(), rank)
    return dict(kfac_final=float(kfacs[-1]), sqrt_kappa=sk,
                kfac_over_sqrtk=float(kfacs[-1]/sk) if sk else None,
                final_loss=losses[-1], prods=prods, kfac_traj=kfacs)


# ---- 图学习: GCN node classification on SBM ----
def train_gcn(method, rank=16, epochs=500, lr=1e-3, gauge_Q=None, seed=42):
    set_seed(seed)
    X, Ahat, y, tr, te = make_sbm_graph(seed=seed)
    Xt = torch.from_numpy(X); At = torch.from_numpy(Ahat); yt = torch.from_numpy(y)
    trm = torch.from_numpy(tr); tem = torch.from_numpy(te)
    net = GCNLowRank(X.shape[1], 64, int(y.max()+1), rank=rank, seed=seed)
    if gauge_Q is not None:
        B, A = net.lr_layer.get_factors()
        Qi = np.linalg.inv(gauge_Q.astype(np.float64)).astype(F32)
        net.lr_layer.set_factors((B @ gauge_Q).astype(F32), (Qi @ A).astype(F32))
    other = [p for n, p in net.named_parameters() if 'lr_layer' not in n and p.requires_grad]
    topt = torch.optim.Adam(other, lr=lr)
    fopt = make_optimizer(method, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    kfacs = []; prods = []; losses = []
    for t in range(epochs):
        topt.zero_grad()
        if net.lr_layer.B.grad is not None:
            net.lr_layer.B.grad = None; net.lr_layer.A.grad = None
        out = net(Xt, At)
        loss = lossf(out[trm], yt[trm])
        loss.backward()
        topt.step()
        B, A = net.lr_layer.get_factors()
        Bn, An, info = fopt.step(B, A, lambda Bc, Ac: net.lr_layer.grad_factors())
        net.lr_layer.set_factors(Bn, An)
        secB, secA = info['section']
        kf, _, _ = factor_condition_number(secB, secA)
        kfacs.append(kf); prods.append((secB @ secA).astype(F32).copy())
        losses.append(float(loss.item()))
    with torch.no_grad():
        pred = net(Xt, At).argmax(1).cpu().numpy()
    test_acc = float((pred[te] == y[te]).mean())
    sk = sqrt_kappa_of_factors(*net.lr_layer.get_factors(), rank)
    return dict(kfac_final=float(kfacs[-1]), sqrt_kappa=sk,
                kfac_over_sqrtk=float(kfacs[-1]/sk) if sk else None,
                final_loss=losses[-1], final_acc=test_acc, prods=prods, kfac_traj=kfacs)


def make_gauge(r, kind, val, rng):
    if kind == 'scalar':
        return (val * np.eye(r)).astype(F32)
    U = np.linalg.qr(rng.standard_normal((r, r)))[0]
    V = np.linalg.qr(rng.standard_normal((r, r)))[0]
    s = np.logspace(0, np.log10(val), r) if val > 1 else np.ones(r)
    return (U @ np.diag(s) @ V.T).astype(F32)


def gauge_stress(train_fn, method, rank, epochs=300):
    """product trajectory disagreement 相对 gauge=I."""
    ref = train_fn(method, rank=rank, epochs=epochs, gauge_Q=None)['prods']
    T = len(ref)
    gauges = [('scalar', 1e-2), ('scalar', 1e2), ('dense', 1e2), ('dense', 1e3)]
    disags = []
    for (kind, val) in gauges:
        Q = make_gauge(rank, kind, val, np.random.default_rng(456))
        gp = train_fn(method, rank=rank, epochs=epochs, gauge_Q=Q)['prods']
        Tg = min(len(gp), T)
        if Tg == 0:
            disags.append(float('inf')); continue
        disags.append(float(fnorm(gp[Tg-1] - ref[Tg-1])))
    finals = [d for d in disags if np.isfinite(d)]
    return float(np.max(finals)) if finals else float('inf')


def evaluate_win(res):
    """按 PREREG 判据评 win/tie/weak. res = {method: {kfac_over_sqrtk, gauge_disag, metric}}."""
    a = res['adamw']; b = res['balance_only']; q = res['quotientopt']
    # C1: κ_fac 锁 √κ
    c1_adam_drift = (a['kfac_over_sqrtk'] or 0) > 3
    c1_bal_lock = 0.5 <= (b['kfac_over_sqrtk'] or 99) <= 2.0
    c1 = c1_adam_drift and c1_bal_lock
    # C2: gauge robustness
    c2 = False
    if b['gauge_disag'] > 0:
        c2 = (a['gauge_disag'] / b['gauge_disag']) >= 10
    # verdict
    if c1 and c2:
        v = 'WIN'
    elif c1 or c2:
        v = 'TIE'
    else:
        v = 'WEAK'
    return dict(verdict=v, c1_kfac_lock=c1, c2_gauge_robust=c2,
                c1_adam_drift=c1_adam_drift, c1_bal_lock=c1_bal_lock,
                adam_kfac_ratio=a['kfac_over_sqrtk'], bal_kfac_ratio=b['kfac_over_sqrtk'],
                qo_kfac_ratio=q['kfac_over_sqrtk'],
                adam_disag=a['gauge_disag'], bal_disag=b['gauge_disag'], qo_disag=q['gauge_disag'])


def main():
    t0 = time.time()
    # (train_fn, train_ep, gauge_ep) —— idea.md 规定 INR/SciML ≥1000-2000ep;
    #   长训练是这些任务的自然 regime, 也是 C1(AdamW κ 漂移)充分显现所需 (与主线 2000ep 一致).
    # 延长到 2000ep (与主线 matrix sensing 同长, 给 C1/AdamW κ 漂移公平的长训练测试).
    # 加图学习 GCN (idea.md 任务族, self-contained SBM) 找第 2 个 C2 干净任务.
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    all_tasks = {'INR_SIREN': (train_inr, 2000, 800),
                 'PINN_Burgers': (train_pinn, 2000, 800),
                 'GCN_SBM': (train_gcn, 2000, 800)}
    if which == 'gcn':
        tasks = {'GCN_SBM': all_tasks['GCN_SBM']}
    else:
        tasks = all_tasks
    all_out = {}
    for tname, (train_fn, train_ep, gauge_ep) in tasks.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] === {tname} ===", flush=True)
        res = {}
        for method in ['adamw', 'balance_only', 'quotientopt']:
            r = train_fn(method, rank=16, epochs=train_ep)
            disag = gauge_stress(train_fn, method, rank=16, epochs=gauge_ep)
            # metric: PSNR↑ (INR) / accuracy↑ (GCN) / -loss↑ (PINN)
            if r.get('final_psnr') is not None:
                metric = r['final_psnr']
            elif r.get('final_acc') is not None:
                metric = r['final_acc']
            else:
                metric = -r['final_loss']
            # κ_fac 漂移轨迹 (证明长训练才显现, 与主线 P2 一致): 存 checkpoint 比值
            sk = r['sqrt_kappa'] or 1.0
            kt = r.get('kfac_traj', [])
            ckpts = {}
            for ep in [100, 500, 1000, 1500, 2000]:
                if ep <= len(kt):
                    ckpts[str(ep)] = float(kt[ep-1] / sk)
            res[method] = dict(kfac_over_sqrtk=r['kfac_over_sqrtk'],
                               kfac_final=r['kfac_final'], sqrt_kappa=r['sqrt_kappa'],
                               gauge_disag=disag, final_loss=r['final_loss'],
                               metric=metric, final_psnr=r.get('final_psnr'),
                               final_acc=r.get('final_acc'),
                               kfac_ratio_ckpts=ckpts)
            with open(os.path.join(LOGDIR, f"breadth_{tname}_{method}_kfac.jsonl"), 'w') as f:
                for i, kf in enumerate(kt):
                    f.write(json.dumps({"epoch": i, "kfac": kf, "kfac_over_sqrtk": kf/sk}) + '\n')
            print(f"  {method:13}: kfac/√κ={r['kfac_over_sqrtk']:.2f} ckpts={ {k: round(v,2) for k,v in ckpts.items()} } "
                  f"disag={disag:.2e} loss={r['final_loss']:.3e} psnr={r.get('final_psnr')}", flush=True)
        ev = evaluate_win(res)
        all_out[tname] = dict(results=res, evaluation=ev)
        print(f"  >>> {tname} verdict: {ev['verdict']} "
              f"(C1 κ-lock={ev['c1_kfac_lock']} C2 gauge-robust={ev['c2_gauge_robust']})", flush=True)
        print(f"      adam κ/√κ={ev['adam_kfac_ratio']} bal={ev['bal_kfac_ratio']} | "
              f"adam disag={ev['adam_disag']:.2e} bal disag={ev['bal_disag']:.2e}", flush=True)

    # merge with existing (不覆盖其他任务)
    sfile = os.path.join(HERE, 'breadth_scan_summary.json')
    merged = {}
    if os.path.exists(sfile):
        try:
            merged = json.load(open(sfile)).get('tasks', {})
        except Exception:
            merged = {}
    merged.update(all_out)
    out = dict(elapsed_sec=time.time()-t0, prereg='PREREG_breadth_scan.md', tasks=merged)
    with open(sfile, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: None if isinstance(o, np.ndarray) else float(o))
    print(f"\n=== BREADTH SCAN 汇总 ===")
    for tname, r in all_out.items():
        print(f"  {tname}: {r['evaluation']['verdict']}")
    print(f"BREADTH_SCAN_DONE elapsed={out['elapsed_sec']:.1f}s")
    return out


if __name__ == '__main__':
    main()

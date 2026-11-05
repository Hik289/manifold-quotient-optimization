<div align="center">
  <h1>QuotientOpt: Gauge-Invariant Low-Rank Optimization</h1>
  <p><strong>Balanced Quotient Geometry, Convergence, and Generalization</strong></p>
  <p>
    <a href="paper/main.pdf"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg"></a>
    <a href="#quickstart"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg"></a>
    <a href="experiments/full_proposal/geometry_torch.py"><img alt="PyTorch implementation" src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg"></a>
    <a href="#experiments"><img alt="Reproducible experiments" src="https://img.shields.io/badge/Experiments-Protocols-0B877A.svg"></a>
  </p>
  <p>Balanced quotient geometry · coordinate-consistent momentum · horizontal updates</p>
</div>

<p align="center">
  <a href="assets/quotientopt_method_overview.png"><img src="assets/quotientopt_method_overview.png" width="98%" alt="QuotientOpt factorization, balance, gradient, horizontal memory, and factor update"></a>
</p>

## Overview

A low-rank matrix has many equivalent factor pairs: $X=BA=(BQ)(Q^{-1}A)$. Optimizer state can therefore change with the chosen coordinates even when $X$ does not. **QuotientOpt** balances the factors without changing their product, transports persistent momentum to the new coordinates, and removes gauge motion with a small Sylvester solve. The [paper](paper/main.pdf) develops the geometry, convergence and generalization analysis; this repository provides the implementation and experiment protocols.

## Quickstart

Use Python 3.10 or newer. The synthetic CPU smoke run needs no downloaded benchmark data.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/full_proposal/run.py examples/matrix_sensing_smoke.json \
  --device cpu --smoke-epochs 2 --output outputs
```

The run writes its configuration, checkpoints, and trajectory to `outputs/<run-id>/`. The two-epoch smoke run checks the code path; it is **not** a paper result. For a full run, set `epochs` in the configuration and omit `--smoke-epochs`.

Benchmark runs use the additional dependencies in [`requirements-benchmarks.txt`](requirements-benchmarks.txt) and the loaders in [`datasets_full.py`](experiments/full_proposal/datasets_full.py). Set `QUOTIENT_DATA` for the data cache and, when needed, `QUOTIENT_VISION_DATA` for an existing CIFAR cache.

```bash
export QUOTIENT_DATA="$PWD/datasets"
python experiments/full_proposal/run.py path/to/config.json --device cuda:0 --output outputs
```

## Method

For balanced factors $(\bar B,\bar A)$ with shared Gram matrix $H=\bar B^\top\bar B=\bar A\bar A^\top$, a provisional factor update $U=(U_B,U_A)$ is projected by solving

$$
(H+\varepsilon I)\Omega+\Omega(H+\varepsilon I)=\bar B^\top U_B-U_A\bar A^\top,
\qquad
\Pi_H(U)=(U_B-\bar B\Omega,\;U_A+\Omega\bar A).
$$

With $\varepsilon=0$, the update is exactly horizontal; the regularized solve has residual $2\varepsilon\Omega$. The PyTorch implementation is [`geometry_torch.py`](experiments/full_proposal/geometry_torch.py), with [numerical checks](experiments/tests/test_numerics.py).

## Experiments

The code covers matrix sensing and completion, deep linear networks, graph learning, implicit neural representations, scientific ML, vision, and language modeling. The [proposal grid](experiments/full_proposal/grid.py) specifies configurations; the manuscript reports the subset completed and analyzed. Its [experiment section](paper/sections/5_experiments.tex) and [full protocol](paper/sections/full_protocol.tex) give the comparison settings, selection rules, and outcomes, including mixed results.

The reported Cora/GCN, SDF/SIREN, and Poisson/PINN studies include matched-budget optimizer comparisons and balance-only controls. Formal configurations are stored in campaign `plan.json` files beside their runners. Checkpoints are selected on validation metrics before held-out test evaluation.

## Repository

| Path | Contents |
|:--|:--|
| [`experiments/`](experiments/) | Task runners, optimizer implementations, campaign plans, grid generator, and numerical tests. |
| [`paper/`](paper/) | [Manuscript source](paper/main.tex), bibliography, figures, and [PDF](paper/main.pdf). |
| [`figure_code/`](figure_code/) | [Loss-curve builder](figure_code/plot_core_figures.py) and bundled compact curve data. |
| [`figures/`](figures/) | Diagram specification and plotting scripts. |
| [`examples/`](examples/) | CPU smoke-run configuration. |


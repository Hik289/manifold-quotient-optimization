<div align="center">
  <h1>QuotientOpt: Gauge-Invariant Low-Rank Optimization</h1>
  <p><strong>A low-rank optimizer designed to give consistent updates across equivalent factor pairs.</strong></p>
  <p>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-2ea44f.svg"></a>
    <a href="#quickstart"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg"></a>
  </p>
  <p>Implementation, experiment code, and paper source.</p>
</div>

<p align="center">
  <a href="assets/quotientopt_method_overview.png"><img src="assets/quotientopt_method_overview.png" width="98%" alt="QuotientOpt workflow: factorize, balance, compute gradients, update momentum, and rebuild the matrix"></a>
</p>

## Overview

In a low-rank model, $X=BA$. For any invertible $Q$, replacing $B$ with $BQ$ and $A$ with $Q^{-1}A$ leaves $X$ unchanged. An optimizer can still update these two factor pairs differently. **QuotientOpt** balances the factors, moves momentum into the balanced coordinates, and reduces updates that change the factors without changing $X$. This consistency across equivalent pairs is the “gauge invariance” in the title. The repository includes the [paper source](paper/main.tex), implementation, and experiment code.

## Quickstart

Use Python 3.10 or newer. This two-epoch CPU run uses generated data, so it needs no benchmark download:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/full_proposal/run.py examples/matrix_sensing_smoke.json \
  --device cpu --smoke-epochs 2 --output outputs
```

The command saves its configuration, checkpoints, and training record under `outputs/<run-id>/`. It checks that the code runs; it is **not** a paper result. For a full run, set `epochs` in the configuration file and omit `--smoke-epochs`.

For benchmark runs, install [`requirements-benchmarks.txt`](requirements-benchmarks.txt) and use the data loaders in [`datasets_full.py`](experiments/full_proposal/datasets_full.py). Set `QUOTIENT_DATA` to the directory where you keep datasets. If you already have CIFAR data elsewhere, set `QUOTIENT_VISION_DATA` to that directory.

```bash
python -m pip install -r requirements-benchmarks.txt
export QUOTIENT_DATA="$PWD/datasets"
python experiments/full_proposal/run.py path/to/config.json --device cuda:0 --output outputs
```

## Method

After balancing, the factors $(\bar B,\bar A)$ share a small matrix $H=\bar B^\top\bar B=\bar A\bar A^\top$. Given a proposed update $U=(U_B,U_A)$, QuotientOpt solves

$$
(H+\varepsilon I)\Omega+\Omega(H+\varepsilon I)=\bar B^\top U_B-U_A\bar A^\top,
\qquad
\Pi_H(U)=(U_B-\bar B\Omega,\;U_A+\Omega\bar A).
$$

The resulting update removes changes to the factors that leave $X$ unchanged. This removal is exact when $\varepsilon=0$. With regularization, $\bar B^\top M_B-M_A\bar A^\top=2\varepsilon\Omega$. See the [PyTorch implementation](experiments/full_proposal/geometry_torch.py) and [numerical checks](experiments/tests/test_numerics.py).

## Experiments

The code covers matrix sensing and completion, deep linear networks, graph learning, implicit neural representations, scientific ML, vision, and language modeling. The [grid file](experiments/full_proposal/grid.py) lists planned settings; it does **not** mean every setting was run. The [experiment section](paper/sections/5_experiments.tex) and [full protocol](paper/sections/full_protocol.tex) identify the completed runs and report both favorable and mixed results.

The Cora/GCN, SDF/SIREN, and Poisson/PINN studies compare optimizers with the same training budget. They also include a version that balances the factors without the final projection. Run settings are stored in each campaign's `plan.json`. Checkpoints are chosen using validation data before test results are read.

## Repository

| Path | Contents |
|:--|:--|
| [`experiments/`](experiments/) | Training and evaluation code, run settings, and numerical tests. |
| [`paper/`](paper/) | [LaTeX paper source](paper/main.tex), references, and PNG figures. |
| [`figure_code/`](figure_code/) | [Loss-plot script](figure_code/plot_core_figures.py) and the data needed to redraw the main plots. |
| [`figures/`](figures/) | Figure scripts and notes on the method diagram. |
| [`examples/`](examples/) | Configuration for the small CPU run above. |


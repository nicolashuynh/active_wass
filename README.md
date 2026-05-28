# Active Timepoint Selection for Learning Measure-Valued Trajectories

Code for reproducing the experiments in **Active Timepoint Selection for
Learning Measure-Valued Trajectories** (ICML 2026).

The repository implements active sampling for measure-valued trajectories using
Wasserstein geometry, tangent-space Gaussian process surrogates, and
uncertainty-based acquisition. The reproduction scripts cover synthetic branching,
single-cell reprogramming, and CPS labor market experiments from the paper.
 
## Setup

This project uses `uv` and Python 3.13.

```bash
uv python install 3.13
uv sync
```

Check that the package imports correctly:

```bash
uv run python -c "import active_wasserstein; print('ok')"
```

## Data

| Experiment | Data requirement |
| --- | --- |
| Synthetic branching | No external data required. |
| Schiebinger single-cell | Loaded through CellRank's Schiebinger reprogramming dataset helper. |
| CPS labor market | Requires a preprocessed `.npz` trajectory built from IPUMS CPS data. |

For the single-cell experiments, the trajectory wrapper calls
`cellrank.datasets.reprogramming_schiebinger(...)` with the serum subset. The
provided config sets `allow_download=true` and `use_cellrank_loader=true`, so
CellRank can download/cache the dataset if it is not already present. To use an
existing local copy, set:

```bash
export CR_SERUM_PATH=/path/to/ExprMatrix_cr.h5ad
```

For the CPS experiments, the raw data can be downloaded from
[IPUMS CPS](https://cps.ipums.org/cps-action/samples), which requires an IPUMS
CPS account. The experiment code expects a preprocessed `.npz` trajectory.

## Experiment Scripts

These are the inner scripts that run the paper experiments.

### Synthetic

| Script | What it runs |
| --- | --- |
| `scripts/synthetic/active_uniform_random.sh` | Active vs. uniform vs. random acquisition on oscillatory sequential branching. |
| `scripts/synthetic/ablations.sh` | No-warp, fixed-reference, lower-rank, and RBF-kernel ablations. |
| `scripts/synthetic/interval_sweep.sh` | Interval-width sweep for the two branching events. |

### Single cell

| Script | What it runs |
| --- | --- |
| `scripts/single_cell/active_uniform_random.sh` | Active vs. uniform vs. random acquisition on Schiebinger serum data. |
| `scripts/single_cell/ablations.sh` | No-warp, fixed-reference, lower-rank, and RBF-kernel ablations. |

### Labor market

| Script | What it runs |
| --- | --- |
| `scripts/labor_market/active_uniform_random.sh` | Active vs. uniform vs. random acquisition on preprocessed CPS monthly snapshots. |

## Configuration

Hydra configuration files live in `conf/`.

| Directory | Contents |
| --- | --- |
| `conf/trajectory/` | Synthetic branching, Schiebinger single-cell, and CPS monthly trajectories. |
| `conf/strategy/` | Active, uniform, random, no-warp, RBF, and Matern strategy variants. |
| `conf/surrogate/` | Linearized Wasserstein GP surrogate configuration. |
| `conf/kernel/` | RBF and Matern-5/2 GP kernels. |
| `conf/warper/` | Identity and Wasserstein arc-length time warps. |
| `conf/reference/` | Barycenter reference construction. |
| `conf/transport/` | POT optimal transport solver settings. |
| `conf/baseline/` | Uniform and random baseline acquisition functions. |

## Outputs

Experiment outputs are written under `results/` by default. Typical run folders
contain metrics, per-time errors, checkpoint errors, acquisition traces, timing
tables, metadata, and reconstruction artifacts.

The rendered paper figures are in `figures/`. Post-processing notebooks to obtain these figures are in:

- `notebooks/synthetic/`
- `notebooks/single_cell/`
- `notebooks/labor_market/`

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{huynh2026active,
  title = {Active Timepoint Selection for Learning Measure-Valued Trajectories},
  author = {Nicolas Huynh and Mihaela van der Schaar},
  booktitle = {Proceedings of the 43rd International Conference on Machine learning},
  year = {2026}
}
```

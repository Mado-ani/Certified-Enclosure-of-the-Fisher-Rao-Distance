# Certified enclosures of the Fisher–Rao distance between multivariate normal distributions

This repository contains the code and data for the paper

> Mohammed Thakir (2026). *Certified Enclosures of the Fisher–Rao Distance between Multivariate Normal Distributions*.

It computes intervals that are guaranteed to contain the Fisher–Rao (Rao) distance between two multivariate normal distributions. It also reproduces every number, table and figure in the paper and its Supplementary Materials.

## Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

`python-flint` provides the Arb ball arithmetic, which is used only for the floating-point-verified bounds.

## Quick start

```python
import numpy as np
import sys; sys.path.insert(0, "code")
from frmvn import rho_co, lb_marginal, curve_estimate, beta_bounds, shoot

N1 = (np.zeros(2), np.diag([1.0, 0.1]))          # (mean, covariance)
N2 = (np.ones(2),  np.diag([0.1, 1.0]))

lb = max(rho_co(N1, N2), lb_marginal(N1, N2))    # certified lower bound
ub = curve_estimate(N1, N2, kind="co", T=128)    # certified upper bound
print(lb, ub)                                    # 3.0470 <= rho <= 3.1420
print(shoot(N1, N2)[0])                          # reference value 3.13293
```

## Repository layout

| Path | Contents |
|---|---|
| `code/frmvn.py` | Core library: Jeffreys divergence, curves, curve-length estimator, Calvo–Oller and marginal lower bounds, β-certificate, single- and multiple-shooting reference solvers |
| `code/verified.py` | Floating-point-rigorous versions of the bounds (ball arithmetic) |
| `code/geo_enclosure.py` | Enclosure along a computed geodesic (Theorem 3.6), in ball arithmetic |
| `code/nielsen23b.py` | Reproduction of the lifted curve and recursive algorithm analysed in Section 3.5 |
| `code/experiments.py` | All simulations; writes `results/*.json` and `figures/*.pdf` |
| `code/make_short_tables.py` | Tables of the main paper → `tables/main/` |
| `code/make_tables.py` | Tables of the Supplementary Materials → `tables/supplement/` |
| `results/` | Saved outputs of all experiments (JSON) |
| `figures/` | Figures of the paper and supplement (PDF) |

## Reproducing the results

The saved results let you rebuild all tables without rerunning the simulations:

```bash
cd code
python make_short_tables.py
python make_tables.py
```

To rerun everything from scratch (about 20 minutes on one core):

```bash
cd code
python experiments.py examples fig17 conv commoncov mcA mcB   # core experiments
python experiments.py verify_mc highd regime literature_check # verification, high dimension, comparison
python make_short_tables.py && python make_tables.py
```

| Paper item | Produced by |
|---|---|
| Table 1 (summary), Table 2 (lifted curve), Table 3 (high dimension) | `experiments.py mcA mcB literature_check highd` → `make_short_tables.py` |
| Figure 1 (regime map) | `experiments.py regime` |
| Supplementary tables and figures | `experiments.py examples fig17 conv commoncov verify_mc` → `make_tables.py` |

All random streams are seeded, so reruns reproduce the saved results.

## Citation

If you use this code, please cite the paper above. [BibTeX to be added on publication.]

## License

MIT; see `LICENSE`.

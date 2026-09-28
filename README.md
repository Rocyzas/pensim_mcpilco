# Data-efficient Reinforcement Learning for Non-stationary Fed-batch Processes

MSc Dissertation — University of Edinburgh, School of Informatics, 2026  
**Rokas Pranevičius**.

Supervised by **Prof. Christopher Lucas**

---

## Overview

This repository contains the implementation for an MSc dissertation adapting **MC-PILCO** to optimise a non-stationary fed-batch penicillin fermentation process.

Two research questions investigated:

**RQ1** -- Can MC-PILCO be adapted to reliably improve penicillin yield in a fed-batch simulator under severe data constraints.

**RQ2** -- Does explicitly modelling the growth/production phase transition as two independently trained GP dynamics models (dual-GP) improve upon a single GP.

---

## Key Results

| Method | Mean \delta Yield (kg) | SD across seeds | Violations (30 batches) |
|---|---|---|---|
| Recipe (baseline) | — | — | 0/10 |
| Bayesian Optimisation (15 evals) | +158.7 | +-193.8 | 4/30 |
| MC-PILCO, no time | +148.9 | +-45.0 | 0/30 |
| MC-PILCO, time added | +387.5 | +-56.5 | 1/30 |

---

## Repository Structure

```
pensim_mcpilco/
├── pensim_mcpilco/        # Main package
│   ├── env/               # PenSimPy environment wrapper
│   ├── gp/                # GP dynamics models
│   ├── policy/
│   ├── pilco/             # MC-PILCO algorithm
│   ├── costs/             # Reward functions (including the ones not used)
│   └── notebooks/         # Experiments
├── simple_runner.sh       # bash file for cluster
├── run.sh                 # bash file for cluster
└── README.md
```

---

## Environment

The simulator is **PenSimPy** ([Zhang, 2020](https://github.com/Mohan-Zhang-u/PenSimPy)).

**State space** (4 variables + optional time):
Biomass X (g/L), Penicillin P (g/L), Vessel weight Wt (kg), Viscosity (cP), Time (h) (optional).

P and X are assumed online-observable via Raman spectroscopy with chemometric calibration.

**Action**: Residual correction on the substrate feed rate Fs, bounded at +-50% of the recipe setpoint:

```
u_t = recipe_t * (1 + 0.5 * a_t),   a_t ∈ [-1, 1]
```

This residual parameterisation improves safety and data efficiency by keeping the agent close to a near-optimal industrial recipe.

---

## Algorithm

MC-PILCO ([Amadio et al., 2022](https://doi.org/10.1109/TRO.2022.3184837)) propagates particles through a GP dynamics model and estimates the policy gradient by Monte Carlo.

**Dual-GP extension**: two independently trained GPs (growth phase, production phase) blended via a logistic sigmoid centred on the phase pivot, with combined variance accounting for between-phase spread. Three pivot types tested: morphological (fixed, ~100 h), biomass-weight proxy (adaptive per episode), and per-rollout particle-level.

---

## Installation

```bash
git clone https://github.com/Rocyzas/pensim_mcpilco.git
cd pensim_mcpilco
pip install -e .
```

**Dependencies**: Python 3.9+, PyTorch, GPyTorch, PenSimPy, NumPy, pandas, Matplotlib, SciPy.

---

## Usage

**Single run:**

```bash
bash simple_runner.sh \
  --seed 3 \
  --out_dir results/run_seed3 \
  --num_trials 10 \
  --visc_penalty 0.5 \
  --risk_weight 0.01
```

**Key flags:**

| Flag | Description | Default |
|---|---|---|
| `--seed` | Training seed (controls policy init and particle sampling) | 3 |
| `--num_trials` | Number of policy-search trials | 10 |
| `--visc_penalty` | Weight on viscosity constraint penalty | 0.5 |
| `--risk_weight` | Weight on particle spread penalty | 0.01 |
| `--add_time` | Include time as GP input | False |
| `--dual_gp` | Enable dual-GP phase decomposition | False |
| `--phase_pivot` | Phase pivot type: `morphological`, `biomass`, `rollout` | `morphological` |
| `--num_high_feed_probes` | Number of high-feed exploration probes | 0 |
| `--out_dir` | Output directory for results and plots | `results/` |

---

## Experimental Design

- **Training**: 3 seeds * 10 policy-search trials per configuration
- **Evaluation**: 10 held-out batch seeds per training seed (30 seed * batch evaluations per configuration)
- **Baseline**: Bayesian Optimisation with +-50% recipe residual Fs control, switching every 25 h
- **Primary metric**: total penicillin yield (kg), reported as \delta relative to the recipe baseline of the same seed
- **Constraint metrics**: peak viscosity >100 cP and vessel weight >1.1*10^4 kg

Seeds are shared across all methods to allow paired statistical comparisons.

---

## Reproducibility Notes

- PenSimPy diverges numerically from the MATLAB IndPenSim implementation due to solver differences (SciPy LSODA vs MATLAB ODE solver) and recipe setpoint reading. This implementation follows MATLAB behaviour (`right_sp` at each breakpoint) as it is verified by multiple published studies.
- Historical IndPenSim batches cannot be reused and must be regenerated under this simulator version.
- Results are not directly comparable to published IndPenSim benchmarks using the MATLAB implementation.

---

## Citation

If you use this code or the methods described, please cite the dissertation:

```
Pranevičius, R. (2026). Data-efficient Reinforcement Learning for Non-stationary 
Fed-batch Processes. MSc Dissertation, University of Edinburgh, School of Informatics.
```

The MC-PILCO algorithm is from:

```
Amadio, F., Dalla Libera, A., Antonello, R., Nikovski, D., Carli, R., & Romeres, D. (2022).
Model-based policy search using Monte Carlo gradient estimation with real systems application.
IEEE Transactions on Robotics, 38(6), 3879–3898.
```

The simulator is:

```
Zhang, M. (2020). PenSimPy: The Python implementation of IndPenSim.
https://github.com/Mohan-Zhang-u/PenSimPy
```

---

## Limitations and Future Work

- Single action channel (substrate feed Fs) has limited control authority over penicillin yield relative to inter-batch variability — the main bottleneck is action-signal identifiability, not dynamics modelling.
- n=3 training seeds; conclusions from cross-seed comparisons carry uncertainty.
- Dual-GP decomposition is unlikely to improve control in this action space. Enriching the action space (aeration Fg, agitation RPM) is the more promising direction.
- Method generalisation to other fed-batch organisms (e.g., E. coli acetate overflow, P. pastoris glycerol-to-methanol switch) is expected but not empirically verified.

---

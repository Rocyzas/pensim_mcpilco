"""
Experiment 01: Bayesian Optimisation baseline (SMPL-style).

Reproduces the methodology documented for SMPL's PenSimEnvGym:
  - 6 controls  [discharge, Fs, Foil, Fg, pressure, Fw]  optimised within
    +/-10% of their default setpoints (Fpaa is kept at default, as in SMPL).
  - objective = batch yield  (sum of yield_per_run == SMPL's reward == the
    metric used in 00_batch_recipe_generation.py).
  - GP + Expected Improvement, 10 random starts then GP-EI search.

Runs on the patched fast=True LSODA integrator (same physics as 00 and the
MC-PILCO env). Absolute yields will not match SMPL's published ~3640 kg
(different env/IC bookkeeping); compare against our own recipe baseline.

Usage:
    cd pensim_mcpilco
    PYTHONPATH=.. python -m experiments.01_bo_baseline --n_calls 14 --n_random 6   # smoke test
    PYTHONPATH=.. python -m experiments.01_bo_baseline                              # 10 random + 50 BO
    PYTHONPATH=.. python -m experiments.01_bo_baseline --n_calls 1010 --n_random 10 # full SMPL budget
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)

from utils.ode_patch import patch_fastodeint
# stable LSODA integrator, must run before any PenSimEnv.step()
patch_fastodeint()

from skopt import gp_minimize
from skopt.space import Real


CONC_COL = "Penicillin Concentration"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bo_baseline")

# SMPL's 6 BO controls, optimised as +/-10% scalings of the default recipe.
SCALED = [DISCHARGE, FS, FOIL, FG, PRES, WATER]
DEFAULTS = {
    FS: FS_DEFAULT_PROFILE, FOIL: FOIL_DEFAULT_PROFILE, FG: FG_DEFAULT_PROFILE,
    PRES: PRESS_DEFAULT_PROFILE, DISCHARGE: DISCHARGE_DEFAULT_PROFILE,
    WATER: WATER_DEFAULT_PROFILE, PAA: PAA_DEFAULT_PROFILE,
}
PHYSICAL_P_MAX = 30.0      # g/L  sanity ceiling (flags an ODE blow-up)
RECIPE_MEAN_YIELD = 1513.48  # from 00 results/.../summary_stats.csv — update if you rerun 00


def _scaled_recipe(factors):
    """factors: dict channel -> multiplier. Returns a RecipeCombo."""
    rd = {ch: Recipe([{"time": sp["time"], "value": sp["value"] * factors.get(ch, 1.0)}
                      for sp in prof], ch)
          for ch, prof in DEFAULTS.items()}
    return RecipeCombo(recipe_dict=rd)


def _evaluate(x, seed):
    """Run one batch with the 6 scaling factors x; return (yield, max_conc)."""
    env = PenSimEnv(recipe_combo=_scaled_recipe(dict(zip(SCALED, x))), fast=True)
    (df, _df_raman), batch_yield = env.get_batches(random_seed=seed, include_raman=False)
    return batch_yield, df[CONC_COL].max()


def run(n_calls=1000, n_random=10, base_seed=0):
    space = [Real(0.9, 1.1, name=ch) for ch in SCALED]   # +/-10% of setpoint
    log = []

    def objective(x):
        idx = len(log)
        # Vary seed per call so the GP samples PenSimPy's batch-noise distribution.
        y, max_c = _evaluate(x, seed=base_seed + idx)
        if max_c > PHYSICAL_P_MAX:        # integrator blow-up -> invalid
            y = 0.0
        log.append({"call": idx, **dict(zip(SCALED, x)), "yield": y, "max_conc": max_c})
        best = max(r["yield"] for r in log)
        tag = "rand" if idx < n_random else "BO"
        print(f"[{tag} {idx + 1:3d}] yield={y:8.1f}  best={best:8.1f}")
        return -y                          # gp_minimize minimises

    gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_random,
                acq_func="EI", random_state=base_seed)   # default noise="gaussian"

    df = pd.DataFrame(log)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "bo_metrics.csv"), index=False)

    # --- convergence plot: per-eval yield + best-so-far + recipe baseline ---
    yields = df["yield"].values
    best_so_far = np.maximum.accumulate(yields)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(1, len(yields) + 1), yields, s=30, alpha=.6, label="batch yield")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, color="darkorange", lw=2, label="best so far")
    ax.axhline(RECIPE_MEAN_YIELD, color="crimson", ls="--",
               label=f"recipe baseline ({RECIPE_MEAN_YIELD:.0f})")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Yield")
    ax.set_title(f"BO (SMPL-style 6-var +/-10%) convergence, n={n_calls}")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bo_convergence.png"), dpi=150)
    plt.close(fig)

    best = df.loc[df["yield"].idxmax()]
    print(f"\nBest yield {best['yield']:.1f} vs recipe {RECIPE_MEAN_YIELD:.1f} "
          f"(delta={best['yield'] - RECIPE_MEAN_YIELD:+.1f})")
    print("Best factors:", {ch: round(float(best[ch]), 3) for ch in SCALED})
    print(f"Saved to {os.path.normpath(OUTPUT_DIR)}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_calls", type=int, default=1000)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    args = p.parse_args()
    run(args.n_calls, args.n_random, args.base_seed)

# TODO: interesting to add PAA as a 7th control in scale (diverge from SMPL BO baseline)
#       try with more batches, more aggressive scaling, different BO acquisition functions, etc.
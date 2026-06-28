"""
Experiment 01: Bayesian Optimisation baseline (SMPL-style).

Reproduces the methodology documented for SMPL's PenSimEnvGym:
  - 6 controls  [discharge, Fs, Foil, Fg, pressure, Fw]  optimised within
    +/-10% of their default setpoints (Fpaa is kept at default, as in SMPL).
  - objective = batch yield  (sum of yield_per_run == SMPL's reward == the
    metric used in 00_batch_recipe_generation.py).
  - GP + Expected Improvement, 10 random starts then GP-EI search.
  - DETERMINISTIC objective: the env is seeded once (seed_mode="fixed") so the
    recipe is the only thing the GP sees, matching SMPL. This is what lets BO
    climb the recipe gradient and lift the mean-of-all-evals ~4% above baseline
    (best ~12%), reproducing SMPL's "12% best / 4% on average". seed_mode="vary"
    changes the seed each call; batch noise (~280 kg) then swamps the +/-10%
    recipe signal and the search degenerates to ~random sampling (mean ~=
    baseline).

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

# --- make this script runnable directly (no -m / PYTHONPATH) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

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
PHYSICAL_P_MAX = 40.0
RECIPE_MULTISEED_MEAN = 3485.246

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


def run(n_calls=1000, n_random=10, base_seed=0, seed_mode="fixed"):
    space = [Real(0.9, 1.1, name=ch) for ch in SCALED]   # +/-10% of setpoint
    log = []

    # Apples-to-apples baseline: in "fixed" mode BO optimises a single seed, so
    # compare against the DEFAULT recipe at that same seed (seed 0 alone ~= 2986,
    # not the multi-seed mean 3485). In "vary" mode keep the multi-seed mean.
    if seed_mode == "fixed":
        baseline_yield, _ = _evaluate([1.0] * len(SCALED), seed=base_seed)
        baseline_label = f"default recipe @ seed {base_seed}"
    else:
        baseline_yield = RECIPE_MULTISEED_MEAN
        baseline_label = "recipe baseline (multi-seed mean)"
    print(f"Baseline ({baseline_label}) = {baseline_yield:.1f}\n")

    def objective(x):
        idx = len(log)
        # SMPL optimises a DETERMINISTIC objective: the env is seeded once so the
        # recipe is the only thing that varies, letting the GP learn the
        # recipe->yield surface. Varying the seed per call (mode="vary") injects
        # ~280 kg of batch noise that swamps the +/-10% recipe signal, so EI
        # chases noise and the search degenerates to ~random sampling of the box
        # (mean ~= baseline). Default "fixed" reproduces SMPL.
        seed = base_seed if seed_mode == "fixed" else base_seed + idx
        y, max_c = _evaluate(x, seed=seed)
        if max_c > PHYSICAL_P_MAX:        # integrator blow-up -> invalid
            y = 0.0
        log.append({"call": idx, **dict(zip(SCALED, x)), "yield": y, "max_conc": max_c})
        best = max(r["yield"] for r in log)
        tag = "rand" if idx < n_random else "BO"
        print(f"[{tag} {idx + 1:3d}] yield={y:8.1f}  best={best:8.1f}")
        return -y                          # gp_minimize minimises

    # noise=1e-10: tell skopt the objective is deterministic (fixed seed) so the
    # GP exploits the recipe gradient instead of fitting a noise floor.
    gp_noise = 1e-10 if seed_mode == "fixed" else "gaussian"
    gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_random,
                acq_func="EI", random_state=base_seed, noise=gp_noise)

    df = pd.DataFrame(log)
    # Persist the comparison baseline so the display-only plot scripts can draw it
    # without re-running the env (constant columns; same for every row).
    df["baseline_yield"] = baseline_yield
    df["base_seed"] = base_seed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "bo_metrics.csv"), index=False)

    # --- convergence plot: per-eval yield + best-so-far + recipe baseline ---
    yields = df["yield"].values
    best_so_far = np.maximum.accumulate(yields)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(1, len(yields) + 1), yields, s=30, alpha=.6, label="batch yield")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, color="darkorange", lw=2, label="best so far")
    ax.axhline(baseline_yield, color="crimson", ls="--",
               label=f"{baseline_label} ({baseline_yield:.0f})")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Yield")
    ax.set_title(f"BO (SMPL-style 6-var +/-10%) convergence, n={n_calls}")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bo_convergence.png"), dpi=150)
    plt.close(fig)

    best = df.loc[df["yield"].idxmax()]
    overall_mean = yields.mean()
    best_impr = 100 * (best["yield"] - baseline_yield) / baseline_yield
    mean_impr = 100 * (overall_mean - baseline_yield) / baseline_yield
    print(f"\nBaseline ({baseline_label}) = {baseline_yield:.1f}")
    print(f"Best yield {best['yield']:.1f}  (delta={best['yield'] - baseline_yield:+.1f}, "
          f"{best_impr:+.2f}%)   <- SMPL 'best run' ~+12%")
    print(f"Overall mean of all evals {overall_mean:.1f}  ({mean_impr:+.2f}%)"
          f"   <- SMPL 'on average across batches' ~+4%")
    print("Best factors:", {ch: round(float(best[ch]), 3) for ch in SCALED})
    print(f"Saved to {os.path.normpath(OUTPUT_DIR)}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_calls", type=int, default=1000)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--seed_mode", choices=["fixed", "vary"], default="fixed",
                   help="fixed: deterministic objective (SMPL-style, default); "
                        "vary: change seed per call (noise-dominated, old behaviour)")
    args = p.parse_args()
    run(args.n_calls, args.n_random, args.base_seed, args.seed_mode)

# TODO: interesting to add PAA as a 7th control in scale (diverge from SMPL BO baseline)
#       try with more batches, more aggressive scaling, different BO acquisition functions, etc.
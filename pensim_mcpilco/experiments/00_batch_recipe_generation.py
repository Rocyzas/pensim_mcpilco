import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv
from PenSimPy.pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from PenSimPy.pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE

from utils.ode_patch import patch_fastodeint
# must run before any PenSimEnv.step()
patch_fastodeint()


CONC_COL = "Penicillin Concentration"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "batch_recipe_generation")


def run(n_batches = 10):
    """
    Basic batch generation example which simulates the Sequential Batch Control.
    :return: batch data and Raman spectra in pandas dataframe
    """
    recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                   FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                   FG: Recipe(FG_DEFAULT_PROFILE, FG),
                   PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                #    DISCHARGE: Recipe([{"time": 0, "value": 0}], DISCHARGE),
                   DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                   WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                   PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}

    recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
    env = PenSimEnv(recipe_combo=recipe_combo, fast=True)

    per_batch = []
    conc_curves = {}
    for i in range(n_batches):
        (df, _df_raman), batch_yield = env.get_batches(random_seed=i, include_raman=False)
        conc_curves[f"batch_{i}"] = df[CONC_COL]
        per_batch.append({
            "batch": i,
            "yield": batch_yield,
            "final_penicillin_conc": df[CONC_COL].iloc[-1],
            "max_penicillin_conc": df[CONC_COL].max(),
            "final_volume": df["Volume"].iloc[-1],
            "mean_pH": df["pH"].mean(),
            "mean_temperature": df["Temperature"].mean(),
        })
        print(f"batch {i}: yield={batch_yield:.2f}, "
              f"final conc={df[CONC_COL].iloc[-1]:.3f}")

    metrics = pd.DataFrame(per_batch).set_index("batch")
    summary = metrics.agg(["mean", "std"])

    conc_df = pd.DataFrame(conc_curves)
    conc_df.index.name = "time_h"

    # --- save CSVs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics.to_csv(os.path.join(OUTPUT_DIR, "per_batch_metrics.csv"))
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"))
    conc_df.to_csv(os.path.join(OUTPUT_DIR, "penicillin_concentration_timeseries.csv"))

    # --- plot 1: penicillin concentration over time (per batch + mean +/- std) ---
    mean_curve = conc_df.mean(axis=1)
    std_curve = conc_df.std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in conc_df.columns:
        ax.plot(conc_df.index, conc_df[col], color="0.8", linewidth=0.8)
    ax.plot(mean_curve.index, mean_curve, color="C0", linewidth=2, label="mean")
    ax.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve,
                    color="C0", alpha=0.2, label="+/- 1 std")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(CONC_COL)
    ax.set_title(f"Penicillin concentration over time (n={n_batches})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "penicillin_concentration.png"), dpi=150)
    plt.close(fig)

    # --- plot 2: yield per batch (bar + mean line) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(metrics.index, metrics["yield"], color="C1", label="batch yield")
    ax.axhline(metrics["yield"].mean(), color="k", linestyle="--",
               label=f"mean = {metrics['yield'].mean():.2f}")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Yield")
    ax.set_title(f"Yield per batch (n={n_batches})")
    ax.set_xticks(metrics.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "yield_per_batch.png"), dpi=150)
    plt.close(fig)

    print("\nSummary (mean / std):")
    print(summary)
    print(f"\nSaved CSVs and PNGs to {os.path.normpath(OUTPUT_DIR)}")
    return metrics, summary
if __name__ == "__main__":
    run(n_batches=10)

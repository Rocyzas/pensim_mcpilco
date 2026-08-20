import os

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
from PenSimPy.pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from PenSimPy.pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE

from utils.ode_patch import patch_fastodeint
# must run before any PenSimEnv.step()
patch_fastodeint()


CONC_COL = "Penicillin Concentration"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "batch_recipe_generation")

# Constraint thresholds an actual MC-PILCO policy is trained/penalised against (see
# mcpilco/pensim_wrapper.py:272-276, mcpilco/penicillin_cost.py:21-25) -- duplicated here as
# plain floats rather than importing mcpilco.pensim_wrapper, which pulls in the full
# MC-PILCO/torch stack this recipe-only script doesn't otherwise need. Wt is in kg (Channel
# 'Vessel Weight', y_unit 'Kg' -- PenSimPy/pensimpy/data/batch_data.py:32), not the separate
# Volume (L) channel.
VISC_SOFT_START = 80.0   # penalty ramp begins
VISC_MAX = 100.0         # hard ceiling / collapse regime
WT_SOFT_START = 1.1e5    # penalty ramp begins (kg)
WT_OVERFLOW = 1.2e5      # hard tank-overflow limit (kg)


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
    paa_curves = {}        # PAA concentration (mg/L) -- only on raw batch_data
    visc_curves = {}       # Viscosity (cP)           -- only on raw batch_data
    biomass_curves = {}    # Biomass concen. (g/L)    -- only on raw batch_data
    weight_curves = {}     # Vessel Weight (kg)
    fpaa_curves = {}       # PAA flow-rate setpoint (L/h)
    for i in range(n_batches):
        (df, _df_raman), batch_yield, bx = env.get_batches(
            random_seed=i, include_raman=False, return_batch_data=True)
        conc_curves[f"batch_{i}"] = df[CONC_COL]
        paa_curves[f"batch_{i}"] = pd.Series(bx.PAA.y, index=df.index)
        visc_curves[f"batch_{i}"] = pd.Series(bx.Viscosity.y, index=df.index)
        biomass_curves[f"batch_{i}"] = pd.Series(bx.X.y, index=df.index)
        weight_curves[f"batch_{i}"] = df["Vessel Weight"]
        fpaa_curves[f"batch_{i}"] = df["Phenylacetic acid flow-rate"]

        max_visc = visc_curves[f"batch_{i}"].max()
        max_wt = weight_curves[f"batch_{i}"].max()
        visc_collapse = max_visc > VISC_MAX
        wt_collapse = max_wt > WT_OVERFLOW
        per_batch.append({
            "batch": i,
            "yield": batch_yield,
            "final_penicillin_conc": df[CONC_COL].iloc[-1],
            "max_penicillin_conc": df[CONC_COL].max(),
            "mean_penicillin_conc": df[CONC_COL].mean(),
            "final_volume": df["Volume"].iloc[-1],
            "mean_pH": df["pH"].mean(),
            "mean_temperature": df["Temperature"].mean(),
            "final_biomass": biomass_curves[f"batch_{i}"].iloc[-1],
            "max_biomass": biomass_curves[f"batch_{i}"].max(),
            "mean_biomass": biomass_curves[f"batch_{i}"].mean(),
            "final_weight": weight_curves[f"batch_{i}"].iloc[-1],
            "max_weight": max_wt,
            "mean_weight": weight_curves[f"batch_{i}"].mean(),
            "max_viscosity": max_visc,
            "mean_viscosity": visc_curves[f"batch_{i}"].mean(),
            "visc_soft_violation": max_visc > VISC_SOFT_START,
            "visc_collapse": visc_collapse,
            "wt_soft_violation": max_wt > WT_SOFT_START,
            "wt_collapse": wt_collapse,
            "collapsed": visc_collapse or wt_collapse,
        })
        print(f"batch {i}: yield={batch_yield:.2f}, "
              f"final conc={df[CONC_COL].iloc[-1]:.3f}, "
              f"max visc={max_visc:.1f} cP, max weight={max_wt:.0f} kg"
              f"{'  [COLLAPSED]' if (visc_collapse or wt_collapse) else ''}")

    metrics = pd.DataFrame(per_batch).set_index("batch")
    summary = metrics.agg(["mean", "std"])

    conc_df = pd.DataFrame(conc_curves)
    conc_df.index.name = "time_h"
    paa_df = pd.DataFrame(paa_curves);         paa_df.index.name = "time_h"
    visc_df = pd.DataFrame(visc_curves);       visc_df.index.name = "time_h"
    biomass_df = pd.DataFrame(biomass_curves); biomass_df.index.name = "time_h"
    weight_df = pd.DataFrame(weight_curves);   weight_df.index.name = "time_h"
    fpaa_df = pd.DataFrame(fpaa_curves);       fpaa_df.index.name = "time_h"

    # --- save CSVs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics.to_csv(os.path.join(OUTPUT_DIR, "per_batch_metrics.csv"))
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"))
    conc_df.to_csv(os.path.join(OUTPUT_DIR, "penicillin_concentration_timeseries.csv"))
    paa_df.to_csv(os.path.join(OUTPUT_DIR, "paa_concentration_timeseries.csv"))
    visc_df.to_csv(os.path.join(OUTPUT_DIR, "viscosity_timeseries.csv"))
    biomass_df.to_csv(os.path.join(OUTPUT_DIR, "biomass_timeseries.csv"))
    weight_df.to_csv(os.path.join(OUTPUT_DIR, "weight_timeseries.csv"))
    fpaa_df.to_csv(os.path.join(OUTPUT_DIR, "fpaa_setpoint_timeseries.csv"))

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

    # --- plot 3: viscosity over time (per batch + mean +/- std), soft/hard limits marked ---
    mean_v, std_v = visc_df.mean(axis=1), visc_df.std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in visc_df.columns:
        ax.plot(visc_df.index, visc_df[col], color="0.8", linewidth=0.8)
    ax.plot(mean_v.index, mean_v, color="C2", linewidth=2, label="mean")
    ax.fill_between(mean_v.index, mean_v - std_v, mean_v + std_v,
                    color="C2", alpha=0.2, label="+/- 1 std")
    ax.axhline(VISC_SOFT_START, color="orange", linestyle=":",
               label=f"soft limit ({VISC_SOFT_START:g} cP)")
    ax.axhline(VISC_MAX, color="red", linestyle="--",
               label=f"collapse limit ({VISC_MAX:g} cP)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Viscosity (cP)")
    ax.set_title(f"Viscosity over time (n={n_batches})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "viscosity.png"), dpi=150)
    plt.close(fig)

    # --- plot 4: vessel weight over time (per batch + mean +/- std), soft/hard limits marked ---
    mean_w, std_w = weight_df.mean(axis=1), weight_df.std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in weight_df.columns:
        ax.plot(weight_df.index, weight_df[col], color="0.8", linewidth=0.8)
    ax.plot(mean_w.index, mean_w, color="C4", linewidth=2, label="mean")
    ax.fill_between(mean_w.index, mean_w - std_w, mean_w + std_w,
                    color="C4", alpha=0.2, label="+/- 1 std")
    ax.axhline(WT_SOFT_START, color="orange", linestyle=":",
               label=f"soft limit ({WT_SOFT_START:.2e} kg)")
    ax.axhline(WT_OVERFLOW, color="red", linestyle="--",
               label=f"overflow limit ({WT_OVERFLOW:.2e} kg)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Vessel Weight (kg)")
    ax.set_title(f"Vessel weight over time (n={n_batches})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "weight.png"), dpi=150)
    plt.close(fig)

    n_collapsed = int(metrics["collapsed"].sum())
    n_visc = int(metrics["visc_collapse"].sum())
    n_wt = int(metrics["wt_collapse"].sum())
    print(f"\nCollapsed batches: {n_collapsed}/{n_batches} "
          f"(viscosity > {VISC_MAX:g} cP: {n_visc}, weight > {WT_OVERFLOW:.2e} kg: {n_wt})")

    print("\nSummary (mean / std):")
    print(summary)
    print(f"\nSaved CSVs and PNGs to {os.path.normpath(OUTPUT_DIR)}")
    return metrics, summary
if __name__ == "__main__":
    run(n_batches=100)

"""
PYTHONPATH=.. python -m evaluations.report_pivot_plots
    [--clock_dir results/biomass/clock_pivot/seed4_1]
    [--general_dir results/biomass/general/seed4_1]
    [--rollout_dir results/biomass/rollout/seed4_1]
    [--out results/biomass/pivot_report_summary.png]

Report-ready, simplified versions of the 3 phase-split/pivot diagnostics already computed for
the biomass-pivot experiments (results/biomass/{clock_pivot,general,rollout}/seed4_1/). NOT a
rigorous re-derivation -- deliberately simplified for a write-up: ONE horizontally-elongated
plot, all three pivot types overlaid on the SAME axes, one colour per series, minimal (3-entry)
legend. Reads only already-cached CSVs/note.txt -- no GP reconstruction, no simulator rollouts.

  1. "Fixed pivot" (clock_pivot, pivot_mode="time"): the exact blend-weight sigmoid formula used
     by DualPhaseModelLearning._blend_weight (mcpilco/model_learning_dual_phase.py) --
     w(t) = 1 / (1 + exp(-k*(t - pivot_hours))), k = ln(99)/blend_half_width_hours -- evaluated
     directly from clock_pivot/seed4_1's own note.txt parameters. No variance shown: this pivot
     is deterministic by construction, identical on every rollout.

  2. "Dynamical pivot" (general, pivot_mode="biomass", on_each_rollout=False): reuses the 15
     cached per-training-episode biomass-crossing hours in C0f_split_distribution.csv (NOT
     C0_blend_weight_sanity.csv -- see check_blend_weight_sanity's own docstring: that check
     stays time-based whenever on_each_rollout=False, since pivot_mode="biomass" only changes
     how the training data is split, not the rollout-time blend query, so it isn't a real
     biomass signal). Builds one illustrative sigmoid per episode, centred on that episode's own
     crossing hour, same width/approximation approach as series 3, then takes the median and
     5-95th percentile across episodes at each time point.

  3. "Dynamical pivot for rollouts" (rollout, pivot_mode="biomass", on_each_rollout=True): reuses
     the 400 cached per-particle crossing times in C0g_rollout_pivot_distribution.csv. Builds one
     illustrative sigmoid per particle, centred on that particle's own crossing time, using the
     SAME width as series 1 (not each particle's true biomass-space width, which doesn't
     translate cleanly to an hours-space width -- an approximation, again fine for a report
     figure), then takes the median and 5-95th percentile across particles at each time point.
"""
import argparse
import math
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluations.eval_single_phase_lib import parse_run_params

COLOR_FIXED = "C0"
COLOR_GENERAL = "C1"
COLOR_ROLLOUT = "C2"
BAND_ALPHA = 0.25
T_GRID = np.linspace(0, 230, 400)


def sigmoid_weight(t, pivot_hours, half_width_hours):
    k = math.log(99.0) / half_width_hours
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(t) - pivot_hours)))


def fixed_curve(run_dir):
    params = parse_run_params(Path(run_dir) / "note.txt")
    pivot_hours, half_width = params["pivot_hours"], params["blend_half_width_hours"]
    return T_GRID, sigmoid_weight(T_GRID, pivot_hours, half_width)


def dynamic_general_band(run_dir):
    """Variance across the ACTUAL biomass-threshold crossing hour of every training episode
    (C0f_split_distribution.csv) -- NOT C0_blend_weight_sanity.csv, which (per its own
    docstring/on-figure warning in eval_multi_phase_lib.check_blend_weight_sanity) stays
    time-based whenever on_each_rollout=False: pivot_mode="biomass" only changes how the
    TRAINING DATA is split, not that rollout-time blend query, so that CSV looks identical to a
    fixed pivot and isn't the right source for "dynamical, with variance across episodes"."""
    df = pd.read_csv(Path(run_dir) / "C0f_split_distribution.csv")
    params = parse_run_params(Path(run_dir) / "note.txt")
    half_width = params["blend_half_width_hours"]  # same illustrative width as the fixed pivot
    ws = np.stack([sigmoid_weight(T_GRID, h, half_width) for h in df["hours"].values])
    lo, hi = np.percentile(ws, [5, 95], axis=0)
    return T_GRID, lo, hi, np.median(ws, axis=0), float(np.median(df["hours"].values))


def dynamic_rollout_band(run_dir):
    """Variance across the 400 cached per-particle pivot-crossing times of one rollout."""
    df = pd.read_csv(Path(run_dir) / "C0g_rollout_pivot_distribution.csv")
    params = parse_run_params(Path(run_dir) / "note.txt")
    half_width = params["blend_half_width_hours"]  # same illustrative width as the fixed pivot
    ws = np.stack([sigmoid_weight(T_GRID, tc, half_width) for tc in df["t_cross_h"].values])
    p5, p95 = np.percentile(ws, [5, 95], axis=0)
    return T_GRID, p5, p95, np.median(ws, axis=0), float(np.median(df["t_cross_h"].values))


def plot_overlay(ax, clock_dir, general_dir, rollout_dir):
    params = parse_run_params(Path(clock_dir) / "note.txt")
    pivot1 = params["pivot_hours"]
    t1, w1 = fixed_curve(clock_dir)
    ax.plot(t1, w1, color=COLOR_FIXED, lw=2.5, label="Fixed pivot")
    ax.axvline(pivot1, color=COLOR_FIXED, ls="--", lw=1.5, alpha=.65)

    t2, lo2, hi2, mean2, pivot2 = dynamic_general_band(general_dir)
    ax.fill_between(t2, lo2, hi2, color=COLOR_GENERAL, alpha=BAND_ALPHA, lw=0)
    ax.plot(t2, mean2, color=COLOR_GENERAL, lw=2.5, label="Dynamical pivot")
    ax.axvline(pivot2, color=COLOR_GENERAL, ls="--", lw=1.5, alpha=.65)

    t3, lo3, hi3, med3, pivot3 = dynamic_rollout_band(rollout_dir)
    ax.fill_between(t3, lo3, hi3, color=COLOR_ROLLOUT, alpha=BAND_ALPHA, lw=0)
    ax.plot(t3, med3, color=COLOR_ROLLOUT, lw=2.5, label="Dynamical pivot for rollouts")
    ax.axvline(pivot3, color=COLOR_ROLLOUT, ls="--", lw=1.5, alpha=.65)

    ax.set_title("Phase transition. Fixed vs. Dynamical vs. Dynamical per rollout", fontsize=14)
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("Blend Weight")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=.25)
    ax.legend(fontsize=20, loc="lower right")


def plot_distribution(ax, clock_dir, general_dir, rollout_dir):
    """Distribution of pivot CROSSING TIMES themselves (not the blend curves) -- same idea as
    the left panel of C0g_rollout_pivot_distribution.png, extended to all three pivot types on
    one shared axis: the fixed pivot has no distribution (a single vertical line), the
    dynamical pivot's 15 per-training-episode crossing hours and the per-rollout pivot's 400
    per-particle crossing hours are both histograms, density-normalised (very different sample
    sizes: 15 vs 400) and drawn on shared bins so their shapes are directly comparable."""
    pivot_fixed = parse_run_params(Path(clock_dir) / "note.txt")["pivot_hours"]
    general_hours = pd.read_csv(Path(general_dir) / "C0f_split_distribution.csv")["hours"].values
    rollout_hours = pd.read_csv(
        Path(rollout_dir) / "C0g_rollout_pivot_distribution.csv")["t_cross_h"].values

    all_vals = np.concatenate([general_hours, rollout_hours, [pivot_fixed]])
    bins = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 20)

    ax.hist(rollout_hours, bins=bins, color=COLOR_ROLLOUT, alpha=.55, density=True,
           label="Dynamical pivot for rollouts")
    ax.hist(general_hours, bins=bins, color=COLOR_GENERAL, alpha=.75, density=True,
           label="Dynamical pivot")
    ax.axvline(pivot_fixed, color=COLOR_FIXED, ls="--", lw=2.5, label="Fixed pivot")

    ax.set_title("Pivot crossing-time distribution", fontsize=14)
    ax.set_xlabel("crossing time (h)")
    ax.set_ylabel("density")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, loc="upper right")


def main(clock_dir, general_dir, rollout_dir, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_overlay(ax, clock_dir, general_dir, rollout_dir)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clock_dir", default="../results/biomass/clock_pivot/seed4_1")
    p.add_argument("--general_dir", default="../results/biomass/general/seed4_1")
    p.add_argument("--rollout_dir", default="../results/biomass/rollout/seed4_1")
    p.add_argument("--out", default="../results/biomass/pivot_report_summary.png")
    args = p.parse_args()
    main(args.clock_dir, args.general_dir, args.rollout_dir, args.out)

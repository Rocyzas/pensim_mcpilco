"""Offline growth->production phase-transition (pivot) diagnostic.

The dual-phase GP's pivot is a hardcoded constant, PIVOT_HOURS=100 (mcpilco/pensim_wrapper.py),
used only to split GP training data between phase-1/phase-2 -- never derived from data. This
script estimates the *actual* pivot per batch, grounded in the structured-biomass model IndPenSim
implements: growing hyphal regions (A0) either branch (stay A0) or differentiate into non-growing,
product-forming regions (A1). The paper's cleanest scalar trigger is the peak of A0 -- where
dA0/dt changes sign from positive to negative. Unlike a real plant, this simulator tracks A0/A1
directly as ODE states (bx.a0.y / bx.a1.y), so no proxy or arbitrary weighting is needed for the
primary detector. A secondary, no-A0-required cross-check (specific growth rate mu_X collapsing
while substrate S is driven down toward the production-rate center) is also computed, since that
is what a real plant without direct A0/A1 access would have to rely on instead.

Strictly offline/post-hoc, and touches no existing file: every input is read directly off
PenSimEnv.get_batches(..., return_batch_data=True)'s raw batch_data object.

Seed argument matches experiments/02_mcpilco_single_phase.py / 03_mcpilco_dual_phase.py's
--seed: both configs set wrapper_par["seed_offset"] = seed*1000 (config_single_phase.py:194,
config_dual_phase.py:174), so a training run started with --seed N generates its first
(exploration episode 0) batch at simulator random_seed = N*1000. This script reproduces that
exact batch for direct comparison -- pass the same --seed value here.

Usage:
    python phase_transition_diagnostic.py 1            # single seed (matches --seed 1)
    python phase_transition_diagnostic.py 1-5           # range, inclusive (matches --seed 1..5)
    python phase_transition_diagnostic.py 1,3,5-7 --show
or:
    from evaluations.phase_transition_diagnostic import run
    results = run(training_seeds=[0, 1, 2], show=True)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from utils.constants import STEP_IN_HOURS
from utils.peni_env_setup import PenSimEnv
# Importing mcpilco.pensim_wrapper already calls patch_fastodeint() at module load time
# (required before any PenSimEnv.step()/get_batches() call), so it's not repeated here.
from mcpilco.pensim_wrapper import PenSimWrapper, PIVOT_HOURS

# Gaussian center (g/L) of the penicillin-production-rate term r_p -- the substrate
# concentration production is centered around, i.e. the paper's "s_max_P" analogue. Hardcoded
# local constant in PenSimPy/pensimpy/ode/indpensim_ode_py.py:17 (mirrors
# IndPenSim_V2.02/Parameter_list.m:8); not exported by that module, so duplicated here.
MEAN_P = 0.002

# Matches config_single_phase.py:194 / config_dual_phase.py:174's wrapper_par["seed_offset"] =
# seed*1000 -- the mapping from a training run's --seed to its actual (episode-0) simulator seed.
SEED_MULTIPLIER = 1000

PIVOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point")


def _parse_seed_spec(spec):
    """Parses '1', '1-5', or a comma-separated mix like '1,3,5-7' into a sorted list of unique
    training seeds (the same values you'd pass as --seed to experiments/02.../03...)."""
    seeds = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-")
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def get_trajectory(seed):
    """Runs one default-recipe batch and pulls its raw per-timestep channels."""
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    (_df, _df_raman), _yield, bx = env.get_batches(
        random_seed=seed, include_raman=False, return_batch_data=True)
    n = len(bx.X.y)
    t = np.array([(i + 1) * STEP_IN_HOURS for i in range(n)])
    return {
        "t": t,
        "X": np.array(bx.X.y), "S": np.array(bx.S.y), "P": np.array(bx.P.y),
        "a0": np.array(bx.a0.y), "a1": np.array(bx.a1.y),
        "OUR": np.array(bx.OUR.y), "CER": np.array(bx.CER.y),
        "Fs": np.array(bx.Fs.y),
        # Wt, Viscosity: not used by the primary/proxy detectors here, but included so
        # phase_transition_ruptures.py can restrict itself to STATE_NAMES's own channels
        # (pensim_wrapper.py:41) -- what the dual-phase GP model actually observes.
        "Wt": np.array(bx.Wt.y), "Viscosity": np.array(bx.Viscosity.y),
    }


def _smooth(y, t, window_h):
    dt = float(np.mean(np.diff(t)))
    window = max(5, int(round(window_h / dt)))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    return savgol_filter(y, window, polyorder=min(3, window - 1))


def detect_pivot_a0(t, a0, smooth_window_h=3.0, exclude_before_h=5.0, exclude_after_h=10.0):
    """Primary detector: time of A0's peak (dA0/dt sign change, + to -), the model's own
    growth->production switch. No weights, nothing arbitrary."""
    a0_s = _smooth(a0, t, smooth_window_h)
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idx = np.flatnonzero(mask)[np.argmax(a0_s[mask])]
    return float(t[idx]), a0_s


def compute_proxy_signals(t, X, S, P, smooth_window_h=3.0):
    """No-A0-required signals: specific growth rate mu_X = Xdot/X, and dP/dt -- the measurable
    proxies a real plant (without direct A0/A1 access) would have to rely on instead."""
    X_s, S_s, P_s = (_smooth(v, t, smooth_window_h) for v in (X, S, P))
    Xdot, Pdot = np.gradient(X_s, t), np.gradient(P_s, t)
    return {"X_s": X_s, "S_s": S_s, "P_s": P_s, "Xdot": Xdot, "Pdot": Pdot,
            "mu_X": Xdot / np.clip(X_s, 1e-6, None)}


def detect_pivot_proxy(t, proxy, S, exclude_before_h=5.0, exclude_after_h=10.0,
                        mu_frac=0.2, s_percentile=10.0):
    """Secondary cross-check: first time after mu_X's own peak where growth rate has collapsed
    to `mu_frac` of its peak AND substrate has been driven down near mean_P (approximated as
    below the batch's own `s_percentile`, since mean_P=0.002 g/L is tiny relative to typical S
    scale and a hard absolute threshold would be fragile). Returns NaN if never jointly met."""
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    mu_X = proxy["mu_X"]
    peak_idx = idxs[np.argmax(mu_X[idxs])]
    mu_peak = mu_X[peak_idx]
    s_thresh = np.percentile(S[idxs], s_percentile)
    for idx in idxs[idxs >= peak_idx]:
        if mu_X[idx] <= mu_frac * mu_peak and S[idx] <= s_thresh:
            return float(t[idx])
    return float("nan")


def plot_and_report(training_seed, out_dir=None, show=False):
    """training_seed is the value you'd pass as --seed to experiments/02.../03...; internally
    mapped to the same simulator seed those scripts use for episode 0 (see SEED_MULTIPLIER)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj = get_trajectory(sim_seed)
    t = traj["t"]

    pivot_a0, a0_s = detect_pivot_a0(t, traj["a0"])
    proxy = compute_proxy_signals(t, traj["X"], traj["S"], traj["P"])
    pivot_proxy = detect_pivot_proxy(t, proxy, traj["S"])
    RQ = traj["CER"] / np.clip(traj["OUR"], 1e-9, None)

    print(f"[seed {training_seed} (sim_seed={sim_seed})] A0-peak pivot = {pivot_a0:.1f} h | "
          f"proxy (mu_X collapse + S->mean_P) pivot = {pivot_proxy:.1f} h | "
          f"hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h")

    fig, axes = plt.subplots(5, 1, figsize=(9, 16), sharex=True)

    ax = axes[0]
    ax.plot(t, traj["a0"], color="0.7", lw=0.8, label="A0 (raw)")
    ax.plot(t, a0_s, color="C0", lw=1.5, label="A0 (smoothed)")
    ax.plot(t, traj["a1"], color="C1", lw=1.2, label="A1")
    ax.axvline(pivot_a0, color="purple", lw=1.5, label=f"A0-peak pivot ({pivot_a0:.1f} h)")
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0, label=f"hardcoded PIVOT_HOURS ({PIVOT_HOURS:g} h)")
    ax.set_ylabel("Biomass region (g/L)")
    ax.set_title(f"Growth->production pivot -- seed {training_seed} (sim_seed={sim_seed})")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    ax.plot(t, traj["X"], label="X (biomass)")
    ax.plot(t, traj["S"], label="S (substrate)")
    ax.plot(t, traj["P"], label="P (penicillin)")
    ax.axvline(pivot_a0, color="purple", lw=1.0)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("Conc. (g/L)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t, proxy["mu_X"], color="C0", label="mu_X (specific growth rate, 1/h)")
    axb = ax.twinx()
    axb.plot(t, traj["S"], color="C2", alpha=0.6, label="S")
    axb.axhline(MEAN_P, color="C3", ls=":", lw=1.0, label=f"mean_P ({MEAN_P} g/L)")
    ax.axvline(pivot_proxy, color="teal", lw=1.5, label=f"proxy pivot ({pivot_proxy:.1f} h)")
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("mu_X (1/h)")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")

    ax = axes[3]
    ax.plot(t, proxy["Pdot"], color="C0", label="dP/dt")
    axb = ax.twinx()
    axb.plot(t, traj["Fs"], color="C4", alpha=0.7, label="Fs")
    ax.axvline(pivot_a0, color="purple", lw=1.0)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("dP/dt")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")

    ax = axes[4]
    ax.plot(t, traj["OUR"], label="OUR")
    ax.plot(t, traj["CER"], label="CER")
    axb = ax.twinx()
    axb.plot(t, RQ, color="C5", alpha=0.7, label="RQ = CER/OUR")
    ax.axvline(pivot_a0, color="purple", lw=1.0)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Rate")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")

    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"phase_transition_seed{training_seed}.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"training_seed": training_seed, "sim_seed": sim_seed, "pivot_a0_hours": pivot_a0,
            "pivot_proxy_hours": pivot_proxy, "hardcoded_pivot_hours": PIVOT_HOURS}


def _save_csv(results, out_dir):
    """Per-seed pivot points, plus a mean/std/min/max range summary over this run's seeds."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pivot_points.csv")
    fieldnames = ["training_seed", "sim_seed", "pivot_a0_hours", "pivot_proxy_hours",
                  "hardcoded_pivot_hours"]
    a0_vals = np.array([r["pivot_a0_hours"] for r in results])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
        for label, val in (("mean", np.mean(a0_vals)), ("std", np.std(a0_vals)),
                           ("min", np.min(a0_vals)), ("max", np.max(a0_vals))):
            w.writerow({"training_seed": label, "sim_seed": "", "pivot_a0_hours": val,
                        "pivot_proxy_hours": "", "hardcoded_pivot_hours": PIVOT_HOURS})
    print(f"Saved {path}")
    return path


def run(training_seeds=(0,), out_dir=PIVOT_DIR, show=False):
    results = [plot_and_report(s, out_dir=out_dir, show=show) for s in training_seeds]
    a0_vals = [r["pivot_a0_hours"] for r in results]
    print(f"\nMean A0-peak pivot over {len(results)} batch(es): "
          f"{np.mean(a0_vals):.1f} +/- {np.std(a0_vals):.1f} h "
          f"(hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h)")
    if out_dir:
        _save_csv(results, out_dir)
        print(f"Saved plots to {os.path.normpath(out_dir)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=str,
                        help="Seed(s) to evaluate, matching the --seed you'd pass to "
                             "experiments/02_mcpilco_single_phase.py / 03_mcpilco_dual_phase.py. "
                             "Single int ('1'), inclusive range ('1-5'), or comma-separated "
                             "mix ('1,3,5-7').")
    parser.add_argument("--out_dir", type=str, default=PIVOT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(training_seeds=_parse_seed_spec(args.seeds), out_dir=args.out_dir, show=args.show)

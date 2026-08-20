"""Where does the BIOMASS pivot (--pivot_mode biomass) actually land, batch by batch?

Companion to phase_transition_diagnostic_updated.py. That script asks "where is the true
growth->production transition?" using mechanistic detectors (K_diff floor, A0 peak, substrate
proxy). This one asks the narrower operational question: given the threshold that
DualPhaseModelLearning actually uses, at what HOUR does each batch get split -- and how does
that compare both to the fixed pivot_hours it replaces and to those mechanistic detectors?

WHAT IS MEASURED
    BM(t) = X(t) [g/L] * Wt(t) [kg] / 1000              <- NOT Wt*V; X*Wt (see below)
    pivot = first decision step where running-max(BM) >= pivot_bm

BM is a stand-in for CER, the best-scoring online phase coordinate measured in
evaluations/ryu_mu_check (residual across-batch variance of production rate 0.52, vs biomass X
0.61, time 0.82, OUR 0.94). The simulator builds CER as (a0+a1)*q_co2*V -- active biomass x
volume -- and corr(CER, X*Wt) = 0.9910, so X*Wt recovers the same coordinate from channels that
are already in the observed state. The running max is required because raw BM peaks around 134h
and declines: 13 of 26 diagnostic batches dip back below a fixed threshold after first crossing
it, so a raw crossing test would be ambiguous late in the batch.

EXACTNESS
    This does NOT re-implement the split. It builds the decision-grid state array exactly as
    PenSimWrapper.rollout does (extract_state at k = K_WARM + d*STEPS_PER_DECISION) and calls
    DualPhaseModelLearning._biomass_pivot_step on it -- the same production code path training
    uses -- so pivot_bm_hours here is the number that run would actually split at. That method
    also carries the min_phase_steps clamp and the never-crossed fallback to pivot_step, both of
    which are reported.

Strictly offline/post-hoc: runs default-recipe batches, touches no existing file, writes only
into pivot_point_biomass/.

Usage:
    python phase_transition_diagnostic_biomass.py 1-20
    python phase_transition_diagnostic_biomass.py 1,3,5-7 --pivot_bm 1600
    python phase_transition_diagnostic_biomass.py 1-20 --plot
or:
    from evaluations.phase_transition_diagnostic_biomass import run
    results = run(training_seeds=range(1, 21))
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse, don't reimplement: get_trajectory runs the batch and pulls every channel needed, and
# the K_diff / A0 detectors are this repo's existing mechanistic pivot estimates -- carrying
# them as cross-check columns is what makes the biomass pivot interpretable rather than just a
# number. SEED_MULTIPLIER keeps the training-seed -> sim-seed mapping identical across both
# scripts, so rows here line up with rows in pivot_points_updated.csv.
from phase_transition_diagnostic_updated import (
    detect_pivot_kdiff, detect_pivot_a0, _parse_seed_spec, SEED_MULTIPLIER)
from mcpilco.pensim_wrapper import (extract_state, K_WARM, STEPS_PER_DECISION, T_SAMPLING,
                                    CONTROL_H, PIVOT_HOURS)
from mcpilco.model_learning_dual_phase import (DualPhaseModelLearning, BM_PIVOT_DEFAULT,
                                               _bm_from_states)

PIVOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point_biomass")


def _decision_states(bx):
    """The (n_decisions+1, STATE_DIM) normalised state array that PenSimWrapper.rollout would
    have handed to add_data for this batch.

    Mirrors rollout's own indexing: states[0] = extract_state(bx, K_WARM), and thereafter
    states[d] is taken at the native step satisfying (k - K_WARM - 1) % spd == spd - 1, i.e.
    k = K_WARM + d*STEPS_PER_DECISION. Reproduced here rather than re-running a rollout so this
    stays a cheap post-hoc read of an already-simulated batch."""
    n_decisions = int(CONTROL_H / T_SAMPLING)
    return np.stack([extract_state(bx, K_WARM + d * STEPS_PER_DECISION)
                     for d in range(n_decisions + 1)])


def _split_via_production_code(states, pivot_bm):
    """Call the REAL DualPhaseModelLearning._biomass_pivot_step, so this diagnostic can never
    drift from what training does.

    __init__ builds two full phase models (GPs), which this has no use for, so the instance is
    created with object.__new__ and given only the five attributes that method touches. If
    _biomass_pivot_step ever starts reading more state, this raises AttributeError immediately
    rather than silently diverging -- which is the intended failure mode."""
    stub = object.__new__(DualPhaseModelLearning)
    stub.pivot_step = int(round(PIVOT_HOURS / T_SAMPLING))   # the never-crossed fallback
    stub.pivot_bm = pivot_bm
    stub.min_phase_steps = 3
    stub._split_log = []
    step = stub._biomass_pivot_step(states)
    _, hours, crossed = stub._split_log[-1]
    return step, hours, bool(crossed)


def analyse_seed(training_seed, pivot_bm=BM_PIVOT_DEFAULT):
    """One batch, one row. Runs the simulation exactly ONCE: _run_batch keeps both the channel
    dict (for the mechanistic cross-check detectors) and the batch object (which extract_state
    needs, and which get_trajectory discards)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj, bx = _run_batch(sim_seed)

    # Native-resolution BM, for the plot and the headroom columns. The decision-grid value below
    # is what training actually splits on; these two differ only by the 5h grid.
    bm_native = traj["X"] * traj["Wt"] / 1000.0
    bm_env_native = np.maximum.accumulate(bm_native)
    hit = np.flatnonzero(bm_env_native >= pivot_bm)
    pivot_native_h = float(traj["t"][hit[0]]) if hit.size else float("nan")

    states = _decision_states(bx)
    step, hours, crossed = _split_via_production_code(states, pivot_bm)
    bm_grid = _bm_from_states(states)

    kdiff_h = detect_pivot_kdiff(traj["t"], traj["Culture_age"], traj["X"])[0]
    a0_h = detect_pivot_a0(traj["t"], traj["a0"])[0]

    return {
        "training_seed": training_seed,
        "sim_seed": sim_seed,
        # the operational answer: where THIS batch is split under --pivot_mode biomass
        "pivot_bm_hours": hours,
        "pivot_bm_step": step,
        "pivot_bm_crossed": int(crossed),
        "pivot_bm_hours_native": pivot_native_h,
        # context for judging whether pivot_bm is well placed
        "pivot_bm_threshold": pivot_bm,
        "bm_at_pivot": float(bm_grid[step]),
        "bm_max": float(bm_native.max()),
        "bm_max_over_threshold": float(bm_native.max() / pivot_bm),
        # what it replaces, and independent mechanistic estimates of the true transition
        "hardcoded_pivot_hours": float(PIVOT_HOURS),
        "pivot_kdiff_hours": kdiff_h,
        "pivot_a0_hours": a0_h,
        "_t": traj["t"], "_bm": bm_native, "_bm_env": bm_env_native,
    }


def _run_batch(sim_seed):
    """Run one default-recipe batch, returning BOTH the per-channel dict the cross-check
    detectors want and the raw batch object extract_state needs.

    get_trajectory() in phase_transition_diagnostic_updated.py returns only the former (it
    discards bx), so this repeats its four lines of setup rather than simulating twice -- a
    batch is ~7s and this script runs one per seed."""
    from utils.peni_env_setup import PenSimEnv
    from mcpilco.pensim_wrapper import PenSimWrapper
    from utils.constants import STEP_IN_HOURS
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    (_df, _raman), _y, bx = env.get_batches(random_seed=sim_seed, include_raman=False,
                                            return_batch_data=True)
    n = len(bx.X.y)
    traj = {
        "t": np.array([(i + 1) * STEP_IN_HOURS for i in range(n)]),
        "X": np.array(bx.X.y), "Wt": np.array(bx.Wt.y), "a0": np.array(bx.a0.y),
        "Culture_age": np.array(bx.Culture_age.y), "CER": np.array(bx.CER.y),
    }
    return traj, bx


_CSV_FIELDS = ["training_seed", "sim_seed", "pivot_bm_hours", "pivot_bm_step",
               "pivot_bm_crossed", "pivot_bm_hours_native", "pivot_bm_threshold",
               "bm_at_pivot", "bm_max", "bm_max_over_threshold",
               "hardcoded_pivot_hours", "pivot_kdiff_hours", "pivot_a0_hours"]


def _save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pivot_points_biomass.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in _CSV_FIELDS})
        # Summary rows, same convention as pivot_points_updated.csv's trailing mean/std/min/max.
        for label, fn in (("mean", np.nanmean), ("std", np.nanstd),
                          ("min", np.nanmin), ("max", np.nanmax)):
            row = {k: "" for k in _CSV_FIELDS}
            row["training_seed"] = label
            for col in ("pivot_bm_hours", "pivot_bm_hours_native", "bm_max",
                        "pivot_kdiff_hours", "pivot_a0_hours"):
                row[col] = float(fn([r[col] for r in results]))
            w.writerow(row)
    return path


def _plot(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hrs = np.array([r["pivot_bm_hours"] for r in results], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        ax[0].plot(r["_t"], r["_bm_env"], lw=0.9, alpha=.55, color="purple")
    ax[0].axhline(results[0]["pivot_bm_threshold"], color="k", ls="--", lw=1.4,
                  label=f"pivot_bm = {results[0]['pivot_bm_threshold']:g}")
    ax[0].set_xlabel("batch time (h)"); ax[0].set_ylabel("running-max BM = X*Wt/1000")
    ax[0].set_title("Biomass-progress signal per batch"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].hist(hrs, bins=min(20, max(5, len(hrs) // 2)), color="purple", alpha=.7)
    ax[1].axvline(PIVOT_HOURS, color="k", ls="--", lw=1.4,
                  label=f"fixed pivot_hours = {PIVOT_HOURS:g}h")
    ax[1].axvline(float(np.nanmean(hrs)), color="crimson", lw=1.6,
                  label=f"mean biomass pivot = {np.nanmean(hrs):.0f}h")
    ax[1].set_xlabel("split time (h)"); ax[1].set_ylabel("batches")
    ax[1].set_title("Where the biomass pivot lands"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.suptitle(f"Biomass training-split pivot over {len(results)} batches")
    fig.tight_layout()
    path = os.path.join(out_dir, "pivot_points_biomass.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(training_seeds=(1,), out_dir=PIVOT_DIR, pivot_bm=BM_PIVOT_DEFAULT, plot=False):
    results = []
    for s in training_seeds:
        r = analyse_seed(s, pivot_bm=pivot_bm)
        results.append(r)
        print(f"  seed {s:>3} (sim {r['sim_seed']}): split at {r['pivot_bm_hours']:6.1f}h "
              f"(step {r['pivot_bm_step']:>2}, BM {r['bm_at_pivot']:7.0f}, peak "
              f"{r['bm_max']:7.0f}){'' if r['pivot_bm_crossed'] else '   <-- NEVER CROSSED, fell back'}"
              f"   | K_diff {r['pivot_kdiff_hours']:6.1f}h  A0 {r['pivot_a0_hours']:5.1f}h")

    hrs = np.array([r["pivot_bm_hours"] for r in results], dtype=float)
    n_fb = sum(1 for r in results if not r["pivot_bm_crossed"])
    print(f"\nbiomass pivot over {len(results)} batches (pivot_bm={pivot_bm:g}): "
          f"mean {np.nanmean(hrs):.1f} +/- {np.nanstd(hrs):.1f} h, "
          f"range [{np.nanmin(hrs):.0f}, {np.nanmax(hrs):.0f}] h, "
          f"CV {np.nanstd(hrs)/np.nanmean(hrs):.3f}")
    print(f"  fixed pivot it replaces: {PIVOT_HOURS:g} h (CV 0 by construction)")
    print(f"  never crossed pivot_bm (fell back to the fixed step): {n_fb}/{len(results)}"
          + ("   <-- pivot_bm may be too high" if n_fb > 0.25 * len(results) else ""))

    if out_dir:
        print(f"Saved {_save_csv(results, out_dir)}")
        if plot:
            print(f"Saved {_plot(results, out_dir)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", type=str,
                        help="Training seed(s), same convention as "
                             "phase_transition_diagnostic_updated.py: '1', '1-20', '1,3,5-7'.")
    parser.add_argument("--pivot_bm", type=float, default=BM_PIVOT_DEFAULT,
                        help=f"biomass threshold to test (default {BM_PIVOT_DEFAULT:g}). Sweep "
                             f"this to choose the value for --pivot_mode biomass runs.")
    parser.add_argument("--out_dir", type=str, default=PIVOT_DIR)
    parser.add_argument("--plot", action="store_true", help="also write pivot_points_biomass.png")
    args = parser.parse_args()
    run(training_seeds=_parse_seed_spec(args.seeds), out_dir=args.out_dir,
        pivot_bm=args.pivot_bm, plot=args.plot)

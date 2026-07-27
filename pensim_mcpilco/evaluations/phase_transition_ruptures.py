"""Offline changepoint detection via the `ruptures` library, as a cross-check to
phase_transition_diagnostic.py's A0-peak detector.

That script's primary detector reads the simulator's privileged ground-truth A0/A1 biomass
states, which the dual-phase GP model never observes. This script instead defaults to exactly
the channels the model DOES see -- Wt, X, P, Viscosity (STATE_NAMES minus "time",
pensim_wrapper.py:41) -- so its output is something the model could plausibly act on. It also
drops the A0-peak detector's shape assumption: PELT with an RBF cost model finds where the
joint DISTRIBUTION of the chosen channel(s) changes, a more general (and more standard, in the
changepoint-detection literature) way to locate a regime shift.

Reuses get_trajectory/_parse_seed_spec/SEED_MULTIPLIER from phase_transition_diagnostic.py (same
batch-generation path, same --seed <-> sim_seed convention as experiments/02.../03...) rather
than duplicating them -- see that module's docstring for the seed-mapping rationale.

Usage:
    python phase_transition_ruptures.py 1-5 --show
    python phase_transition_ruptures.py 1 --channels a0 a1 --pen 40   # privileged ground truth
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`, `evaluations`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import ruptures as rpt

from evaluations.phase_transition_diagnostic import (
    get_trajectory, _parse_seed_spec, SEED_MULTIPLIER, PIVOT_HOURS,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point")
# Wt, X, P, Viscosity: exactly STATE_NAMES minus "time" (pensim_wrapper.py:41) -- the only
# channels the dual-phase GP model (and so any real routing decision) actually observes.
# Deliberately NOT a0/a1: those are privileged simulator-only ground truth the RL agent never
# sees, so a changepoint found only in a0/a1-space isn't something the model could ever act on.
DEFAULT_CHANNELS = ("Wt", "X", "P", "Viscosity")


def _zscore(y):
    mu, sd = float(np.mean(y)), float(np.std(y))
    return (y - mu) / sd if sd > 1e-12 else y - mu


def detect_changepoints(traj, channels=DEFAULT_CHANNELS, pen=5.0, model="rbf"):
    """Returns (times_hours, sample_indices) for each detected changepoint. Channels are
    z-scored first so ruptures isn't dominated by whichever channel has the largest raw scale
    (e.g. OUR ~1e7 vs a0 ~1-20)."""
    X = np.column_stack([_zscore(traj[c]) for c in channels])
    algo = rpt.Pelt(model=model).fit(X)
    breaks = algo.predict(pen=pen)
    idxs = [b for b in breaks if b < len(X)]  # ruptures' last entry is len(X), an end-of-series
    times = [float(traj["t"][i]) for i in idxs]  # marker, not a real changepoint -- drop it
    return times, idxs


def plot_and_report(training_seed, channels=DEFAULT_CHANNELS, pen=5.0, model="rbf",
                     out_dir=None, show=False):
    """training_seed is the value you'd pass as --seed to experiments/02.../03...; internally
    mapped to the same simulator seed those scripts use for episode 0 (see SEED_MULTIPLIER)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj = get_trajectory(sim_seed)
    t = traj["t"]
    times, idxs = detect_changepoints(traj, channels=channels, pen=pen, model=model)

    print(f"[seed {training_seed} (sim_seed={sim_seed})] ruptures changepoints "
          f"(channels={list(channels)}, pen={pen:g}): "
          f"{[f'{x:.1f}h' for x in times]} | hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h")

    fig, axes = plt.subplots(len(channels) + 1, 1, figsize=(9, 3 * (len(channels) + 1)),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, c in zip(axes, channels):
        ax.plot(t, traj[c], color="C0", label=c)
        for x in times:
            ax.axvline(x, color="crimson", lw=1.3)
        ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
        ax.set_ylabel(c)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"ruptures changepoints -- seed {training_seed} (sim_seed={sim_seed}, "
                       f"channels={list(channels)}, pen={pen:g})")

    ax = axes[-1]
    for c in channels:
        ax.plot(t, _zscore(traj[c]), label=f"{c} (z-scored)")
    for i, x in enumerate(times):
        ax.axvline(x, color="crimson", lw=1.3, label=f"changepoint ({x:.1f} h)" if i == 0 else None)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0, label=f"hardcoded PIVOT_HOURS ({PIVOT_HOURS:g} h)")
    ax.set_xlabel("Time (h)")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"ruptures_seed{training_seed}.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"training_seed": training_seed, "sim_seed": sim_seed,
            "changepoints_hours": times, "hardcoded_pivot_hours": PIVOT_HOURS}


def _save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ruptures_pivot_points.csv")
    max_cp = max((len(r["changepoints_hours"]) for r in results), default=0)
    fieldnames = (["training_seed", "sim_seed"]
                  + [f"changepoint_{i + 1}_hours" for i in range(max_cp)]
                  + ["hardcoded_pivot_hours"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {"training_seed": r["training_seed"], "sim_seed": r["sim_seed"],
                   "hardcoded_pivot_hours": r["hardcoded_pivot_hours"]}
            for i, x in enumerate(r["changepoints_hours"]):
                row[f"changepoint_{i + 1}_hours"] = x
            w.writerow(row)
    print(f"Saved {path}")
    return path


def run(training_seeds=(0,), channels=DEFAULT_CHANNELS, pen=5.0, model="rbf",
        out_dir=OUT_DIR, show=False):
    results = [plot_and_report(s, channels=channels, pen=pen, model=model,
                                out_dir=out_dir, show=show) for s in training_seeds]
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
    parser.add_argument("--channels", type=str, nargs="+", default=list(DEFAULT_CHANNELS),
                        help="Trajectory channels to feed ruptures, any of: "
                             "X,S,P,a0,a1,OUR,CER,Fs. Default: a0 a1.")
    parser.add_argument("--pen", type=float, default=5.0,
                        help="PELT penalty (higher = fewer/coarser changepoints).")
    parser.add_argument("--model", type=str, default="rbf",
                        help="ruptures cost model (rbf, l2, l1, ...).")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(training_seeds=_parse_seed_spec(args.seeds), channels=args.channels, pen=args.pen,
        model=args.model, out_dir=args.out_dir, show=args.show)

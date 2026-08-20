"""
PYTHONPATH=.. python -m evaluations.report_pivot_evolution
    [--run_dir results/biomass/rollout/seed4_1]
    [--out results/biomass/pivot_rollout_evolution.png]
    [--trials 1,2,3,...]  [--n_particles N]

Extra report figure (separate image from report_pivot_plots.py): how the on_each_rollout
per-particle pivot-crossing distribution (eval_multi_phase_lib.check_rollout_pivot_distribution,
C0g) evolves across TRAINING TRIALS, not just at the single final trial C0g normally reports.

Reuses check_rollout_pivot_distribution UNCHANGED -- same production particle-rollout
crossing-time computation used during policy optimisation -- called once per trial, writing its
own files to a throwaway scratch directory that's deleted afterwards; only the returned t_cross
array is kept. The cached final-trial C0g_rollout_pivot_distribution.{csv,png,txt} in the real
run directory is never touched.

This is NOT free like report_pivot_plots.py's other panels: it reconstructs the GP at every
requested trial (default: all of them) and rolls out n_particles through it each time, so it's a
separate script/image rather than folded into that one.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

import evaluations.eval_multi_phase_lib as lib
from mcpilco.config_dual_phase_baseline import get_config as dual_phase_baseline_get_config

COLOR = "C2"  # matches report_pivot_plots.py's "Dynamical pivot for rollouts" colour
FIXED_COLOR = "C0"  # matches report_pivot_plots.py's "Fixed pivot" colour


def evolution(run, trials, n_particles=None, get_config_fn=None):
    rows = []
    scratch = tempfile.mkdtemp(prefix="pivot_evolution_scratch_")
    try:
        for k in trials:
            gp_agent, gp_idx = lib.reconstruct_gp_agent(run, idx=k, get_config_fn=get_config_fn)
            t_cross = lib.check_rollout_pivot_distribution(
                gp_agent, gp_idx, run, out_dir=scratch, n_particles=n_particles)
            if t_cross is None:
                continue
            ok = np.isfinite(t_cross)
            print(f"trial {k}: {int(ok.sum())}/{len(t_cross)} particles crossed, "
                 f"median={np.median(t_cross[ok]):.1f}h" if ok.any() else f"trial {k}: none crossed")
            rows.append({"trial": k, "t_cross": t_cross[ok]})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rows


def plot_evolution(rows, out_path, pivot_hours=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    trials = [r["trial"] for r in rows]
    med = [np.median(r["t_cross"]) for r in rows]
    lo = [np.percentile(r["t_cross"], 5) for r in rows]
    hi = [np.percentile(r["t_cross"], 95) for r in rows]
    ax.fill_between(trials, lo, hi, color=COLOR, alpha=.25, label="5-95th pct")
    ax.plot(trials, med, "-o", color=COLOR, lw=2.5, ms=6, label="median")
    if pivot_hours is not None:
        ax.axhline(pivot_hours, color=FIXED_COLOR, ls="--", lw=2, label="Fixed pivot")
    ax.set_xlabel("training trial (episode)")
    ax.set_ylabel("pivot crossing time (h)")
    ax.set_title("Rollout pivot-crossing distribution over training")
    ax.grid(alpha=.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main(run_dir, out_path, trials=None, n_particles=None, get_config_fn=dual_phase_baseline_get_config):
    run = lib.load_run(run_dir, get_config_fn=get_config_fn)
    if not run.on_each_rollout:
        raise ValueError(f"{run_dir}: on_each_rollout=False -- this run has no per-rollout "
                         f"pivot spread to track evolution of")
    if trials is None:
        # Saved GP trial numbers are NOT necessarily 1..n_trials_in_log (exploration episodes
        # shift the numbering) -- read the actual parameters_gp_<i> keys, same as
        # eval_multi_phase_lib._resolve_trial does for a single trial.
        trials = sorted(int(k.split("_")[-1]) for k in run.log if k.startswith("parameters_gp_"))
    rows = evolution(run, trials, n_particles=n_particles, get_config_fn=get_config_fn)
    plot_evolution(rows, out_path, pivot_hours=run.pivot_hours)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="../results/biomass/rollout/seed4_1")
    p.add_argument("--out", default="../results/biomass/pivot_rollout_evolution.png")
    p.add_argument("--trials", type=str, default=None,
                   help="comma-separated trial numbers (default: every trial in the log)")
    p.add_argument("--n_particles", type=int, default=None,
                   help="default: this run's own training particle count")
    args = p.parse_args()
    trial_list = [int(x) for x in args.trials.split(",")] if args.trials else None
    main(args.run_dir, args.out, trials=trial_list, n_particles=args.n_particles)

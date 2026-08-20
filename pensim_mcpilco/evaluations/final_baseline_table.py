"""The one summary table: recipe vs BO vs MC-PILCO, same budget, same held-out batches.

Every arm is the artifact the method would hand you after 5 exploration/random-design batches +
10 more (15 simulated batches total), evaluated on the SAME held-out block (default
700000-700009) against the SAME recipe reference:

  recipe    the untouched default profile
  BO        the best-scoring FEASIBLE schedule among its first 15 evaluations, ranked on its own
            training batch only (never on held-out data) -- i.e. what gp_minimize would return at
            that budget, NOT the best of the full 100-call search
  MC-PILCO  each run's trial-10 policy

`collapsed` counts (run x held-out batch) pairs that breached the operating envelope --
Viscosity > VISC_MAX or Wt > WT_OVERFLOW, the same rule as eval_utils.feasibility_gated_yield_kg.
Yields are RAW: a collapsed batch keeps the yield it actually produced and is flagged in its own
column rather than being zeroed.

Usage
    PYTHONPATH=.. python evaluations/final_baseline_table.py
    PYTHONPATH=.. python evaluations/final_baseline_table.py --trial 10 --n_eval_seeds 10
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))

from experiments.eval_utils import constraint_diagnostics
from evaluations.bo_baseline_holdout import Segmentation, simulate
from evaluations import test_seed_policies as tsp

# MC-PILCO arms: label -> (run dirs relative to results/, setup name for --setup resolution)
MCPILCO_ARMS = [
    ("MC-PILCO single-phase (no time)",
     ["full/ConcCost/single-phase/No_time/seed4_0",
      "full/ConcCost/single-phase/No_time/seed5_0",
      "full/ConcCost/single-phase/No_time/seed6_0"], "single_phase_baseline"),
    ("MC-PILCO single-phase (added time)",
     ["full/ConcCost/single-phase/Added_time/seed4_1",
      "full/ConcCost/single-phase/Added_time/seed5_1",
      "full/ConcCost/single-phase/Added_time/seed6_1"], "single_phase_baseline_time"),
]
BO_SEEDS = (4000, 5000, 6000)


def _collapsed(mon):
    d = constraint_diagnostics(mon)
    return bool(d["wt_overflow"] or d["visc_exceed"])


def bo_arm(bo_dir, n_evals, eval_seeds):
    """BO's shipped schedule after `n_evals` evaluations, per training seed, on the held-out block."""
    import json
    seg = None
    per_run, collapsed = [], 0
    for s in BO_SEEDS:
        lg = pd.read_csv(Path(bo_dir) / f"bo_log_seed{s}.csv")
        meta = json.loads((Path(bo_dir) / f"meta_seed{s}.json").read_text())
        seg = seg or Segmentation(meta.get("segment_hours", 25.0))
        sub = lg.iloc[:n_evals]
        best = sub.loc[sub["yield_gated"].idxmax()]
        fac = seg.unpack(best[seg.factor_columns()].values)
        print(f"  BO seed {s}: shipping call {int(best['call'])} of first {n_evals} "
              f"({best['yield_gated']:.0f} kg in-sample)", flush=True)
        rs = [simulate(seg, fac, h) for h in eval_seeds]
        per_run.append(float(np.mean([r["yield_raw"] for r in rs])))
        collapsed += int(sum(r["collapsed"] for r in rs))
    return per_run, collapsed, seg


def recipe_arm(seg, eval_seeds):
    rs = [simulate(seg, {}, h) for h in eval_seeds]
    ys = [r["yield_raw"] for r in rs]
    return float(np.mean(ys)), int(sum(r["collapsed"] for r in rs)), ys


def mcpilco_arm(run_dirs, setup, trial, eval_seeds):
    """Each run's trial-`trial` policy on the held-out block."""
    per_run, collapsed = [], 0
    for rd in run_dirs:
        path = str(Path(_ROOT) / "results" / rd)
        lib, get_cfg, results_root, _ = tsp._resolve(path, setup=setup)
        run = lib.load_run(path, get_config_fn=get_cfg, results_root=results_root)
        _, eval_wrapper, _, _, _ = lib.build_policy_agent(run)
        k = min(trial, run.n_trials_in_log)
        pol = lib.load_stage_policy(run, k)
        mons = [lib.run_arm(eval_wrapper, h, policy=pol, pid_baseline=False) for h in eval_seeds]
        ys = [lib.yield_kg(m) for m in mons]
        per_run.append(float(np.mean(ys)))
        collapsed += int(sum(_collapsed(m) for m in mons))
        print(f"  {Path(rd).name}: trial {k} -> {np.mean(ys):.1f} kg, "
              f"{sum(_collapsed(m) for m in mons)} collapsed", flush=True)
    return per_run, collapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=int, default=10,
                   help="MC-PILCO trial / BO guided-evaluation count to report (default 10)")
    p.add_argument("--n_random", type=int, default=5,
                   help="BO random-design points, = MC-PILCO's num_explorations (default 5)")
    p.add_argument("--bo_budgets", type=int, nargs="+", default=None,
                   help="BO evaluation budgets to report, one row each (default: the "
                        "budget-matched n_random+trial, plus 100 = the full search as a ceiling)")
    p.add_argument("--n_eval_seeds", type=int, default=10)
    p.add_argument("--eval_base", type=int, default=700000)
    p.add_argument("--bo_dir", type=str, default=str(Path(_ROOT) / "results" / "bo_baseline"))
    p.add_argument("--out", type=str,
                   default=str(Path(_ROOT) / "results" / "bo_baseline" / "final_table.csv"))
    args = p.parse_args()

    eval_seeds = [args.eval_base + i for i in range(args.n_eval_seeds)]
    n_evals = args.n_random + args.trial
    if args.bo_budgets is None:
        args.bo_budgets = sorted({n_evals, 100})
    n_pairs = 3 * len(eval_seeds)
    print(f"held-out block {eval_seeds[0]}-{eval_seeds[-1]} | budget = {args.n_random} + "
          f"{args.trial} = {n_evals} batches\n")

    # BO at each requested budget. The budget-matched one (n_random + trial) is the fair
    # comparison; the larger ones answer "does BO just need more batches?" and are a ceiling, not
    # a like-for-like row -- label them as such in the write-up.
    bo_rows, seg = [], None
    for b in args.bo_budgets:
        print(f"BO @ {b} evaluations:")
        means, coll, seg = bo_arm(args.bo_dir, b, eval_seeds)
        bo_rows.append((b, means, coll))

    print("recipe:")
    rec_mean, rec_coll, _ = recipe_arm(seg, eval_seeds)
    print(f"  {rec_mean:.1f} kg, {rec_coll} collapsed of {len(eval_seeds)}")

    rows = [{
        "method": "Recipe (no tuning)", "batches_used": 0,
        "mean_yield": rec_mean, "sd_across_runs": float("nan"),
        "delta_vs_recipe": 0.0,
        "collapsed": f"{rec_coll}/{len(eval_seeds)}",
    }]
    for b, means, coll in bo_rows:
        rows.append({
            "method": f"BO (best of first {b} evaluations)", "batches_used": b,
            "mean_yield": float(np.mean(means)), "sd_across_runs": float(np.std(means, ddof=1)),
            "delta_vs_recipe": float(np.mean(means) - rec_mean),
            "collapsed": f"{coll}/{n_pairs}",
        })

    for label, dirs, setup in MCPILCO_ARMS:
        print(f"{label}:")
        means, coll = mcpilco_arm(dirs, setup, args.trial, eval_seeds)
        rows.append({
            "method": f"{label}, trial {args.trial}", "batches_used": n_evals,
            "mean_yield": float(np.mean(means)), "sd_across_runs": float(np.std(means, ddof=1)),
            "delta_vs_recipe": float(np.mean(means) - rec_mean),
            "collapsed": f"{coll}/{n_pairs}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n=== {args.out} ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

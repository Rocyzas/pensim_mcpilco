"""
PYTHONPATH=.. python -m evaluations.episode_holdout_two_group \
    --group_a "seed 4/5/6=results/full/ConcCost/single-phase/No_time/seed4_0,results/full/ConcCost/single-phase/No_time/seed5_0,results/full/ConcCost/single-phase/No_time/seed6_0" \
    --group_b "seed 4/5/6 (time)=results/full/ConcCost/single-phase/Added_time/seed4_1,results/full/ConcCost/single-phase/Added_time/seed5_1,results/full/ConcCost/single-phase/Added_time/seed6_1" \
    --setup single_phase_baseline --title "No_time vs Added_time, ConcCost single-phase"

Two-group variant of episode_holdout_multiseed.py: exactly group A vs group B, plotted with a
custom title -- for when two named groups on one plot (with an explicit title) is clearer to
read than the general N-group overlay. Same shaded +-std band style as the parent script.

Reuses episode_holdout_multiseed.py's data loading/aggregation UNCHANGED (parse_group_arg,
load_group, aggregate, default_out_dir) -- the only new code here is the plot itself. That keeps
this script from silently drifting on what "average across training seeds" means; only the
picture changes.

Writes episode_holdout_two_group.{csv,png} -- deliberately NOT the same filenames
episode_holdout_multiseed.py uses, so running both against the same --out_dir never overwrites
either one's output.
"""
import argparse
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluations.episode_holdout_multiseed as ehm
import evaluations.test_seed_policies as tsp


def _panel_band(ax, summary, groups, value_col, std_col, ylabel, title, show_variance=True):
    """Mean line + translucent +-std band per group, in the group's own colour."""
    for i, label in enumerate(groups):
        s = summary[summary["group"] == label].sort_values("episode")
        colour = f"C{i}"
        ax.plot(s["episode"], s[value_col], "-o", color=colour, lw=2, ms=4, zorder=4,
                label=f"{label} (n={int(s['n_runs'].max())} runs)")
        if show_variance:
            band = s[std_col].to_numpy()
            if np.isfinite(band).any():
                ax.fill_between(s["episode"], s[value_col] - band, s[value_col] + band,
                                color=colour, alpha=0.18, lw=0, zorder=3)
        partial = s[s["n_runs"] < s["n_runs"].max()]
        if not partial.empty:
            ax.plot(partial["episode"], partial[value_col], "x", color=colour, ms=9, mew=2,
                    zorder=5, label=f"{label}: fewer runs at this episode")
    ax.set_xlabel("training episode (policy trial)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot(summary, groups, title, out_path, show_variance=True):
    """Paired delta vs recipe -- see episode_holdout_multiseed.plot's docstring for why absolute
    yield isn't what gets plotted (held-out batch difficulty, not policy quality, dominates it)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _panel_band(ax, summary, groups, "mean_delta_vs_recipe", "std_delta_across_runs",
               "yield - recipe (kg)", title, show_variance=show_variance)
    ax.axhline(0.0, color="crimson", ls="--", lw=1.4, zorder=1, label="recipe")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main(group_a, group_b, title=None, setup=None, setup_a=None, setup_b=None, n_eval_seeds=10,
        eval_base=700000, refresh=False, truncate=False, out_dir=None, show_variance=True):
    """setup_a/setup_b each fall back to the shared `setup` when not given, so group A and group
    B can be genuinely different setups (e.g. single_phase_baseline vs single_phase_baseline_time)
    -- a single shared setup can't be right for both if their architectures actually differ, and
    would silently reconstruct the wrong GP (or crash on an active_dims shape mismatch, see
    eval_single_phase_lib.reconstruct_gp_agent's docstring) for whichever group it doesn't match,
    on any run that isn't already covered by a cached episode_holdout_curve.csv."""
    groups = [group_a, group_b]
    labels = [lb for lb, _ in groups]
    setups = [setup_a if setup_a is not None else setup,
             setup_b if setup_b is not None else setup]
    if title is None:
        title = f"Held-out yield vs recipe: {labels[0]} vs {labels[1]}"

    summaries = []
    for (label, run_paths), grp_setup in zip(groups, setups):
        raw = ehm.load_group(label, run_paths, setup=grp_setup, n_eval_seeds=n_eval_seeds,
                             eval_base=eval_base, refresh=refresh, truncate=truncate)
        summaries.append(ehm.aggregate(raw, label))

    summary = pd.concat(summaries, ignore_index=True)

    out_dir = ehm.default_out_dir(groups) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episode_holdout_two_group.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")
    plot(summary, labels, title, out_dir / "episode_holdout_two_group.png",
        show_variance=show_variance)

    print("\n--- final-episode comparison ---")
    finals = {}
    for label in labels:
        s = summary[summary["group"] == label].sort_values("episode")
        finals[label] = (float(s["mean_yield"].iloc[-1]), float(s["std_across_runs"].iloc[-1]))
    (ma, sa), (mb, sb) = finals[labels[0]], finals[labels[1]]
    gap = ma - mb
    pooled = float(np.sqrt(np.nansum([sa ** 2, sb ** 2])))
    verdict = "larger than" if abs(gap) > pooled else "WITHIN"
    print(f"{labels[0]} vs {labels[1]}: gap {gap:+.1f} kg, combined across-run SD {pooled:.1f} kg "
         f"-> {verdict} training-seed spread")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Two-group held-out yield comparison with a shaded +-std band per group.",
        epilog="example: --group_a \"A=path1,path2,path3\" --group_b \"B=path4,path5,path6\" "
              "--setup single_phase_baseline --title \"my comparison\"")
    p.add_argument("--group_a", type=str, required=True, metavar="LABEL=P1,P2,...",
                   help="group A: a label and its run folders, e.g. 'no_time=path1,path2,path3'")
    p.add_argument("--group_b", type=str, required=True, metavar="LABEL=P1,P2,...",
                   help="group B: a label and its run folders, same format as --group_a")
    p.add_argument("--title", type=str, default=None,
                   help="plot title (default: auto-generated from the two group labels)")
    p.add_argument("--setup", choices=list(tsp.SETUPS), default=None,
                   help="which eval library/config BOTH groups' runs use. Required whenever a "
                        "run folder's PARENT directory is not itself a setup name -- which is "
                        "the case for results/full/... paths, where the parent is e.g. "
                        "'No_time'. Ignored (per run) whenever that run's episode_holdout_curve"
                        ".csv is already cached. Overridden per group by --setup_a/--setup_b.")
    p.add_argument("--setup_a", choices=list(tsp.SETUPS), default=None,
                   help="setup for group A only, overriding --setup -- use this (and/or "
                        "--setup_b) when the two groups are genuinely different architectures, "
                        "e.g. group A is single_phase_baseline and group B is "
                        "single_phase_baseline_time")
    p.add_argument("--setup_b", choices=list(tsp.SETUPS), default=None,
                   help="setup for group B only, overriding --setup (see --setup_a)")
    p.add_argument("--n_eval_seeds", type=int, default=10,
                   help="held-out seeds, used only when a curve has to be computed")
    p.add_argument("--eval_base", type=int, default=700000, help="first held-out seed")
    p.add_argument("--refresh", action="store_true",
                   help="recompute every run's curve instead of reading its cached CSV")
    p.add_argument("--truncate", action="store_true",
                   help="restrict each group to the episode range every run in it covers")
    p.add_argument("--out_dir", type=str, default=None,
                   help="where to write the CSV/PNG (default: the runs' common parent directory)")
    p.add_argument("--no_variance", action="store_true",
                   help="hide the shaded +-std band per group (mean lines only)")
    args = p.parse_args()

    group_a = ehm.parse_group_arg(args.group_a)
    group_b = ehm.parse_group_arg(args.group_b)

    main(group_a, group_b, title=args.title, setup=args.setup, setup_a=args.setup_a,
        setup_b=args.setup_b, n_eval_seeds=args.n_eval_seeds, eval_base=args.eval_base,
        refresh=args.refresh, truncate=args.truncate, out_dir=args.out_dir,
        show_variance=not args.no_variance)

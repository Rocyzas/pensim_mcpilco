"""
PYTHONPATH=.. python -m evaluations.episode_holdout_multiseed \
    results/full/ConcCost/single-phase/No_time/seed4_0 \
    results/full/ConcCost/single-phase/No_time/seed5_0 \
    results/full/ConcCost/single-phase/No_time/seed6_0 \
    --setup single_phase_baseline --label "single-phase, No_time"

Per-episode held-out yield averaged across SEVERAL TRAINING RUNS.

test_seed_policies.py already gives the per-episode held-out curve for ONE run, averaged over a
block of held-out simulator seeds. Nothing averaged the other axis: several training runs
(seed4_x, seed5_x, seed6_x) of the same configuration collapsed into one mean +- variance curve.
analyze_single_phase.py and imagined_vs_real.py do take several runs, but draw one line each;
every "seed-averaged" helper in the eval libs (plot_seed_avg_vars) averages EVALUATION seeds
inside a single run. So a config-vs-config comparison rested on n=1 training seed and could not
be told apart from training-seed noise.

This script does not re-evaluate anything it doesn't have to. test_seed_policies.py writes
<run_dir>/episode_holdout_curve.csv, and those already exist for most runs -- so the common case
reads cached CSVs and touches no simulator. When one is missing (or under --refresh) it calls
test_seed_policies.main for that run, which computes and writes the CSV, and reads it back. That
is why test_seed_policies.py needs no changes: its evaluation path is reused exactly, not
reimplemented.

Two variances are reported rather than one, because they answer different questions:
`std_across_runs` is spread between TRAINING seeds -- the one that says whether a gap between two
configurations is real -- while `mean_within_run_std` is spread between EVALUATION seeds within a
run. Collapsing them into a single number would make a config comparison look more (or less)
significant than it is.
"""
import argparse
import os
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if _os.path.dirname(_ROOT) not in _sys.path:
    _sys.path.insert(0, _os.path.dirname(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluations.test_seed_policies as tsp

CURVE_CSV = "episode_holdout_curve.csv"


def seed_columns(df):
    """The `yield_seed_<s>` columns, sorted by seed number. Their SET identifies which held-out
    block a run was evaluated on -- which is not uniform across the results tree (results/full and
    results/biomass runs carry 10 seeds, results/dual_phase_baseline runs carry 5), hence the
    guard in load_group."""
    cols = [c for c in df.columns if c.startswith("yield_seed_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def load_run_curve(run_path, setup=None, n_eval_seeds=10, eval_base=700000, refresh=False):
    """One run's per-episode curve, from its cached CSV when possible.

    On a miss, test_seed_policies.main does the real work (loads every episode's policy and rolls
    it on the held-out block) and writes the CSV into the run's own folder; we then read it back.
    Deliberately not reimplemented here -- the per-episode policy loop has enough setup/config
    resolution in it that a second copy would drift."""
    run_dir = Path(run_path)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory")
    csv_path = run_dir / CURVE_CSV
    if refresh or not csv_path.exists():
        why = "--refresh" if refresh else "no cached curve"
        print(f"[compute] {run_dir} ({why}) -> running test_seed_policies "
              f"(this rolls the simulator and will take a while)")
        tsp.main(run_id=str(run_dir), n_eval_seeds=n_eval_seeds, eval_base=eval_base, setup=setup)
    else:
        print(f"[cached]  {run_dir}/{CURVE_CSV}")
    df = pd.read_csv(csv_path)
    df["run"] = run_dir.name
    df["run_dir"] = str(run_dir)
    return df


def load_group(label, run_paths, setup=None, n_eval_seeds=10, eval_base=700000, refresh=False,
               truncate=False):
    """Every run in one group, checked for comparability before anything is averaged."""
    print(f"\n=== group '{label}' ({len(run_paths)} runs) ===")
    dfs = [load_run_curve(p, setup=setup, n_eval_seeds=n_eval_seeds, eval_base=eval_base,
                          refresh=refresh)
           for p in run_paths]

    # Averaging runs scored on different held-out blocks would compare different batch
    # difficulties while looking like a like-for-like mean, so this raises rather than warns.
    blocks = {tuple(seed_columns(d)) for d in dfs}
    if len(blocks) > 1:
        detail = "\n".join(
            f"  {d['run_dir'].iloc[0]}: {len(seed_columns(d))} seeds "
            f"[{', '.join(c.rsplit('_', 1)[1] for c in seed_columns(d))}]" for d in dfs)
        raise ValueError(
            f"group '{label}' mixes different held-out seed blocks, which cannot be averaged:\n"
            f"{detail}\nRe-run with --refresh --n_eval_seeds N to put every run on one block.")

    if truncate:
        last = min(int(d["episode"].max()) for d in dfs)
        dfs = [d[d["episode"] <= last] for d in dfs]
        print(f"  --truncate: episodes 1..{last} (shortest run in the group)")

    df = pd.concat(dfs, ignore_index=True)
    df["group"] = label
    return df


def aggregate(df, label):
    """Per-episode statistics across the group's runs."""
    scols = seed_columns(df)
    rows = []
    for ep, g in df.groupby("episode", sort=True):
        per_run_mean = g["mean_yield"].to_numpy()
        per_run_delta = g["mean_delta_vs_recipe"].to_numpy()
        # Every (run x eval seed) yield, so pooled spread is exact rather than reconstructed
        # from the per-run summaries.
        pooled = g[scols].to_numpy().ravel()
        n_runs = len(g)
        # Recipe baseline isn't stored, but delta = yield - baseline, so it inverts exactly. It
        # depends only on the held-out block, so all runs in a group must agree -- a disagreement
        # means two different evaluation setups slipped past the seed-block guard.
        baselines = per_run_mean - per_run_delta
        if n_runs > 1 and not np.allclose(baselines, baselines[0], rtol=0, atol=1e-6):
            raise ValueError(
                f"group '{label}' episode {ep}: runs disagree on the recipe baseline "
                f"({np.min(baselines):.3f}..{np.max(baselines):.3f} kg) despite sharing a seed "
                f"block -- they were not evaluated against the same reference.")
        rows.append({
            "group": label, "episode": int(ep), "n_runs": n_runs, "n_eval_seeds": len(scols),
            "mean_yield": float(per_run_mean.mean()),
            # ddof=1: these runs are a sample of training seeds, not the population. Undefined
            # for a single run, which is correct -- one seed carries no spread information.
            "std_across_runs": float(per_run_mean.std(ddof=1)) if n_runs > 1 else float("nan"),
            "sem_across_runs": (float(per_run_mean.std(ddof=1) / np.sqrt(n_runs))
                                if n_runs > 1 else float("nan")),
            "mean_within_run_std": float(g["std_yield"].mean()),
            "pooled_std": float(pooled.std(ddof=1)),
            "mean_delta_vs_recipe": float(per_run_delta.mean()),
            "std_delta_across_runs": (float(per_run_delta.std(ddof=1)) if n_runs > 1
                                      else float("nan")),
            "worst_delta": float(g["worst_delta"].min()),
            "recipe_baseline": float(baselines[0]),
        })
    return pd.DataFrame(rows)


def _panel(ax, summary, raw, groups, value_col, std_col, per_run_col, ylabel, title,
          show_variance=True):
    for i, label in enumerate(groups):
        s = summary[summary["group"] == label].sort_values("episode")
        r = raw[raw["group"] == label]
        colour = f"C{i}"
        if show_variance:
            # Individual runs stay visible: a 3-run band is a summary of so few points that
            # hiding them would obscure whether the spread is symmetric or driven by one
            # outlier seed. Gated on show_variance too -- both are spread indicators, so
            # --no_variance means "mean lines only", not "just the band off".
            for _, run_df in r.groupby("run"):
                run_df = run_df.sort_values("episode")
                ax.plot(run_df["episode"], run_df[per_run_col], color=colour, lw=0.8,
                        alpha=0.35, zorder=2)
        ax.plot(s["episode"], s[value_col], "-o", color=colour, lw=2, ms=4, zorder=4,
                label=f"{label} (n={int(s['n_runs'].max())} runs)")
        if show_variance:
            band = s[std_col].to_numpy()
            if np.isfinite(band).any():
                ax.fill_between(s["episode"], s[value_col] - band, s[value_col] + band,
                                color=colour, alpha=0.2, lw=0, zorder=3)
        # Episodes where some run in the group stopped early: the mean there is over fewer runs
        # and is not comparable to the rest of the curve.
        partial = s[s["n_runs"] < s["n_runs"].max()]
        if not partial.empty:
            ax.plot(partial["episode"], partial[value_col], "x", color=colour, ms=9, mew=2,
                    zorder=5,
                    label=f"{label}: fewer runs at this episode")
    ax.set_xlabel("training episode (policy trial)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot(summary, raw, groups, out_path, title=None, show_variance=True):
    """Plots the PAIRED DELTA vs recipe, not absolute yield.

    Absolute kg is dominated by how hard the held-out batches happen to be, which is a property
    of the evaluation seeds rather than of the policy -- so two configurations can only be
    compared after that common term is subtracted. test_seed_policies computes the delta per
    evaluation seed against the recipe run on that SAME seed, so the subtraction is paired, not a
    difference of two independent averages. The absolute yield and the recipe baseline are still
    in the CSV for anyone who needs the raw numbers."""
    if title is None:
        title = "Held-out yield vs recipe"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _panel(ax, summary, raw, groups, "mean_delta_vs_recipe", "std_delta_across_runs",
           "mean_delta_vs_recipe", "yield - recipe (kg)", title, show_variance=show_variance)
    ax.axhline(0.0, color="crimson", ls="--", lw=1.4, zorder=1, label="recipe")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def parse_group_arg(spec):
    """'label=path1,path2,path3' -> (label, [paths])."""
    if "=" not in spec:
        raise ValueError(f"--group must be 'label=path1,path2,...', got {spec!r}")
    label, paths = spec.split("=", 1)
    runs = [p for p in paths.split(",") if p]
    if not runs:
        raise ValueError(f"--group {spec!r} lists no run paths")
    return label.strip(), runs


def default_out_dir(groups):
    """Deepest directory containing every run in every group -- for one config that is the config
    folder, for a cross-config comparison it rises to their common ancestor."""
    all_paths = [str(Path(p).resolve()) for _, runs in groups for p in runs]
    return Path(os.path.commonpath(all_paths)) if len(all_paths) > 1 else Path(all_paths[0]).parent


def main(groups, setup=None, n_eval_seeds=10, eval_base=700000, refresh=False, truncate=False,
         out_dir=None, title=None, show_variance=True):
    raw_frames, summaries = [], []
    for label, run_paths in groups:
        raw = load_group(label, run_paths, setup=setup, n_eval_seeds=n_eval_seeds,
                         eval_base=eval_base, refresh=refresh, truncate=truncate)
        raw_frames.append(raw)
        summaries.append(aggregate(raw, label))

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    labels = [lb for lb, _ in groups]

    out_dir = default_out_dir(groups) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episode_holdout_multiseed.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")
    plot(summary, raw_all, labels, out_dir / "episode_holdout_multiseed.png", title=title,
        show_variance=show_variance)

    for label in labels:
        s = summary[summary["group"] == label].sort_values("episode")
        print(f"\n--- {label} ---")
        print(f"{'ep':>3} {'mean yield':>11} {'+-SD(runs)':>11} {'+-SD(eval)':>11} "
              f"{'delta vs recipe':>16} {'n':>3}")
        for _, r in s.iterrows():
            print(f"{int(r['episode']):>3} {r['mean_yield']:>11.1f} {r['std_across_runs']:>11.1f} "
                  f"{r['mean_within_run_std']:>11.1f} {r['mean_delta_vs_recipe']:>+16.1f} "
                  f"{int(r['n_runs']):>3}")
        best = s.loc[s["mean_yield"].idxmax()]
        print(f"peak mean yield {best['mean_yield']:.1f} kg @ episode {int(best['episode'])} "
              f"of {int(s['episode'].max())}")

    # The comparison the script exists for: is a gap between configurations larger than the
    # training-seed spread it has to clear to mean anything?
    if len(labels) > 1:
        print("\n--- final-episode comparison ---")
        finals = {}
        for label in labels:
            s = summary[summary["group"] == label].sort_values("episode")
            finals[label] = (float(s["mean_yield"].iloc[-1]), float(s["std_across_runs"].iloc[-1]))
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                (ma, sa), (mb, sb) = finals[a], finals[b]
                gap = ma - mb
                pooled = float(np.sqrt(np.nansum([sa ** 2, sb ** 2])))
                verdict = ("larger than" if abs(gap) > pooled else "WITHIN")
                print(f"{a} vs {b}: gap {gap:+.1f} kg, combined across-run SD {pooled:.1f} kg "
                      f"-> {verdict} training-seed spread")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Per-episode held-out yield averaged across training runs.",
        epilog="example: results/full/ConcCost/single-phase/No_time/seed{4_0,5_0,6_0} "
               "--setup single_phase_baseline")
    p.add_argument("runs", nargs="*", default=[], metavar="RUN_DIR",
                   help="run folders forming one group (see --label); or use --group instead")
    p.add_argument("--label", type=str, default="runs",
                   help="name for the group given as positional run folders")
    p.add_argument("--group", action="append", default=[], metavar="LABEL=P1,P2,...",
                   help="named group of run folders; repeat to overlay several configurations")
    p.add_argument("--setup", choices=list(tsp.SETUPS), default=None,
                   help="which eval library/config the runs use. Required whenever a run folder's "
                        "PARENT directory is not itself a setup name -- which is the case for "
                        "results/full/... paths, where the parent is e.g. 'No_time'")
    p.add_argument("--n_eval_seeds", type=int, default=10,
                   help="held-out seeds, used only when a curve has to be computed")
    p.add_argument("--eval_base", type=int, default=700000, help="first held-out seed")
    p.add_argument("--refresh", action="store_true",
                   help="recompute every run's curve instead of reading its cached CSV")
    p.add_argument("--truncate", action="store_true",
                   help="restrict each group to the episode range every run in it covers")
    p.add_argument("--out_dir", type=str, default=None,
                   help="where to write the CSV/PNG (default: the runs' common parent directory)")
    p.add_argument("--title", type=str, default=None,
                   help="plot title (default: 'Held-out yield vs recipe')")
    p.add_argument("--no_variance", action="store_true",
                   help="hide the shaded +-std band per group (mean lines only)")
    args = p.parse_args()

    groups = [parse_group_arg(g) for g in args.group]
    if args.runs:
        groups.insert(0, (args.label, args.runs))
    if not groups:
        p.error("give run folders as positional arguments, or use --group LABEL=P1,P2,...")

    main(groups, setup=args.setup, n_eval_seeds=args.n_eval_seeds, eval_base=args.eval_base,
         refresh=args.refresh, truncate=args.truncate, out_dir=args.out_dir, title=args.title,
         show_variance=not args.no_variance)

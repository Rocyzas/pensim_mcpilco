"""
 PYTHONPATH=.. python -m experiments.01c_bo_baseline_recipe_scaling --seed 4000 --n_calls 100
 PYTHONPATH=.. python -m experiments.01c_bo_baseline_recipe_scaling --seed 4000 --segment_hours 25

Scripted version of 01_bo_baseline_adapted_action.ipynb: Bayesian optimisation over multiplicative
scale factors applied to the default Fs recipe profile, writing its CSVs and PNGs to disk instead of
rendering them into notebook cells.

The search space is set by --segment_hours:

    --segment_hours 25    the notebook's behaviour -- the batch is cut into ceil(230/25) = 10
                          segments and BO searches ONE FACTOR PER SEGMENT, so the recipe can be
                          pushed up early and pulled back late.
    (omitted)             OPEN LOOP: a single factor per channel, held for the whole batch. One
                          search dimension per channel. This is the 01_bo_baseline_simple.ipynb
                          behaviour and the default here.

NOTE ON "OPEN LOOP". Both modes are open-loop in the control-theory sense: the feed profile is fixed
before the batch starts and never reacts to the process. The flag chooses between a CONSTANT and a
PIECEWISE-CONSTANT profile, not between open and closed loop. For a genuinely closed-loop BO arm --
a policy that reads the process state at every decision step, which is what makes BO comparable to
MC-PILCO on equal terms -- see 01b_bo_baseline_closed_loop_policy.ipynb.

SEEDS. --seed is the SIMULATOR seed of the batch BO searches on, exactly as in the notebook, and it
doubles as gp_minimize's random_state unless --bo_seed overrides it. It is NOT an MC-PILCO --seed:
a run launched as `--seed 4` sets seed_offset = 4*1000 and trains on simulator seeds 4000, 4001, ...
So `--seed 4000` here searches on the same batch realisation MC-PILCO's `--seed 4` run started from.

The held-out block (--eval_base, default 700000) is the SAME five batches every MC-PILCO run is
evaluated on -- eval_{single,multi}_phase_lib.eval_held_out hardcodes eval_base=700000, independent
of the run's --seed (verified: the yield_recipe column is identical across 167 logged runs spanning
--seed 3..9). That is what makes A1_held_out.csv's yield_rl and this script's held-out numbers
directly comparable, whichever MC-PILCO seed you compare against.
"""
import argparse
import datetime
import math
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: this script only ever writes PNGs, never shows them
import matplotlib.pyplot as plt

from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv
from utils.ode_patch import patch_fastodeint
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from skopt import gp_minimize
from skopt.space import Real

patch_fastodeint()

CONC_COL = "Penicillin Concentration"
PHYSICAL_P_MAX = 40.0          # concentrations above this are unphysical -> the batch scores 0
BATCH_HOURS = 230.0

_RESULTS_ROOT = Path(_ROOT) / "results" / "bo_recipe_scaling"

DEFAULTS = {
    FS: FS_DEFAULT_PROFILE, FOIL: FOIL_DEFAULT_PROFILE, FG: FG_DEFAULT_PROFILE,
    PRES: PRESS_DEFAULT_PROFILE, DISCHARGE: DISCHARGE_DEFAULT_PROFILE,
    WATER: WATER_DEFAULT_PROFILE, PAA: PAA_DEFAULT_PROFILE,
}
CHANNEL_BY_NAME = {"FS": FS, "FG": FG, "FOIL": FOIL, "PAA": PAA, "WATER": WATER}


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def build_segmentation(segment_hours):
    """(n_segments, segment_edges, segment_of) for a given --segment_hours.

    segment_hours=None is the open-loop case: one segment covering the whole batch, so every
    setpoint lands in segment 0 and the channel gets a single scale factor.
    """
    if segment_hours is None:
        return 1, [BATCH_HOURS], (lambda t: 0)

    n_seg = int(math.ceil(BATCH_HOURS / segment_hours))
    edges = [min((k + 1) * segment_hours, BATCH_HOURS) for k in range(n_seg)]

    def segment_of(t, sh=segment_hours, n=n_seg):
        """Segment a setpoint at time t belongs to.

        Recipe.get_value_at returns the *right* setpoint, so the setpoint at time t governs the
        interval ENDING at t -- hence ceil, not floor.
        """
        return min(max(int(math.ceil(t / sh)) - 1, 0), n - 1)

    return n_seg, edges, segment_of


def piecewise_profile(profile, factors, edges, segment_of):
    """Scale a default profile by a per-segment factor.

    Setpoints are inserted at every segment edge first. Because the inserted value is read off the
    original recipe, that insertion alone does not change the control trajectory -- it only gives
    every segment a setpoint of its own to scale. (Asserted by the all-ones check in run_search.)
    """
    base = Recipe([dict(sp) for sp in profile], "base")
    times = sorted({sp["time"] for sp in profile} | set(edges))
    return [{"time": t, "value": base.get_value_at(t) * factors[segment_of(t)]} for t in times]


def scaled_recipe(factors_by_channel, edges, segment_of):
    rd = {}
    for ch, prof in DEFAULTS.items():
        sps = (piecewise_profile(prof, factors_by_channel[ch], edges, segment_of)
               if ch in factors_by_channel else [dict(sp) for sp in prof])
        rd[ch] = Recipe(sps, ch)
    return RecipeCombo(recipe_dict=rd)


def unpack(x, channels, n_seg):
    """Flat BO vector -> {channel: [factor per segment]}."""
    x = np.asarray(x, dtype=float).reshape(len(channels), n_seg)
    return {ch: x[i].tolist() for i, ch in enumerate(channels)}


def evaluate(factors_by_channel, seed, edges, segment_of):
    """One simulated batch. Returns (batch_yield_kg, max penicillin concentration)."""
    env = PenSimEnv(recipe_combo=scaled_recipe(factors_by_channel, edges, segment_of), fast=True)
    (df, _), batch_yield = env.get_batches(random_seed=int(seed), include_raman=False)
    return batch_yield, float(df[CONC_COL].max())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search(args, channels, n_seg, edges, segment_of):
    """The BO search itself. Returns (search log DataFrame, baseline yield, best factors)."""
    # All-ones must reproduce the untouched recipe exactly, or the segmentation is not neutral and
    # every "% vs baseline" below is measured against the wrong reference.
    baseline, _ = evaluate({}, args.seed, edges, segment_of)
    ones, _ = evaluate({ch: [1.0] * n_seg for ch in channels}, args.seed, edges, segment_of)
    assert abs(ones - baseline) < 1e-6, f"segmentation is not neutral: {ones} vs {baseline}"
    print(f"baseline {baseline:.1f} kg on simulator seed {args.seed} "
          f"(all-ones reproduces it exactly, diff {ones - baseline:.2e})")

    log = []

    def objective(x):
        fac = unpack(x, channels, n_seg)
        y, max_c = evaluate(fac, args.seed, edges, segment_of)
        penalised = max_c > PHYSICAL_P_MAX
        if penalised:
            y = 0.0
        log.append({"evaluation": len(log) + 1, "yield": y, "max_concentration": max_c,
                    "penalised": bool(penalised),
                    "is_random_init": len(log) < args.n_random,
                    **{f"{ch}_s{k}": fac[ch][k] for ch in channels for k in range(n_seg)}})
        print(f"  eval {log[-1]['evaluation']:3d}/{args.n_calls}: yield {y:8.1f} kg"
              f"{'  [PENALISED: max conc %.1f > %.0f]' % (max_c, PHYSICAL_P_MAX) if penalised else ''}")
        return -y

    space = [Real(args.low, args.high, name=f"{ch}_s{k}") for ch in channels for k in range(n_seg)]
    gp_minimize(objective, space, n_calls=args.n_calls, n_initial_points=args.n_random,
                acq_func="EI", random_state=args.bo_seed, noise=1e-10)

    df = pd.DataFrame(log)
    y = df["yield"].to_numpy()
    # The running summaries the CSV is meant to carry, computed once here so the plot and the file
    # can never disagree about them.
    df["best_so_far"] = np.maximum.accumulate(y)
    df["avg_so_far"] = np.cumsum(y) / np.arange(1, len(y) + 1)
    df["baseline"] = baseline
    df["delta_vs_baseline"] = y - baseline
    df["pct_vs_baseline"] = 100.0 * (y - baseline) / baseline
    df["best_pct_vs_baseline"] = 100.0 * (df["best_so_far"] - baseline) / baseline

    factor_cols = [f"{ch}_s{k}" for ch in channels for k in range(n_seg)]
    best_fac = unpack(df.loc[df["yield"].idxmax(), factor_cols].to_numpy(dtype=float),
                      channels, n_seg)
    return df, baseline, best_fac, factor_cols


def run_held_out(best_fac, args, edges, segment_of):
    """Re-run the tuned schedule and the plain recipe on the held-out block.

    This is the number that is comparable to an MC-PILCO run's A1_held_out.csv: same five batch
    realisations, same batch_yield metric. The search yield above is IN-SAMPLE -- BO tuned the
    schedule on seed {args.seed} and is scored on it -- so it is not.
    """
    seeds = [args.eval_base + i for i in range(args.n_eval_seeds)]
    assert args.seed not in seeds, f"--seed {args.seed} must not be inside the held-out block"
    rows = []
    for h in seeds:
        y_bo, _ = evaluate(best_fac, h, edges, segment_of)
        y_rec, _ = evaluate({}, h, edges, segment_of)
        rows.append({"seed": h, "yield_bo": y_bo, "yield_recipe": y_rec,
                     "delta": y_bo - y_rec, "pct": 100.0 * (y_bo - y_rec) / y_rec})
        print(f"  held-out seed {h}: BO {y_bo:8.1f}  recipe {y_rec:8.1f}  "
              f"delta {y_bo - y_rec:+8.1f} ({rows[-1]['pct']:+.1f}%)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_search_curve(df, baseline, args, out_dir, label):
    """The notebook's figure: per-evaluation yield, best-so-far, average-so-far, baseline."""
    y = df["yield"].to_numpy()
    n = len(y)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["evaluation"], y, s=30, alpha=.6, label="batch yield")
    ax.plot(df["evaluation"], df["best_so_far"], color="darkorange", lw=2, label="best so far")
    ax.plot(df["evaluation"], df["avg_so_far"], color="darkgreen", lw=2, label="average so far")
    ax.axhline(baseline, color="crimson", ls="--", label=f"baseline ({baseline:.0f} kg)")
    ax.axvline(args.n_random + .5, color="grey", ls=":", label="end of random init")
    if bool(df["penalised"].any()):
        p = df[df["penalised"]]
        ax.scatter(p["evaluation"], p["yield"], s=80, facecolors="none", edgecolors="crimson",
                   lw=1.5, label=f"penalised, conc > {PHYSICAL_P_MAX:.0f} ({len(p)})")
    ax.set(xlabel="evaluation", ylabel="batch yield (kg)",
           title=f"BO search on simulator seed {args.seed} -- {label}\n"
                 f"best {y.max():.0f} kg ({100*(y.max()-baseline)/baseline:+.2f}%), "
                 f"mean {y.mean():.0f} kg ({100*(y.mean()-baseline)/baseline:+.2f}%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "search_curve.png", dpi=150)
    plt.close(fig)


def plot_best_schedule(best_fac, channels, edges, segment_of, args, out_dir):
    """Best factor per segment, and the feed profile it produces."""
    edge_grid = np.array([0.0] + list(edges))
    grid = np.arange(0, BATCH_HOURS, 0.5)
    fig, axes = plt.subplots(len(channels), 2, figsize=(13, 3.2 * len(channels)), squeeze=False)
    for i, ch in enumerate(channels):
        axes[i][0].step(edge_grid, [best_fac[ch][0]] + list(best_fac[ch]), where="pre", lw=2)
        axes[i][0].axhline(1.0, color="crimson", ls="--")
        axes[i][0].set(title=f"{ch}: best scale per segment", xlabel="time (h)", ylabel="factor",
                       ylim=(args.low - .05, args.high + .05))

        default = Recipe([dict(sp) for sp in DEFAULTS[ch]], ch)
        tuned = Recipe(piecewise_profile(DEFAULTS[ch], best_fac[ch], edges, segment_of), ch)
        axes[i][1].plot(grid, [default.get_value_at(t) for t in grid], color="crimson", ls="--",
                        label="default")
        axes[i][1].plot(grid, [tuned.get_value_at(t) for t in grid], lw=2, label="BO")
        axes[i][1].set(title=f"{ch}: setpoint profile", xlabel="time (h)", ylabel=ch)
        axes[i][1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "best_schedule.png", dpi=150)
    plt.close(fig)


def plot_held_out(held, args, out_dir, label, mcpilco=None):
    """The MC-PILCO-comparable figure: per-batch paired yields on the shared held-out block.

    Left panel is the per-seed pairing (the arrangement A3_paired_yield.png uses, as bars rather
    than a scatter because there are only five batches); right panel is the means with standard
    errors, the A4_total_yield.png readout. `mcpilco` is an optional yield_rl series read from a
    run's A1_held_out.csv -- the whole reason the held-out block is worth running.
    """
    seeds = held["seed"].to_numpy()
    idx = np.arange(len(seeds))
    # (legend label, short tick label, values, colour) -- the legend carries the full mode
    # description, the bar-chart ticks a short one, or they overrun the axis.
    arms = [("recipe", "recipe", held["yield_recipe"].to_numpy(), "crimson"),
            (f"BO ({label})", "BO", held["yield_bo"].to_numpy(), "C0")]
    if mcpilco is not None:
        arms.append(("MC-PILCO", "MC-PILCO", np.asarray(mcpilco, dtype=float), "darkgreen"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    w = 0.8 / len(arms)
    for j, (name, _short, vals, colour) in enumerate(arms):
        ax[0].bar(idx + j * w - 0.4 + w / 2, vals, width=w, color=colour, alpha=.85, label=name)
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels([str(s) for s in seeds], rotation=20)
    ax[0].set(xlabel="held-out batch (simulator seed)", ylabel="batch yield (kg)",
              title=f"Held-out batches {seeds.min()}-{seeds.max()}\n"
                    "(the same block every MC-PILCO run is evaluated on)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")

    means = [v.mean() for _, _, v, _ in arms]
    sems = [v.std(ddof=1) / np.sqrt(len(v)) for _, _, v, _ in arms]
    ax[1].bar([a[1] for a in arms], means, yerr=sems, capsize=6,
              color=[a[3] for a in arms], alpha=.85)
    ax[1].axhline(means[0], color="crimson", ls="--", lw=1)
    for i, (m, s) in enumerate(zip(means, sems)):
        ax[1].text(i, m + s + 40, f"{m:.0f}\n({100*(m-means[0])/means[0]:+.1f}%)",
                   ha="center", fontsize=9)
    ax[1].set(ylabel="mean held-out yield (kg)",
              title=f"Mean over {len(seeds)} held-out batches (+/- s.e.m.)")
    ax[1].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "held_out.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def _next_run_dir(root, seed, tag):
    n = 1
    while (root / f"seed{seed}_{tag}_{n}").exists():
        n += 1
    return root / f"seed{seed}_{tag}_{n}"


def main():
    p = argparse.ArgumentParser(
        description="BO over Fs recipe scale factors; writes CSVs and PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=4000,
                   help="SIMULATOR seed of the batch BO searches on (NOT an MC-PILCO --seed: that "
                        "one maps to seed*1000, so MC-PILCO --seed 4 <-> --seed 4000 here)")
    p.add_argument("--segment_hours", type=float, default=None,
                   help="length of one scaling segment in hours; OMIT for open loop (a single "
                        "factor per channel held for the whole batch)")
    p.add_argument("--n_calls", type=int, default=100, help="BO evaluations (= simulated batches)")
    p.add_argument("--n_random", type=int, default=5, help="random-init evaluations before EI")
    p.add_argument("--bo_seed", type=int, default=None,
                   help="gp_minimize random_state; defaults to --seed, matching the notebook (which "
                        "used one integer for both, confounding the batch realisation with the "
                        "search's own randomness -- set this to separate them)")
    p.add_argument("--channels", type=str, default="FS",
                   help=f"comma-separated recipe channels to scale, from {sorted(CHANNEL_BY_NAME)}")
    p.add_argument("--low", type=float, default=0.5, help="lower bound on the scale factor")
    p.add_argument("--high", type=float, default=1.5, help="upper bound on the scale factor")
    p.add_argument("--eval_base", type=int, default=700000,
                   help="first held-out simulator seed; 700000 is what eval_*_phase_lib uses for "
                        "EVERY MC-PILCO run regardless of its --seed")
    p.add_argument("--n_eval_seeds", type=int, default=5, help="held-out batches (0 to skip)")
    p.add_argument("--mcpilco_run", type=str, default=None,
                   help="path to an MC-PILCO run dir; its A1_held_out.csv yield_rl column is "
                        "overlaid on held_out.png for a like-for-like comparison")
    p.add_argument("--out_dir", type=str, default=None, help="override the output directory")
    args = p.parse_args()

    if args.bo_seed is None:
        args.bo_seed = args.seed

    channels = []
    for name in args.channels.split(","):
        key = name.strip().upper()
        if key not in CHANNEL_BY_NAME:
            p.error(f"unknown channel {name!r}; pick from {sorted(CHANNEL_BY_NAME)}")
        channels.append(CHANNEL_BY_NAME[key])

    n_seg, edges, segment_of = build_segmentation(args.segment_hours)
    label = ("open loop, 1 factor/channel" if args.segment_hours is None
             else f"{n_seg} segments of {args.segment_hours:g} h")
    tag = "openloop" if args.segment_hours is None else f"seg{args.segment_hours:g}"

    out_dir = Path(args.out_dir) if args.out_dir else _next_run_dir(_RESULTS_ROOT, args.seed, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"search space  : {n_seg} segment(s) x {len(channels)} channel(s) = "
          f"{n_seg * len(channels)} dimensions  ({label})")
    print(f"budget        : {args.n_calls} evaluations = {args.n_calls} simulated batches")
    print(f"output        : {out_dir}\n")

    df, baseline, best_fac, factor_cols = run_search(args, channels, n_seg, edges, segment_of)

    # --- CSV: one row per evaluation, carrying its own running summaries and the baseline --------
    lead = ["evaluation", "yield", "best_so_far", "avg_so_far", "baseline", "delta_vs_baseline",
            "pct_vs_baseline", "best_pct_vs_baseline", "is_random_init", "penalised",
            "max_concentration"]
    df[lead + factor_cols].to_csv(out_dir / "search_log.csv", index=False)

    y = df["yield"].to_numpy()
    print(f"\nbaseline {baseline:.1f}")
    print(f"best     {y.max():.1f}  ({100*(y.max()-baseline)/baseline:+.2f}%)")
    print(f"mean     {y.mean():.1f}  ({100*(y.mean()-baseline)/baseline:+.2f}%)")
    best_call = int(df.loc[df['yield'].idxmax(), 'evaluation'])
    print(f"best found at evaluation {best_call} of {args.n_calls}"
          + ("  <-- during random init; the EI phase never improved on it"
             if best_call <= args.n_random else ""))

    plot_search_curve(df, baseline, args, out_dir, label)
    plot_best_schedule(best_fac, channels, edges, segment_of, args, out_dir)

    # --- held-out block: the MC-PILCO-comparable readout -----------------------------------------
    held = None
    if args.n_eval_seeds > 0:
        print(f"\nheld-out evaluation on {args.n_eval_seeds} unseen batches from {args.eval_base}:")
        held = run_held_out(best_fac, args, edges, segment_of)
        held.to_csv(out_dir / "held_out.csv", index=False)

        mc = None
        if args.mcpilco_run:
            a1 = Path(args.mcpilco_run) / "A1_held_out.csv"
            if a1.exists():
                d = pd.read_csv(a1).set_index("seed")
                # Cross-check on the arm both files share: if the recipe yields disagree, the two
                # runs are not measuring the same batches and overlaying them would be misleading.
                dev = float(np.max(np.abs(
                    held.set_index("seed")["yield_recipe"] - d["yield_recipe"].reindex(held["seed"]).values)))
                assert dev < 1e-6, (f"recipe arm differs from {a1} by {dev:.3e} kg -- not the same "
                                    "held-out batches, refusing to overlay")
                mc = d["yield_rl"].reindex(held["seed"]).to_numpy()
                print(f"  overlaying MC-PILCO from {a1} (recipe arms agree to {dev:.1e} kg)")
            else:
                print(f"  !! {a1} not found -- plotting without the MC-PILCO overlay")

        plot_held_out(held, args, out_dir, label, mcpilco=mc)
        print(f"\nheld-out mean: BO {held['yield_bo'].mean():.1f} kg  "
              f"recipe {held['yield_recipe'].mean():.1f} kg  "
              f"({100*(held['yield_bo'].mean()-held['yield_recipe'].mean())/held['yield_recipe'].mean():+.2f}%)"
              + (f"  MC-PILCO {mc.mean():.1f} kg" if mc is not None else ""))
        if held["yield_bo"].min() < 0.5 * held["yield_recipe"].min():
            print("  !! at least one held-out batch collapsed: the schedule tuned on a single "
                  "search seed does not transfer to every realisation.")

    # --- run note --------------------------------------------------------------------------------
    lines = [f"run timestamp : {datetime.datetime.now().isoformat(timespec='seconds')}", "",
             "== run parameters =="]
    lines += [f"{k} = {v}" for k, v in vars(args).items()]
    lines += ["", "== derived ==",
              f"channels = {channels}", f"n_segments = {n_seg}",
              f"search_dimensions = {n_seg * len(channels)}", f"mode = {label}",
              f"segment_edges = {edges}", "", "== results ==",
              f"baseline_kg = {baseline:.4f}", f"best_kg = {y.max():.4f}",
              f"mean_kg = {y.mean():.4f}", f"best_at_evaluation = {best_call}",
              f"best_factors = {best_fac}"]
    if held is not None:
        lines += [f"heldout_mean_bo_kg = {held['yield_bo'].mean():.4f}",
                  f"heldout_mean_recipe_kg = {held['yield_recipe'].mean():.4f}"]
    (out_dir / "note.txt").write_text("\n".join(lines) + "\n")

    written = ["search_log.csv", "search_curve.png", "best_schedule.png", "note.txt"]
    if held is not None:
        written += ["held_out.csv", "held_out.png"]
    print(f"\nwritten to {out_dir}:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()

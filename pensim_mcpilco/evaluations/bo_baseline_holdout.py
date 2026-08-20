"""BO open-loop recipe baseline, scored under the MC-PILCO held-out protocol.

Port of experiments/01_bo_baseline_adapted_action.ipynb into a runnable experiment, with the
three changes needed to make its numbers commensurable with results/full's A1/A2 tables:

  1. TUNE ON A TRAINING BATCH, SCORE ON THE HELD-OUT ONES. The notebook tunes and scores on the
     same simulator seed, which is hindsight-optimal and not comparable with a policy that never
     saw those batches. Here BO searches on ONE training realisation (4000 / 5000 / 6000 -- the
     first batch of each `--seed 4/5/6` run's block, since wrapper_par seed_offset = seed*1000),
     then the winning schedule is REPLAYED on the same five held-out batches the RL arm is graded
     on (eval_single_phase_lib.eval_held_out's eval_base = 700000).

  2. THE SAME FEASIBILITY GATE AS THE RL ARM. The notebook only zeroed yield above 40 g/L
     penicillin, which is not the envelope the policy is graded against. Here a batch is
     COLLAPSED iff Viscosity peaks above VISC_MAX or Wt above WT_OVERFLOW -- the exact rule in
     eval_utils.feasibility_gated_yield_kg -- and a collapsed batch scores 0. Deliberately ONLY
     those two bounds: PAA_BAND is reported as a diagnostic (paa_frac_out_of_band) but does not
     gate, because it does not gate the RL arm either, and max_P is reported (p_implausible) but
     likewise does not gate. Adding bounds the RL arm is not held to would make BO look worse for
     reasons unrelated to control quality.
     The gate is applied INSIDE the search objective as well as in the replay table, so BO is
     optimising against the same bar it is scored on. `constraint_diagnostics` is imported from
     experiments/eval_utils.py rather than reimplemented, so there is exactly one definition of
     "breached" shared with the RL evaluation.

  3. YIELD IS PenSimPy's `batch_yield`. Identical to the RL arm's yield_kg (= sum of the
     per-step yield_per_run); verified equal to ~1e-12 on seeds 4/5/6/4000/700000-700004.

Two asymmetries are LEFT IN and reported rather than papered over -- both cut against BO:
  * budget: BO gets `--n_calls` (default 100) simulated batches on one realisation; a `--seed N`
    MC-PILCO run gets 15 (5 explorations + 10 trials) across 15 different realisations.
  * BO's schedule is OPEN-LOOP -- `--segment_hours` (default 25 h) fixed multipliers on the Fs
    setpoint chosen before the batch starts, no state feedback -- against the policy's 5 h
    closed-loop decisions. The train->held-out drop is the price of that, and is reported as
    `generalisation_gap` in summary_final.csv.

Usage
    PYTHONPATH=.. python evaluations/bo_baseline_holdout.py                     # all three seeds
    PYTHONPATH=.. python evaluations/bo_baseline_holdout.py --train_seeds 4000  # one seed
    PYTHONPATH=.. python evaluations/bo_baseline_holdout.py --aggregate         # re-build summaries

Per-seed work is independent, so the three searches can be run as three parallel processes and
then combined with --aggregate.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # save-only: this script is run headless
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))

from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv
from utils.constants import STEP_IN_HOURS
from utils.ode_patch import patch_fastodeint
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from skopt import gp_minimize
from skopt.space import Real
from scipy import stats

# Shared with the RL evaluation: one definition of "breached" for both arms (see module docstring).
from experiments.eval_utils import constraint_diagnostics
from mcpilco.pensim_wrapper import VISC_MAX, WT_OVERFLOW

patch_fastodeint()

CONC_COL = "Penicillin Concentration"
BATCH_HOURS = 230.0
SCALED = [FS]
# Matches the RL action range exactly: PenSimWrapper applies Fs * (1 + FS_SCALE * a) with
# FS_SCALE = 0.5 and a in [-1, 1], i.e. the same [0.5, 1.5] multiplier. Do not widen.
LOW, HIGH = 0.5, 1.5
P_IMPLAUSIBLE = 40.0           # reported only; does NOT gate (see docstring)

TRAIN_SEEDS = (4000, 5000, 6000)          # = seed*1000 for --seed 4/5/6
EVAL_SEEDS = tuple(700000 + i for i in range(5))   # eval_single_phase_lib's eval_base block

DEFAULTS = {
    FS: FS_DEFAULT_PROFILE, FOIL: FOIL_DEFAULT_PROFILE, FG: FG_DEFAULT_PROFILE,
    PRES: PRESS_DEFAULT_PROFILE, DISCHARGE: DISCHARGE_DEFAULT_PROFILE,
    WATER: WATER_DEFAULT_PROFILE, PAA: PAA_DEFAULT_PROFILE,
}


# ---------------------------------------------------------------------------
# Recipe construction (verbatim from the notebook -- the segmentation must stay
# neutral at all-ones factors, which _sanity_check below asserts)
# ---------------------------------------------------------------------------

class Segmentation:
    """Segment edges and the factor->profile mapping for one SEGMENT_HOURS setting."""

    def __init__(self, segment_hours):
        self.segment_hours = float(segment_hours)
        self.n_seg = int(math.ceil(BATCH_HOURS / self.segment_hours))
        self.edges = [min((k + 1) * self.segment_hours, BATCH_HOURS) for k in range(self.n_seg)]

    def segment_of(self, t):
        """Segment a setpoint at time t belongs to.

        Recipe.get_value_at returns the *right* setpoint, so the setpoint at time t governs the
        interval ending at t -- hence ceil, not floor.
        """
        return min(max(int(math.ceil(t / self.segment_hours)) - 1, 0), self.n_seg - 1)

    def piecewise_profile(self, profile, factors):
        """Scale a default profile by a per-segment factor.

        Setpoints are inserted at every segment edge first. Because the inserted value is read off
        the original recipe, that insertion alone does not change the control trajectory -- it only
        gives every segment a setpoint of its own to scale.
        """
        base = Recipe([dict(sp) for sp in profile], "base")
        times = sorted({sp["time"] for sp in profile} | set(self.edges))
        return [{"time": t, "value": base.get_value_at(t) * factors[self.segment_of(t)]}
                for t in times]

    def scaled_recipe(self, factors_by_channel):
        rd = {}
        for ch, prof in DEFAULTS.items():
            sps = (self.piecewise_profile(prof, factors_by_channel[ch])
                   if ch in factors_by_channel else [dict(sp) for sp in prof])
            rd[ch] = Recipe(sps, ch)
        return RecipeCombo(recipe_dict=rd)

    def unpack(self, x):
        """Flat BO vector -> {channel: [factor per segment]}."""
        x = np.asarray(x, dtype=float).reshape(len(SCALED), self.n_seg)
        return {ch: x[i].tolist() for i, ch in enumerate(SCALED)}

    def space(self):
        return [Real(LOW, HIGH, name=f"{ch}_s{k}")
                for ch in SCALED for k in range(self.n_seg)]

    def factor_columns(self):
        return [f"{ch}_s{k}" for ch in SCALED for k in range(self.n_seg)]


# ---------------------------------------------------------------------------
# Simulation + feasibility
# ---------------------------------------------------------------------------

def _monitor_from_batch_data(bx):
    """Build the monitor dict constraint_diagnostics expects from PenSimPy's batch_data.

    Same channels PenSimWrapper.rollout records, read off the same arrays (bx.<ch>.y), so the
    breach flags here are computed on exactly the signals the RL arm's flags are computed on.
    """
    def arr(name):
        return np.nan_to_num(np.asarray(getattr(bx, name).y, dtype=float), nan=0.0)

    P = arr("P")
    return {"t": np.arange(1, len(P) + 1) * STEP_IN_HOURS, "P": P, "Wt": arr("Wt"),
            "Viscosity": arr("Viscosity"), "PAA": arr("PAA"), "Fpaa": arr("Fpaa")}


def simulate(seg, factors_by_channel, seed):
    """One batch. Returns raw yield, the feasibility-gated yield, and the diagnostics."""
    env = PenSimEnv(recipe_combo=seg.scaled_recipe(factors_by_channel), fast=True)
    (df, _), batch_yield, bx = env.get_batches(random_seed=seed, include_raman=False,
                                               return_batch_data=True)
    diag = constraint_diagnostics(_monitor_from_batch_data(bx))
    # ONLY these two bounds gate -- identical to eval_utils.feasibility_gated_yield_kg.
    collapsed = bool(diag["wt_overflow"] or diag["visc_exceed"])
    max_P = float(df[CONC_COL].max())
    return {
        "yield_raw": float(batch_yield),
        "yield_gated": 0.0 if collapsed else float(batch_yield),
        "collapsed": collapsed,
        "max_P": max_P,
        "p_implausible": bool(max_P > P_IMPLAUSIBLE),
        **{k: v for k, v in diag.items()},
    }


def _sanity_check(seg, seed):
    """All-ones factors must reproduce the untouched recipe exactly (the notebook's cell 4)."""
    base = simulate(seg, {}, seed)
    ones = simulate(seg, {ch: [1.0] * seg.n_seg for ch in SCALED}, seed)
    assert abs(ones["yield_raw"] - base["yield_raw"]) < 1e-6, \
        f"segmentation is not neutral: {ones['yield_raw']} vs {base['yield_raw']}"
    return base


# ---------------------------------------------------------------------------
# Search + held-out replay
# ---------------------------------------------------------------------------

def run_seed(seg, train_seed, eval_seeds, n_calls, n_random, out_dir):
    """BO search on `train_seed`, then replay the winner on `eval_seeds`. Writes per-seed CSVs."""
    print(f"\n=== train seed {train_seed} | {seg.n_seg} segments x {len(SCALED)} channels "
          f"= {seg.n_seg * len(SCALED)} dims | n_calls={n_calls} ===", flush=True)

    baseline = _sanity_check(seg, train_seed)
    print(f"recipe baseline on {train_seed}: {baseline['yield_raw']:.1f} kg "
          f"(collapsed={baseline['collapsed']}, max_visc={baseline['max_viscosity']:.1f})",
          flush=True)

    cols = seg.factor_columns()
    log = []

    def objective(x):
        fac = seg.unpack(x)
        r = simulate(seg, fac, train_seed)
        log.append({"call": len(log) + 1, **{k: r[k] for k in
                    ("yield_raw", "yield_gated", "collapsed", "max_viscosity", "max_Wt",
                     "wt_overflow", "visc_exceed", "max_P", "p_implausible",
                     "paa_frac_out_of_band", "final_P")},
                    **{c: v for c, v in zip(cols, np.asarray(x, dtype=float))}})
        if len(log) % 10 == 0:
            best = max(r_["yield_gated"] for r_ in log)
            print(f"  call {len(log):3d}/{n_calls}  y={r['yield_gated']:7.1f}  best={best:7.1f}",
                  flush=True)
        return -r["yield_gated"]

    gp_minimize(objective, seg.space(), n_calls=n_calls, n_initial_points=n_random,
                acq_func="EI", random_state=train_seed, noise=1e-10)

    log_df = pd.DataFrame(log)
    log_df.to_csv(Path(out_dir) / f"bo_log_seed{train_seed}.csv", index=False)

    # Winner = best FEASIBLE yield (gated), so a collapsed batch can never be selected.
    best_row = log_df.loc[log_df["yield_gated"].idxmax()]
    best_fac = seg.unpack(best_row[cols].values)
    pd.DataFrame([{"train_seed": train_seed, **{c: best_row[c] for c in cols}}]).to_csv(
        Path(out_dir) / f"best_schedule_seed{train_seed}.csv", index=False)

    # --- held-out replay -----------------------------------------------------
    rows = []
    for h in eval_seeds:
        bo = simulate(seg, best_fac, h)
        rec = simulate(seg, {}, h)
        rows.append({
            "train_seed": train_seed, "eval_seed": h,
            "yield_bo": bo["yield_raw"], "yield_bo_gated": bo["yield_gated"],
            "yield_recipe": rec["yield_raw"], "yield_recipe_gated": rec["yield_gated"],
            "delta": bo["yield_raw"] - rec["yield_raw"],
            "delta_gated": bo["yield_gated"] - rec["yield_gated"],
            "bo_collapsed": bo["collapsed"], "recipe_collapsed": rec["collapsed"],
            "bo_max_viscosity": bo["max_viscosity"], "bo_max_Wt": bo["max_Wt"],
            "bo_visc_exceed": bo["visc_exceed"], "bo_wt_overflow": bo["wt_overflow"],
            "bo_final_P": bo["final_P"], "bo_max_P": bo["max_P"],
            "bo_paa_frac_out_of_band": bo["paa_frac_out_of_band"],
            "recipe_max_viscosity": rec["max_viscosity"], "recipe_final_P": rec["final_P"],
        })
        print(f"  held-out {h}: bo={bo['yield_raw']:7.1f} "
              f"({'COLLAPSED' if bo['collapsed'] else 'ok':9s}) "
              f"recipe={rec['yield_raw']:7.1f}  delta={rows[-1]['delta']:+8.1f}", flush=True)

    hold_df = pd.DataFrame(rows)
    hold_df.to_csv(Path(out_dir) / f"holdout_seed{train_seed}.csv", index=False)

    meta = {"train_seed": train_seed, "segment_hours": seg.segment_hours, "n_seg": seg.n_seg,
            "n_calls": n_calls, "n_random": n_random, "low": LOW, "high": HIGH,
            "visc_max": VISC_MAX, "wt_overflow": WT_OVERFLOW,
            "train_baseline_yield": baseline["yield_raw"],
            "eval_seeds": list(eval_seeds)}
    (Path(out_dir) / f"meta_seed{train_seed}.json").write_text(json.dumps(meta, indent=2))

    plot_convergence(log_df, baseline["yield_raw"], train_seed, n_random, out_dir)
    plot_best_schedule(seg, best_fac, train_seed, out_dir)
    return log_df, hold_df


# ---------------------------------------------------------------------------
# Per-seed plots (the notebook's cell 6 and cell 9, plus collapse marking)
# ---------------------------------------------------------------------------

def plot_convergence(log_df, baseline, train_seed, n_random, out_dir):
    """Search trace on RAW yield.

    Collapsed batches are NOT drawn -- only feasible batches appear as points -- but they are
    still counted in the running average at the yield they actually produced, so the mean is
    never pulled toward zero by a breach (their count lives in summary_final.csv's
    train_n_collapsed). The only gated series is "best feasible so far": the search optimises the
    gated objective, so that curve is what BO was actually climbing and its endpoint is the
    schedule replayed on the held-out batches.
    """
    raw = log_df["yield_raw"].values
    gated = log_df["yield_gated"].values
    coll = log_df["collapsed"].values.astype(bool)
    n = len(raw)
    x = np.arange(1, n + 1)
    avg = np.cumsum(raw) / x

    # Second average over the SEARCH-PROPER evaluations only: the first n_random calls are a
    # random design (BO's analogue of MC-PILCO's exploration episodes) and are not the result of
    # any optimisation, so including them drags the running mean toward random-schedule yield for
    # the whole trace. This line answers "how good are the batches BO actually chose".
    n_r = int(n_random)
    xb, avg_b = None, None
    if n > n_r:
        xb = x[n_r:]
        avg_b = np.cumsum(raw[n_r:]) / np.arange(1, n - n_r + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.scatter(x[~coll], raw[~coll], s=26, alpha=.5, color="C0", label="batch yield")
    ax.plot(x, np.maximum.accumulate(gated), color="darkorange", lw=2, label="best so far")
    ax.plot(x, avg, color="darkgreen", lw=2, label="average so far (all evaluations)")
    if avg_b is not None:
        ax.plot(xb, avg_b, color="rebeccapurple", lw=2,
                label=f"average so far (excl. {n_r} random init)")
    ax.axhline(baseline, color="crimson", ls="--", label=f"recipe baseline ({baseline:.0f})")
    ax.axvline(n_random + .5, color="grey", ls=":", label="end of random init")

    # The two numbers asked for, on each average line: early in the search and at the end.
    for k in sorted({m for m in (10, n) if 1 <= m <= n}):
        last = (k == n)
        ax.annotate(f"avg@{k} = {avg[k - 1]:.0f}", xy=(k, avg[k - 1]),
                    xytext=(6, -20 if last else 12), textcoords="offset points",
                    ha="right" if last else "left", fontsize=9, color="darkgreen",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkgreen", alpha=.85))
        if avg_b is not None and k > n_r:
            v = avg_b[k - 1 - n_r]
            ax.annotate(f"avg@{k} = {v:.0f}", xy=(k, v),
                        xytext=(6, 16 if last else -22), textcoords="offset points",
                        ha="right" if last else "left", fontsize=9, color="rebeccapurple",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="rebeccapurple",
                                  alpha=.85))

    ax.set(xlabel="evaluation", ylabel="batch yield (kg)",
           title=f"BO search on training batch {train_seed}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"bo_convergence_seed{train_seed}.png", dpi=150)
    plt.close(fig)


def plot_best_schedule(seg, best_fac, train_seed, out_dir):
    edges = np.array([0.0] + seg.edges)
    grid = np.arange(0, BATCH_HOURS, 0.5)

    fig, axes = plt.subplots(len(SCALED), 2, figsize=(13, 3.4 * len(SCALED)), squeeze=False)
    for i, ch in enumerate(SCALED):
        axes[i][0].step(edges, [best_fac[ch][0]] + best_fac[ch], where="pre", lw=2)
        axes[i][0].axhline(1.0, color="crimson", ls="--")
        axes[i][0].set(title=f"{ch}: best scale per segment (train {train_seed})",
                       xlabel="time (h)", ylabel="factor", ylim=(LOW - .05, HIGH + .05))

        default = Recipe([dict(sp) for sp in DEFAULTS[ch]], ch)
        tuned = Recipe(seg.piecewise_profile(DEFAULTS[ch], best_fac[ch]), ch)
        axes[i][1].plot(grid, [default.get_value_at(t) for t in grid], color="crimson", ls="--",
                        label="default recipe")
        axes[i][1].plot(grid, [tuned.get_value_at(t) for t in grid], lw=2, label="BO")
        axes[i][1].set(title=f"{ch}: setpoint profile", xlabel="time (h)", ylabel=ch)
        axes[i][1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"bo_best_schedule_seed{train_seed}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _paired_stats(d):
    """Paired BO-vs-recipe stats over a set of held-out batches (mirrors A2_paired_stats.csv)."""
    n = len(d)
    delta = d["delta"].values
    sem = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    out = {
        "n": n,
        "mean_yield_bo": float(d["yield_bo"].mean()),
        "mean_yield_bo_gated": float(d["yield_bo_gated"].mean()),
        "mean_yield_recipe": float(d["yield_recipe"].mean()),
        "mean_delta": float(delta.mean()),
        "delta_ci95_lo": float(delta.mean() - tcrit * sem) if n > 1 else float("nan"),
        "delta_ci95_hi": float(delta.mean() + tcrit * sem) if n > 1 else float("nan"),
        "winrate_bo_gt_recipe": float((delta > 0).mean()),
    }
    out["ttest_p"] = float(stats.ttest_rel(d["yield_bo"], d["yield_recipe"]).pvalue) if n > 1 else float("nan")
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(d["yield_bo"], d["yield_recipe"]).pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    return out


def aggregate(out_dir, eval_seeds):
    out_dir = Path(out_dir)
    hold_files = sorted(out_dir.glob("holdout_seed*.csv"))
    log_files = sorted(out_dir.glob("bo_log_seed*.csv"))
    if not hold_files:
        raise SystemExit(f"no holdout_seed*.csv in {out_dir} -- run the searches first")

    hold = pd.concat([pd.read_csv(f) for f in hold_files], ignore_index=True)
    hold.to_csv(out_dir / "holdout_all.csv", index=False)

    logs = {int(f.stem.replace("bo_log_seed", "")): pd.read_csv(f) for f in log_files}
    metas = {}
    for f in out_dir.glob("meta_seed*.json"):
        m = json.loads(f.read_text())
        metas[int(m["train_seed"])] = m

    # Per-seed figures are rebuilt here as well as in run_seed, so --aggregate refreshes every
    # plot from the saved CSVs without re-simulating a single batch.
    for s, lg in logs.items():
        meta = metas.get(s, {})
        plot_convergence(lg, meta.get("train_baseline_yield", float("nan")), s,
                         meta.get("n_random", 5), out_dir)
        bs_file = out_dir / f"best_schedule_seed{s}.csv"
        if bs_file.exists() and "segment_hours" in meta:
            seg = Segmentation(meta["segment_hours"])
            row = pd.read_csv(bs_file).iloc[0]
            plot_best_schedule(seg, seg.unpack(row[seg.factor_columns()].values), s, out_dir)

    # --- paired stats, per train seed and pooled ----------------------------
    rows = [{"train_seed": s, **_paired_stats(d)} for s, d in hold.groupby("train_seed")]
    rows.append({"train_seed": "POOLED", **_paired_stats(hold)})
    pd.DataFrame(rows).to_csv(out_dir / "paired_stats.csv", index=False)

    # --- the requested final summary ---------------------------------------
    # EVERY mean/std/best/min column is computed on the RAW yield: a collapsed batch keeps its
    # natural yield rather than being zeroed. The gated view is carried alongside in the
    # explicitly-suffixed *_gated columns, and the collapse COUNTS (train_n_collapsed /
    # eval_n_collapsed / eval_collapsed_seeds) are how a breach shows up in the raw columns --
    # nothing is silently zeroed. The one exception is train_best_yield_gated, which is the score
    # of the schedule that was actually selected and replayed (the search optimises the gated
    # objective, so the winner is feasible by construction and its gated score equals its raw one).
    def _stats(seed_label, d, lg, meta):
        bo, rec = d["yield_bo"].values, d["yield_recipe"].values
        bo_g, rec_g = d["yield_bo_gated"].values, d["yield_recipe_gated"].values
        row = {
            "train_seed": seed_label,
            "eval_seeds": "|".join(str(int(e)) for e in sorted(d["eval_seed"].unique())),
            "n_eval_batches": len(d),
            # --- training side (raw = collapsed batches keep their yield)
            "n_train_batches": int(len(lg)) if lg is not None else np.nan,
            "train_baseline_yield": meta.get("train_baseline_yield", np.nan),
            "train_best_yield_raw": float(lg["yield_raw"].max()) if lg is not None else np.nan,
            "train_mean_yield_raw": float(lg["yield_raw"].mean()) if lg is not None else np.nan,
            "train_std_yield_raw": float(lg["yield_raw"].std(ddof=1)) if lg is not None else np.nan,
            "train_best_yield_gated": float(lg["yield_gated"].max()) if lg is not None else np.nan,
            "train_mean_yield_gated": float(lg["yield_gated"].mean()) if lg is not None else np.nan,
            "train_n_collapsed": int(lg["collapsed"].sum()) if lg is not None else np.nan,
            "train_collapse_rate": float(lg["collapsed"].mean()) if lg is not None else np.nan,
            # --- held-out side, BO
            "eval_best_bo_raw": float(bo.max()),
            "eval_mean_bo_raw": float(bo.mean()),
            "eval_std_bo_raw": float(bo.std(ddof=1)),
            "eval_min_bo_raw": float(bo.min()),
            "eval_mean_bo_gated": float(bo_g.mean()),
            "eval_std_bo_gated": float(bo_g.std(ddof=1)),
            "eval_n_collapsed": int(d["bo_collapsed"].sum()),
            "eval_collapsed_seeds": "|".join(
                str(int(e)) for e in d.loc[d["bo_collapsed"], "eval_seed"]) or "none",
            # per-batch detail, so this one file stands alone without holdout_all.csv
            "eval_yields_bo": "|".join(f"{int(r.eval_seed)}:{r.yield_bo:.1f}"
                                       f"{'*' if r.bo_collapsed else ''}"
                                       for r in d.itertuples()),
            # --- held-out side, recipe reference
            "eval_mean_recipe": float(rec.mean()),
            "eval_std_recipe": float(rec.std(ddof=1)),
            "eval_mean_recipe_gated": float(rec_g.mean()),
            "eval_recipe_n_collapsed": int(d["recipe_collapsed"].sum()),
            "eval_yields_recipe": "|".join(f"{int(r.eval_seed)}:{r.yield_recipe:.1f}"
                                           f"{'*' if r.recipe_collapsed else ''}"
                                           for r in d.itertuples()),
            # --- comparison (raw)
            "mean_delta_raw": float((bo - rec).mean()),
            "std_delta_raw": float((bo - rec).std(ddof=1)),
            "mean_delta_gated": float((bo_g - rec_g).mean()),
        }
        # in-sample score of the replayed schedule minus what it delivered held-out: the price of
        # tuning on a single realisation.
        row["generalisation_gap"] = (row["train_best_yield_gated"] - row["eval_mean_bo_raw"]
                                     if lg is not None else np.nan)
        return row

    summary = [_stats(s, d, logs.get(s), metas.get(s, {}))
               for s, d in hold.groupby("train_seed")]
    sdf = pd.DataFrame(summary).sort_values("train_seed")

    # POOLED: pool the underlying batches, not the per-seed means, so the std is the spread over
    # all 15 held-out batches / all search batches rather than a mean of standard deviations.
    all_logs = pd.concat(logs.values(), ignore_index=True) if logs else None
    pooled = _stats("POOLED", hold, all_logs,
                    {"train_baseline_yield": float(sdf["train_baseline_yield"].mean())})
    pooled["eval_collapsed_seeds"] = "|".join(
        f"{int(r.train_seed)}:{int(r.eval_seed)}"
        for r in hold[hold["bo_collapsed"]].itertuples()) or "none"
    pooled["eval_yields_bo"] = "|".join(
        f"{int(r.train_seed)}/{int(r.eval_seed)}:{r.yield_bo:.1f}"
        f"{'*' if r.bo_collapsed else ''}" for r in hold.itertuples())
    pooled["eval_yields_recipe"] = sdf["eval_yields_recipe"].iloc[0]  # same 5 batches every row
    pooled["generalisation_gap"] = float(sdf["generalisation_gap"].mean())

    sdf = pd.concat([sdf, pd.DataFrame([pooled])], ignore_index=True)
    sdf.to_csv(out_dir / "summary_final.csv", index=False)

    plot_holdout(hold, out_dir)
    plot_train_vs_holdout(sdf, out_dir)
    plot_collapse(sdf, out_dir)

    print("\n=== summary_final.csv (means/stds on RAW yield: collapsed batches keep theirs) ===")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(sdf[["train_seed", "train_best_yield_gated", "train_mean_yield_raw",
                   "train_n_collapsed", "eval_mean_bo_raw", "eval_std_bo_raw", "eval_n_collapsed",
                   "eval_mean_recipe", "mean_delta_raw",
                   "generalisation_gap"]].to_string(index=False))
    return sdf


# ---------------------------------------------------------------------------
# Budget checkpoints
# ---------------------------------------------------------------------------

def checkpoint_table(out_dir, checkpoints, eval_seeds):
    """Search state and held-out transfer after the first `k` BO evaluations, for each k.

    The point of the k=15 row is budget parity with MC-PILCO: a `--seed N` run spends 5
    exploration batches + 10 trials = 15 simulated batches, so "after 5 expl + 10" is 15 BO
    evaluations here (5 random-design points + 10 surrogate-guided ones). k=100 is the full BO
    budget, 6.7x what the RL arm ever sees.

    For each k the schedule that BO *would have shipped* at that point (its best FEASIBLE batch so
    far) is replayed on the held-out block, so eval columns measure transfer, not hindsight.
    Training columns are raw yields -- collapsed batches keep their own yield and are counted
    separately in train_n_collapsed.
    """
    out_dir = Path(out_dir)
    logs = {int(f.stem.replace("bo_log_seed", "")): pd.read_csv(f)
            for f in sorted(out_dir.glob("bo_log_seed*.csv"))}
    if not logs:
        raise SystemExit(f"no bo_log_seed*.csv in {out_dir} -- run the searches first")
    metas = {}
    for f in out_dir.glob("meta_seed*.json"):
        m = json.loads(f.read_text())
        metas[int(m["train_seed"])] = m

    rows = []
    for s, lg in sorted(logs.items()):
        meta = metas.get(s, {})
        seg = Segmentation(meta.get("segment_hours", 25.0))
        cols = seg.factor_columns()
        for k in checkpoints:
            k = min(int(k), len(lg))
            sub = lg.iloc[:k]
            raw = sub["yield_raw"].values
            best_row = sub.loc[sub["yield_gated"].idxmax()]
            best_fac = seg.unpack(best_row[cols].values)

            print(f"  seed {s} @ {k} evals: replaying its best schedule "
                  f"({best_row['yield_gated']:.1f} kg in-sample) on {len(eval_seeds)} held-out "
                  f"batches", flush=True)
            ev = [simulate(seg, best_fac, h) for h in eval_seeds]
            ey = np.array([e["yield_raw"] for e in ev])
            ec = np.array([e["collapsed"] for e in ev])
            rec = np.array([simulate(seg, {}, h)["yield_raw"] for h in eval_seeds])

            rows.append({
                "train_seed": s,
                "checkpoint": f"ep{k - meta.get('n_random', 5)}" if k < len(lg) else f"ep{k}",
                "n_evals": k,
                # --- search side (raw: collapsed batches keep their yield)
                "train_mean_raw": float(raw.mean()),
                "train_std_raw": float(raw.std(ddof=1)),
                "train_best": float(best_row["yield_gated"]),
                "train_n_collapsed": int(sub["collapsed"].sum()),
                # --- held-out transfer of the schedule BO would have shipped at this point
                "eval_mean_raw": float(ey.mean()),
                "eval_std_raw": float(ey.std(ddof=1)),
                "eval_best_raw": float(ey.max()),
                "eval_min_raw": float(ey.min()),
                "eval_n_collapsed": int(ec.sum()),
                "eval_collapsed_seeds": "|".join(str(int(h)) for h, c in zip(eval_seeds, ec) if c)
                                        or "none",
                "eval_mean_recipe": float(rec.mean()),
                "mean_delta_raw": float((ey - rec).mean()),
                # kept so the POOLED row can pool the actual batches rather than average the
                # per-seed statistics (dropped from the CSV before writing)
                "_eval_yields": ey, "_eval_deltas": ey - rec,
            })

    df = pd.DataFrame(rows)
    # pooled row per checkpoint: pool the batches, not the per-seed means
    pooled = []
    for k, g in df.groupby("n_evals"):
        subs = [logs[s].iloc[:k] for s in g["train_seed"]]
        allraw = np.concatenate([s_["yield_raw"].values for s_ in subs])
        # Pool the underlying batches, so mean/sd here mean the same thing as in
        # summary_final.csv's POOLED row (spread over all held-out batches, NOT the average of
        # the per-seed spreads -- those differ, and sharing a column name while differing would
        # be a trap).
        alleval = np.concatenate(list(g["_eval_yields"]))
        alldelta = np.concatenate(list(g["_eval_deltas"]))
        pooled.append({
            "train_seed": "POOLED", "checkpoint": g["checkpoint"].iloc[0], "n_evals": k,
            "train_mean_raw": float(allraw.mean()), "train_std_raw": float(allraw.std(ddof=1)),
            "train_best": float(g["train_best"].max()),
            "train_n_collapsed": int(sum(s_["collapsed"].sum() for s_ in subs)),
            "eval_mean_raw": float(alleval.mean()),
            "eval_std_raw": float(alleval.std(ddof=1)),
            "eval_best_raw": float(alleval.max()),
            "eval_min_raw": float(alleval.min()),
            "eval_n_collapsed": int(g["eval_n_collapsed"].sum()),
            "eval_collapsed_seeds": "|".join(
                f"{int(r.train_seed)}:{r.eval_collapsed_seeds}"
                for r in g.itertuples() if r.eval_collapsed_seeds != "none") or "none",
            "eval_mean_recipe": float(g["eval_mean_recipe"].iloc[0]),
            "mean_delta_raw": float(alldelta.mean()),
        })
    df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True).sort_values(
        ["n_evals", "train_seed"], kind="stable")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    df.to_csv(out_dir / "checkpoint_table.csv", index=False)

    print("\n=== checkpoint_table.csv ===")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Learning curve in the repo's own episode_holdout_curve.csv format
# ---------------------------------------------------------------------------

# EPISODE AXIS. An MC-PILCO run's episode_holdout_curve.csv is indexed by TRIAL (1..num_trials);
# its num_explorations batches come before trial 1 and are not on that axis. So BO's episode t is
# defined the same way: the schedule it would ship after its n_random random-design batches plus t
# surrogate-guided ones, i.e. after (n_random + t) BO evaluations. Episode t therefore costs both
# methods exactly the same number of simulated batches, which is what makes the shared x-axis a
# fair sample-efficiency comparison.
CURVE_EPISODES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 45, 70, 95)


def curve_runs(out_dir, episodes, eval_seeds):
    """Emit one episode_holdout_curve.csv per training seed, consumable by
    episode_holdout_multiseed.py as just another --group.

    Layout mirrors a real run directory so no plotting code has to change:
        <out_dir>/curve_runs/seed<S>/episode_holdout_curve.csv
    """
    out_dir = Path(out_dir)
    logs = {int(f.stem.replace("bo_log_seed", "")): pd.read_csv(f)
            for f in sorted(out_dir.glob("bo_log_seed*.csv"))}
    if not logs:
        raise SystemExit(f"no bo_log_seed*.csv in {out_dir} -- run the searches first")
    metas = {}
    for f in out_dir.glob("meta_seed*.json"):
        m = json.loads(f.read_text())
        metas[int(m["train_seed"])] = m

    seg = Segmentation(next(iter(metas.values())).get("segment_hours", 25.0))
    print(f"recipe reference on {len(eval_seeds)} held-out batches ...", flush=True)
    recipe = np.array([simulate(seg, {}, h)["yield_raw"] for h in eval_seeds])
    print(f"  recipe mean {recipe.mean():.1f} kg", flush=True)

    written = []
    for s, lg in sorted(logs.items()):
        meta = metas.get(s, {})
        n_r = int(meta.get("n_random", 5))
        cols = seg.factor_columns()
        # The best-so-far schedule only changes a handful of times over 100 calls; replaying an
        # unchanged schedule would re-simulate identical batches, so memoise on the winning call.
        cache = {}
        rows = []
        for t in episodes:
            k = n_r + t
            if k > len(lg):
                continue
            sub = lg.iloc[:k]
            best_row = sub.loc[sub["yield_gated"].idxmax()]
            call = int(best_row["call"])
            if call not in cache:
                fac = seg.unpack(best_row[cols].values)
                cache[call] = np.array([simulate(seg, fac, h)["yield_raw"] for h in eval_seeds])
                print(f"  seed {s} episode {t:3d} (eval {k:3d}): new best from call {call} "
                      f"-> held-out mean {cache[call].mean():.1f} kg", flush=True)
            ys = cache[call]
            rows.append({
                "episode": t,
                "mean_yield": float(ys.mean()),
                "std_yield": float(ys.std(ddof=1)),
                "mean_delta_vs_recipe": float(ys.mean() - recipe.mean()),
                "worst_delta": float((ys - recipe).min()),
                **{f"yield_seed_{h}": float(y) for h, y in zip(eval_seeds, ys)},
            })

        run_dir = out_dir / "curve_runs" / f"seed{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(run_dir / "episode_holdout_curve.csv", index=False)
        written.append(str(run_dir))
        print(f"  wrote {run_dir}/episode_holdout_curve.csv ({len(rows)} episodes, "
              f"{len(cache)} distinct schedules replayed)")

    print("\nUse as a group in episode_holdout_multiseed.py:")
    print(f'  --group "BO={",".join(written)}"')
    return written


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------

def plot_holdout(hold, out_dir):
    seeds = sorted(hold["train_seed"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    lo = min(hold["yield_recipe"].min(), hold["yield_bo"].min())
    hi = max(hold["yield_recipe"].max(), hold["yield_bo"].max())
    pad = 0.05 * (hi - lo)
    for i, s in enumerate(seeds):
        d = hold[hold["train_seed"] == s]
        ok = ~d["bo_collapsed"].astype(bool)
        ax[0].scatter(d.loc[ok, "yield_recipe"], d.loc[ok, "yield_bo"],
                      color=f"C{i}", label=f"train {s}", s=55)
        ax[0].scatter(d.loc[~ok, "yield_recipe"], d.loc[~ok, "yield_bo"],
                      color=f"C{i}", s=90, marker="x", linewidths=2.5)
    ax[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", ls="--", label="y = x")
    ax[0].set(xlabel="recipe yield (kg)", ylabel="BO yield (kg)",
              title="Held-out paired yield (x = collapsed batch)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    width = 0.8 / len(seeds)
    evals = sorted(hold["eval_seed"].unique())
    xs = np.arange(len(evals))
    for i, s in enumerate(seeds):
        d = hold[hold["train_seed"] == s].set_index("eval_seed").loc[evals]
        ax[1].bar(xs + i * width - 0.4 + width / 2, d["delta"].values, width,
                  color=f"C{i}", label=f"train {s}")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([str(int(e)) for e in evals], rotation=30)
    ax[1].set(xlabel="held-out batch", ylabel="delta BO - recipe (kg)",
              title="Per-batch delta vs recipe")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "holdout_paired.png", dpi=150)
    plt.close(fig)


def plot_train_vs_holdout(sdf, out_dir):
    d = sdf[sdf["train_seed"] != "POOLED"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.28, d["train_best_yield_gated"], 0.26, label="in-sample best (training batch)",
           color="C1")
    ax.bar(x, d["eval_mean_bo_raw"], 0.26, yerr=d["eval_std_bo_raw"], capsize=4,
           label="held-out mean (BO, raw)", color="C0")
    ax.bar(x + 0.28, d["eval_mean_recipe"], 0.26, yerr=d["eval_std_recipe"], capsize=4,
           label="held-out mean (recipe)", color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"train {s}" for s in d["train_seed"]])
    ax.set(ylabel="yield (kg)",
           title="What BO finds in hindsight vs what transfers (bars: mean +/- s.d. over 5 batches)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "train_vs_holdout.png", dpi=150)
    plt.close(fig)


def plot_collapse(sdf, out_dir):
    d = sdf[sdf["train_seed"] != "POOLED"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(x, d["train_collapse_rate"] * 100, 0.5, color="crimson", alpha=.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"train {s}" for s in d["train_seed"]])
    ax[0].set(ylabel="% of search batches collapsed",
              title="Envelope breaches during BO search")
    ax[0].grid(alpha=.3, axis="y")

    ax[1].bar(x, d["eval_n_collapsed"], 0.5, color="crimson", alpha=.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"train {s}" for s in d["train_seed"]])
    ax[1].set(ylabel=f"collapsed held-out batches (of {int(d['n_eval_batches'].iloc[0])})",
              title="Envelope breaches when the winning schedule is replayed")
    ax[1].set_ylim(0, max(1, int(d["n_eval_batches"].iloc[0])))
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "collapse_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_seeds", type=int, nargs="+", default=list(TRAIN_SEEDS),
                   help="simulator seeds BO tunes on, one search each (default: 4000 5000 6000, "
                        "i.e. the first training batch of --seed 4/5/6)")
    p.add_argument("--eval_seeds", type=int, nargs="+", default=list(EVAL_SEEDS),
                   help="held-out batches the winning schedule is replayed on "
                        "(default: 700000-700004, the RL arm's block)")
    p.add_argument("--segment_hours", type=float, default=25.0,
                   help="length of one open-loop decision segment (default 25 h = 10 dims)")
    p.add_argument("--n_calls", type=int, default=100, help="BO evaluations per training seed")
    p.add_argument("--n_random", type=int, default=5,
                   help="random initial design points (matches MC-PILCO's 5 explorations in "
                        "role, not in budget -- see module docstring)")
    p.add_argument("--out_dir", type=str, default=str(Path(_ROOT) / "results" / "bo_baseline"))
    p.add_argument("--aggregate", action="store_true",
                   help="skip the searches, just rebuild summaries/plots from existing CSVs")
    p.add_argument("--skip_existing", action="store_true",
                   help="skip a training seed whose holdout CSV already exists")
    p.add_argument("--checkpoints", type=int, nargs="*", default=None,
                   help="build checkpoint_table.csv: for each budget k given here, replay the "
                        "best-feasible-so-far schedule after k BO evaluations on the held-out "
                        "block. Bare flag defaults to 15 100 (15 = 5 explorations + 10 trials, "
                        "budget parity with an MC-PILCO run; 100 = the full BO budget). Runs "
                        "len(checkpoints) x n_seeds x 5 extra simulations.")
    p.add_argument("--curve", type=int, nargs="*", default=None,
                   help="emit episode_holdout_curve.csv per training seed under "
                        "<out_dir>/curve_runs/seed<S>/, so BO can be plotted by "
                        "episode_holdout_multiseed.py as another --group. Optional list of "
                        "EPISODES (default: 1-10, 15, 25, 45, 70, 95), where episode t means "
                        "'after n_random random-design batches + t guided ones' -- the same batch "
                        "count as an MC-PILCO run's trial t. Pair with --eval_seeds so the block "
                        "matches the MC-PILCO curves (they default to 700000-700009).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.curve is not None:
        curve_runs(out_dir, args.curve or list(CURVE_EPISODES), args.eval_seeds)
        return

    if args.checkpoints is not None:
        checkpoint_table(out_dir, args.checkpoints or [15, 100], args.eval_seeds)
        return

    if not args.aggregate:
        seg = Segmentation(args.segment_hours)
        for s in args.train_seeds:
            if args.skip_existing and (out_dir / f"holdout_seed{s}.csv").exists():
                print(f"skip train seed {s} (holdout CSV exists)")
                continue
            run_seed(seg, s, args.eval_seeds, args.n_calls, args.n_random, out_dir)

    aggregate(out_dir, args.eval_seeds)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()

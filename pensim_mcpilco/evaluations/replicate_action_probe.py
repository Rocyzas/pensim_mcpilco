"""
PYTHONPATH=.. python -m evaluations.replicate_action_probe seedX_Y
    [--n_replicates 20] [--base_level 0.0] [--delta 1.0] [--horizons 1,20]
    [--seed_base 810000] [--results_root <path>]
    [--setup {single_phase_baseline,single_phase_baseline_time,dual_phase_baseline,dual_phase_baseline_time}]

The "sharper" paired-action probe -- companion to replicate_noise_probe.py's "clean" noise-only
variant, and the measurement that actually answers the deafness question that script cannot.
replicate_noise_probe.py tells you the real noise floor and whether the GP's own band is
calibrated against it; it says nothing about whether an ACTION has a real, recoverable effect,
because it never contrasts two different actions.

This script does: for each probe step j, it runs N PAIRED real-simulator replicates of recipe A
(base_level, held constant) vs recipe A-with-a-bump (base_level+delta from decision j onward),
sharing the SAME seed within each pair -- exactly action_sensitivity_baseline.py's own
_rollout_base/_rollout_pert (reused unchanged here) -- so the shared process/batch-parameter
noise cancels in the per-replicate difference s_pert_i - s_base_i, instead of adding variance
the way two independently-seeded averages would. Averaging that paired difference across
replicates and dividing by its own standard error (SEM, which shrinks as 1/sqrt(N) -- exactly
the noise-vs-signal test the "clean" probe's docstring describes) gives a z-score per
(j, channel, horizon): |z| clearing ~1.96 means the averaged action effect is real and
recoverable at that probe point; a z-score buried in the noise means the action has no
detectable effect there, however large the raw wiggle looks in a single trajectory.

Every j is bucketed by batch_phase (early/mid/late, via action_sensitivity_baseline.phase_of)
and, for dual-phase setups, ALSO by model_phase (phase1/phase2 blend-routing, from the run's
own pivot_hours) -- so the headline question this answers directly is: does the action effect
clear the noise floor in phase 2? If yes, phase-2 is controllable and excitation/gray-box work
on it is worth pursuing; if it never clears noise there, phase-2 is genuinely uncontrollable
per-step and chasing per-step action sensitivity there is not the fix.

Covers all four *_baseline setups exactly like replicate_noise_probe.py (--setup); the real
simulator side (_rollout_base/_rollout_pert) is phase-agnostic so it's reused unchanged for
both single- and dual-phase runs.
"""
import argparse
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import evaluations.eval_single_phase_lib as single_lib
import evaluations.eval_multi_phase_lib as multi_lib
from evaluations.action_sensitivity_baseline import _rollout_base, _rollout_pert, J_GRID_SHARED, phase_of
from mcpilco.config_single_phase_baseline import get_config as _single_no_time_get_config
from mcpilco.config_single_phase_baseline_time import get_config as _single_time_get_config
from mcpilco.config_dual_phase_baseline import get_config as _dual_no_time_get_config
from mcpilco.config_dual_phase_baseline_time import get_config as _dual_time_get_config
from mcpilco.pensim_wrapper import STATE_NAMES, CONTROL_H, T_SAMPLING

# Same SETUPS shape as replicate_noise_probe.py -- see that file's own comment for why the
# wrong get_config_fn/library is dangerous (silent GP reconstruction mismatch or an immediate
# shape-mismatch crash).
SETUPS = {
    "single_phase_baseline":      (single_lib, _single_no_time_get_config, Path(_ROOT) / "results" / "single_phase_baseline"),
    "single_phase_baseline_time": (single_lib, _single_time_get_config,    Path(_ROOT) / "results" / "single_phase_baseline_time"),
    "dual_phase_baseline":        (multi_lib,  _dual_no_time_get_config,  Path(_ROOT) / "results" / "dual_phase_baseline"),
    "dual_phase_baseline_time":   (multi_lib,  _dual_time_get_config,     Path(_ROOT) / "results" / "dual_phase_baseline_time"),
}

# Reserved seed block for this script's replicate rollouts -- distinct from
# replicate_noise_probe.py's 800000, action_sensitivity_baseline.py's own 424242+i block,
# eval_base=700000 (A1 held-out), MEASUREMENT_SEED_BASE=900_000, and training's seed*1000.
DEFAULT_SEED_BASE = 810000

Z_THRESH = 1.96  # |mean paired delta| / SEM must clear this to count as a real, recoverable effect


def model_phase_of(j, pivot_step):
    return "phase1" if j < pivot_step else "phase2"


def main(run_id, n_replicates=20, base_level=0.0, delta=1.0,
        horizons=(1, 20), seed_base=DEFAULT_SEED_BASE, results_root=None,
        setup="single_phase_baseline"):
    if setup not in SETUPS:
        raise ValueError(f"--setup must be one of {list(SETUPS)}, got '{setup}'")
    lib, get_config_fn, default_root = SETUPS[setup]
    results_root = default_root if results_root is None else results_root
    run = lib.load_run(run_id, get_config_fn=get_config_fn, results_root=results_root)
    out_dir = run.dir
    # tag output filenames by (base_level, delta) so re-running at a different bump size (e.g.
    # comparing delta=1.0's easy case against a smaller, more decisive delta=0.2) doesn't
    # silently overwrite the previous run's results in the same folder.
    tag = f"a{base_level:g}_d{delta:g}".replace(".", "p").replace("-", "neg")

    is_dual = setup.startswith("dual_phase")
    pivot_step = None
    if is_dual:
        pivot_step = int(round(run.pivot_hours / T_SAMPLING))
        print(f"pivot_step={pivot_step} (pivot_hours={run.pivot_hours:g})")

    n_decisions = int(CONTROL_H / T_SAMPLING)
    js = sorted(J_GRID_SHARED)
    horizons = sorted(horizons)
    max_h = max(horizons)
    seeds = [seed_base + i for i in range(n_replicates)]

    print(f"\n----- {n_replicates} PAIRED replicates per probe j (recipe A={base_level} vs "
         f"A+bump={base_level + delta} from j onward), seeds {seed_base}..{seed_base + n_replicates - 1} "
         f"-----")

    # base_level is constant for the whole batch regardless of j -- roll it once per seed and
    # reuse across every j, instead of N_replicates * len(js) redundant base rollouts.
    base_cache = {}
    def get_base(seed):
        if seed not in base_cache:
            base_cache[seed] = _rollout_base(seed, base_level)
        return base_cache[seed]

    z_rows = []
    for j in js:
        bphase = phase_of(j, n_decisions)
        mphase = model_phase_of(j, pivot_step) if is_dual else None

        deltas = []
        for seed in seeds:
            s_base = get_base(seed)
            s_pert = _rollout_pert(seed, j, delta, base_level)
            end = min(j + max_h + 1, s_base.shape[0])
            deltas.append(s_pert[j:end] - s_base[j:end])
        min_len = min(d.shape[0] for d in deltas)
        deltas = np.stack([d[:min_len] for d in deltas])  # (N, steps<=max_h+1, STATE_DIM)

        for k in horizons:
            if k >= min_len:
                continue
            dk = deltas[:, k, :]  # (N, STATE_DIM): paired delta at horizon k past the bump
            mean_dk = dk.mean(axis=0)
            sem_dk = dk.std(axis=0, ddof=1) / np.sqrt(n_replicates)
            z = np.divide(mean_dk, sem_dk, out=np.full_like(mean_dk, np.nan), where=sem_dk > 0)
            for c_idx, name in enumerate(STATE_NAMES):
                zc = float(z[c_idx])
                clears = bool(np.isfinite(zc) and abs(zc) > Z_THRESH)
                z_rows.append({
                    "j": j, "batch_phase": bphase, "model_phase": mphase, "horizon_k": k,
                    "channel": name, "mean_delta": float(mean_dk[c_idx]),
                    "sem_delta": float(sem_dk[c_idx]), "z": zc, "clears_noise_floor": clears,
                })

    df = pd.DataFrame(z_rows)
    df.to_csv(Path(out_dir) / f"replicate_action_probe_{tag}.csv", index=False)

    # ---- aggregate + verdict ----
    channels = [c for c in STATE_NAMES if c != "time"]
    group_cols = ["model_phase"] if is_dual else ["batch_phase"]
    group_label = "model_phase (phase1/phase2 routing)" if is_dual else "batch_phase (early/mid/late)"
    summary_rows = []
    print(f"\n----- fraction of probe j's clearing |z|>{Z_THRESH} (real, recoverable action "
         f"effect), by {group_label} -----")
    for k in horizons:
        sub_k = df[df["horizon_k"] == k]
        print(f"\nhorizon k={k}:")
        header = f"{'phase':>8} " + " ".join(f"{c:>10}" for c in channels)
        print(header)
        for phase, g in sub_k.groupby(group_cols[0]):
            fracs = []
            for c in channels:
                gc = g[g["channel"] == c]
                frac = float(gc["clears_noise_floor"].mean()) if len(gc) else float("nan")
                fracs.append(frac)
                summary_rows.append({"horizon_k": k, group_cols[0]: phase, "channel": c,
                                    "frac_clearing_noise_floor": frac, "n_probes": len(gc)})
            print(f"{phase:>8} " + " ".join(f"{f:>10.2f}" for f in fracs))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(out_dir) / f"replicate_action_probe_summary_{tag}.csv", index=False)

    if is_dual:
        p2 = summary_df[summary_df["model_phase"] == "phase2"]
        print("\n----- VERDICT: phase 2 action-effect identifiability -----")
        for k in horizons:
            p2k = p2[p2["horizon_k"] == k]
            for _, row in p2k.iterrows():
                verdict = ("REAL & recoverable" if row["frac_clearing_noise_floor"] > 0.5 else
                          "buried in noise -- not identifiable per-step")
                print(f"  k={k:>3}  {row['channel']:>10}: "
                     f"{row['frac_clearing_noise_floor']:.2f} of phase-2 probes clear the "
                     f"noise floor -> {verdict}")

    # ---- plot: fraction clearing noise floor, grouped bars by phase, one subplot per horizon ----
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.5 * len(horizons), 4.5), squeeze=False)
    phases_order = (["phase1", "phase2"] if is_dual else ["early", "mid", "late"])
    x = np.arange(len(channels))
    width = 0.8 / len(phases_order)
    for ax_idx, k in enumerate(horizons):
        ax = axes[0, ax_idx]
        sub = summary_df[summary_df["horizon_k"] == k]
        for i, ph in enumerate(phases_order):
            vals = []
            for c in channels:
                row = sub[(sub[group_cols[0]] == ph) & (sub["channel"] == c)]
                vals.append(float(row["frac_clearing_noise_floor"].iloc[0]) if len(row) else float("nan"))
            ax.bar(x + (i - (len(phases_order) - 1) / 2) * width, vals, width, label=ph)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(channels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"fraction of probes with |z|>{Z_THRESH}")
        ax.set_title(f"horizon k={k} steps after the bump")
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle(f"Paired action-effect identifiability - {run.dir.name} "
                f"(delta={delta}, n_replicates={n_replicates}, by {group_label})")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"replicate_action_probe_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n----- results written to {out_dir} -----")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", type=str,
                   help="run to probe, e.g. 'seed3_1' (resolved under results/<setup>/, see "
                        "--setup) or a full/relative path to a run folder")
    p.add_argument("--n_replicates", type=int, default=20,
                   help="number of PAIRED (recipe A, recipe A+bump) replicate rollouts per probe j")
    p.add_argument("--base_level", type=float, default=0.0,
                   help="constant Fs-scale action for recipe A (0.0 = plain recipe/PID schedule)")
    p.add_argument("--delta", type=float, default=1.0,
                   help="bump size: recipe A+bump = base_level+delta from decision j onward "
                        "(default 1.0 = the largest bump the simulator can take, for maximum "
                        "signal-to-noise)")
    p.add_argument("--horizons", type=str, default="1,20",
                   help="comma-separated decision-steps-after-j to test (default matches "
                        "action_sensitivity_baseline.py's HORIZONS=[1,20]: immediate + sustained)")
    p.add_argument("--seed_base", type=int, default=DEFAULT_SEED_BASE,
                   help="first seed for the N replicate pairs (uses seed_base.."
                        "seed_base+n_replicates-1)")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under (default: results/<setup>/)")
    p.add_argument("--setup", choices=list(SETUPS), default="single_phase_baseline",
                   help="which *_baseline variant this run was trained with -- see "
                        "replicate_noise_probe.py's --setup help for the full explanation "
                        "(same four options, same hazards from picking the wrong one)")
    args = p.parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(","))
    main(run_id=args.run_id, n_replicates=args.n_replicates,
        base_level=args.base_level, delta=args.delta, horizons=horizons,
        seed_base=args.seed_base, results_root=args.results_root, setup=args.setup)

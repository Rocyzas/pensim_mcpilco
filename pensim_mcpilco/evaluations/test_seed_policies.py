"""
PYTHONPATH=.. python -m evaluations.test_seed_policies seedX_Y [--setup <setup>]
  (or, from the evaluations/ folder:  python test_seed_policies.py seedX_Y)
  optional: --n_eval_seeds 5  --eval_base 700000

Per-EPISODE held-out learning curve. Loads the policy saved at EVERY training trial of a run
(trial 1 .. trial N -- exploration episodes have no learned policy) and evaluates each on the SAME
fixed block of held-out seeds, to see whether held-out yield was still climbing at the final
episode (=> more training episodes would likely have helped) or had already plateaued/peaked
earlier (=> the extra episodes added nothing, or even hurt).

Works on ALL FOUR setups -- single- and dual-phase, prior-mean and plain-RBF baseline. It
auto-detects which results tree <run_id> lives in and binds the MATCHING evaluation library +
config, so the reconstructed policy/model matches how the run was trained:
  single_phase / single_phase_baseline -> eval_single_phase_lib + config_single_phase[_baseline]
  dual_phase   / dual_phase_baseline   -> eval_multi_phase_lib  + config_dual_phase[_baseline]
Both libraries expose the same helper names (load_run / build_policy_agent / load_stage_policy /
run_arm / yield_kg / n_trials_in_log), so the per-episode loop below is identical -- only which
library `lib` points at changes.

If the same run-id exists in more than one tree (e.g. a 'seed4_1' under BOTH single_phase_baseline
and dual_phase_baseline), pass --setup to disambiguate. Writes episode_holdout_curve.{csv,png} into
the run's own folder and prints a verdict.
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
from mcpilco.config_single_phase import get_config as _single_cfg
from mcpilco.config_single_phase_baseline import get_config as _single_baseline_cfg
from mcpilco.config_single_phase_baseline_time import get_config as _single_baseline_time_cfg
from mcpilco.config_dual_phase import get_config as _dual_cfg
from mcpilco.config_dual_phase_baseline import get_config as _dual_baseline_cfg
from mcpilco.config_dual_phase_baseline_time import get_config as _dual_baseline_time_cfg
from mcpilco.config_dual_phase_baseline_priors import get_config as _dual_baseline_priors_cfg
from mcpilco.config_single_phase_absolute import get_config as _single_absolute_cfg
from mcpilco.config_single_phase_absolute_time import get_config as _single_absolute_time_cfg

_RESULTS = Path(_ROOT) / "results"

# setup name -> (evaluation library module, get_config fn, results-tree folder name). Single- and
# dual-phase use different libs but the SAME helper names, so the loop in main() is lib-agnostic.
SETUPS = {
    "single_phase":               (single_lib, _single_cfg,               "single_phase"),
    "single_phase_baseline":      (single_lib, _single_baseline_cfg,      "single_phase_baseline"),
    "single_phase_baseline_time": (single_lib, _single_baseline_time_cfg, "single_phase_baseline_time"),
    "dual_phase":                 (multi_lib,  _dual_cfg,                  "dual_phase"),
    "dual_phase_baseline":        (multi_lib,  _dual_baseline_cfg,        "dual_phase_baseline"),
    "dual_phase_baseline_time":   (multi_lib,  _dual_baseline_time_cfg,   "dual_phase_baseline_time"),
    "dual_phase_baseline_priors": (multi_lib,  _dual_baseline_priors_cfg, "dual_phase_baseline_priors"),
    # Absolute-action single-phase arms (see mcpilco/config_single_phase_absolute.py). This
    # script's per-episode held-out sweep is encoding-agnostic -- it re-rolls saved policies
    # through whatever wrapper the config builds -- but the config MUST be the matching absolute
    # one: an absolute run's note.txt carries action_mode/fs_abs_*, so a residual get_config
    # raises TypeError rather than silently re-simulating under the wrong action semantics.
    #
    # ONE behavioural note, additive entries notwithstanding: the bare-run-id branch of _resolve
    # below scans EVERY tree in this table, so once results/single_phase_absolute[_time]/ exists
    # a bare id present in both it and an older tree now reports "exists in multiple setups;
    # pass --setup" where it previously resolved silently. That is the table's intended
    # disambiguation and it fails loudly rather than picking the wrong tree -- but it is the one
    # way adding these rows can change an existing command's behaviour.
    "single_phase_absolute":      (single_lib, _single_absolute_cfg,      "single_phase_absolute"),
    "single_phase_absolute_time": (single_lib, _single_absolute_time_cfg, "single_phase_absolute_time"),
}


def _resolve(run_id, setup=None, results_root=None):
    """Return (lib, get_config_fn, results_root, setup_name). results_root is None when a
    filesystem path was given (load_run resolves the path itself). Raises if a bare run-id is
    missing or ambiguous across trees -- in which case pass --setup.

    results_root overrides the default results/<folder> root for the chosen setup (same
    convention as the four evaluations_*.py scripts' own --results_root: it becomes the
    DIRECT parent of run subfolders, not results_root/folder). Since a custom folder's name
    can't be auto-matched against all 6 SETUPS conventions, it requires --setup too."""
    s = str(run_id)
    is_path = _os.path.sep in s or s.endswith(".pkl") or Path(s).is_absolute()

    if results_root is not None and setup is None:
        raise ValueError(
            "--results_root requires --setup (can't auto-detect which eval library/config "
            "a custom results folder uses)")

    # explicit --setup always wins
    if setup is not None:
        if setup not in SETUPS:
            raise ValueError(f"--setup must be one of {list(SETUPS)}, got '{setup}'")
        lib, cfg, folder = SETUPS[setup]
        if results_root is not None:
            return lib, cfg, (None if is_path else Path(results_root)), setup
        return lib, cfg, (None if is_path else _RESULTS / folder), setup

    # a path: infer the tree from the run folder's parent directory name
    if is_path:
        parent = Path(s.rstrip("/")).parent.name
        if parent in SETUPS:
            lib, cfg, _ = SETUPS[parent]
            return lib, cfg, None, parent
        raise ValueError(
            f"could not infer setup from path '{s}' (parent dir '{parent}' is not one of "
            f"{list(SETUPS)}); pass --setup explicitly.")

    # a bare run-id: find which tree(s) actually contain it
    matches = [name for name, (_, _, folder) in SETUPS.items() if (_RESULTS / folder / s).exists()]
    if len(matches) == 1:
        lib, cfg, folder = SETUPS[matches[0]]
        return lib, cfg, _RESULTS / folder, matches[0]
    if not matches:
        raise FileNotFoundError(
            f"'{run_id}' not found under any of "
            f"{[str(_RESULTS / SETUPS[n][2]) for n in SETUPS]}. Pass a full path or a valid run name.")
    raise ValueError(
        f"'{run_id}' exists in multiple setups {matches}; pass --setup <name> to disambiguate.")


def main(run_id, n_eval_seeds=5, eval_base=700000, setup=None, results_root=None):
    lib, get_config_fn, results_root, setup_name = _resolve(run_id, setup=setup,
                                                             results_root=results_root)
    run = lib.load_run(run_id, get_config_fn=get_config_fn, results_root=results_root)
    out_dir = run.dir
    held_out = [eval_base + i for i in range(n_eval_seeds)]
    n_stages = run.n_trials_in_log
    print(f"\nrun {run.dir.name} [{setup_name}]: {n_stages} policy episodes (trials) | "
          f"held-out seeds {held_out}")

    # eval_wrapper (+ policy-load sanity check) from the shared helper; we compute our own recipe
    # baselines on the HELD-OUT seeds (build_policy_agent's ref is on the train seed only).
    _, eval_wrapper, _, _, _ = lib.build_policy_agent(run)

    # recipe baseline per held-out seed -- policy-independent, so evaluated once and reused for
    # every episode's paired delta (this is what makes the per-episode curves comparable).
    base_by_seed = {s: lib.yield_kg(lib.run_arm(eval_wrapper, s, policy=None, pid_baseline=True))
                    for s in held_out}
    base_mean = float(np.mean(list(base_by_seed.values())))
    print(f"recipe baseline (mean over held-out seeds): {base_mean:.1f} kg")

    # every episode's policy, evaluated on the SAME held-out block
    rows = []
    ys_by_seed = {s: [] for s in held_out}
    for k in range(1, n_stages + 1):
        pol = lib.load_stage_policy(run, k)
        ys = np.array([lib.yield_kg(lib.run_arm(eval_wrapper, s, policy=pol, pid_baseline=False))
                       for s in held_out])
        ds = ys - np.array([base_by_seed[s] for s in held_out])
        for s, y in zip(held_out, ys):
            ys_by_seed[s].append(float(y))
        row = {"episode": k, "mean_yield": float(ys.mean()), "std_yield": float(ys.std()),
               "mean_delta_vs_recipe": float(ds.mean()), "worst_delta": float(ds.min())}
        for s, y in zip(held_out, ys):
            row[f"yield_seed_{s}"] = float(y)
        rows.append(row)
        print(f"  episode {k:>2}: held-out yield = {ys.mean():8.1f} +/- {ys.std():5.1f} kg | "
              f"paired delta vs recipe = {ds.mean():+8.1f} kg")

    df = pd.DataFrame(rows)
    csv_path = Path(out_dir) / "episode_holdout_curve.csv"
    df.to_csv(csv_path, index=False)

    # ---- plot: per-seed + mean held-out yield, and paired delta, over episodes ----
    episodes = df["episode"].values
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    seed_colors = plt.cm.tab10(np.linspace(0, 1, len(held_out)))
    for s, c in zip(held_out, seed_colors):
        ax[0].plot(episodes, ys_by_seed[s], color=c, alpha=.5, lw=1, label=f"seed {s}")
    ax[0].plot(episodes, df["mean_yield"], marker="o", color="k", lw=2, label="mean over seeds")
    ax[0].axhline(base_mean, color="crimson", ls="--", lw=1.5, label="recipe (same seeds)")
    ax[0].set_xlabel("training episode (policy trial)"); ax[0].set_ylabel("held-out batch yield (kg)")
    ax[0].set_title(f"Per-episode held-out yield - {run.dir.name} [{setup_name}]")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)

    ax[1].axhline(0.0, color="crimson", ls="--", lw=1.5, label="recipe (paired)")
    ax[1].plot(episodes, df["mean_delta_vs_recipe"], marker="o", color="k", lw=2, label="mean delta")
    ax[1].fill_between(episodes, df["mean_delta_vs_recipe"] - df["std_yield"],
                       df["mean_delta_vs_recipe"] + df["std_yield"], alpha=.15, color="k")
    ax[1].set_xlabel("training episode (policy trial)"); ax[1].set_ylabel("paired yield delta vs recipe (kg)")
    ax[1].set_title("Did held-out performance keep improving?")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"Episode-by-episode held-out policy sweep - {run.dir.name} [{setup_name}]")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "episode_holdout_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- verdict: was the curve still rising at the last episode? ----
    peak_ep = int(df.loc[df["mean_yield"].idxmax(), "episode"])
    peak_val = float(df["mean_yield"].max())
    final_val = float(df["mean_yield"].iloc[-1])
    print(f"\n----- results written to {out_dir} -----")
    print(f"peak held-out yield: {peak_val:.1f} kg @ episode {peak_ep} of {n_stages}")
    print(f"final episode yield: {final_val:.1f} kg (delta vs recipe {df['mean_delta_vs_recipe'].iloc[-1]:+.1f} kg)")
    if peak_ep == n_stages:
        print("VERDICT: held-out yield was still at its best on the FINAL episode -> more "
              "training episodes may have helped (curve not yet plateaued).")
    elif peak_val - final_val > df["std_yield"].mean():
        print(f"VERDICT: held-out yield PEAKED at episode {peak_ep} and dropped by "
              f"{peak_val - final_val:.1f} kg by the final episode -> the last "
              f"{n_stages - peak_ep} episode(s) did not help (over-training / instability).")
    else:
        print(f"VERDICT: held-out yield plateaued by episode {peak_ep} (final within noise of the "
              f"peak) -> the extra episodes were neither clearly helpful nor harmful.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", type=str,
                   help="run to sweep, e.g. 'seed3_5' (auto-resolved under results/{single_phase,"
                        "single_phase_baseline,dual_phase,dual_phase_baseline}) or a path to a run folder")
    p.add_argument("--setup", choices=list(SETUPS), default=None,
                   help="force which setup/config to use; required only if the run-id exists in "
                        "more than one results tree, or whenever --results_root is given")
    p.add_argument("--n_eval_seeds", type=int, default=5,
                   help="number of held-out seeds (block is eval_base..eval_base+n_eval_seeds-1)")
    p.add_argument("--eval_base", type=int, default=700000, help="first held-out seed")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under (default: "
                        "results/<setup>/); requires --setup, since a custom folder's name "
                        "can't be auto-matched to one of the 6 setups")
    args = p.parse_args()
    main(run_id=args.run_id, n_eval_seeds=args.n_eval_seeds, eval_base=args.eval_base,
        setup=args.setup, results_root=args.results_root)

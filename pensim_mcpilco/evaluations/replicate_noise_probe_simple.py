"""
PYTHONPATH=.. python -m evaluations.replicate_noise_probe_simple seedX_Y [--gp_trial k]
    [--n_replicates 20] [--base_level 0.0] [--seed_base 800000] [--results_root <path>]
    [--setup {single_phase_baseline,single_phase_baseline_time,dual_phase_baseline,dual_phase_baseline_time}]

Presentation variant of replicate_noise_probe.py's replicate_noise_vs_gp_band.png. SAME
computation, unchanged -- real-simulator replicate rollouts of a fixed recipe vs GP particle
rollouts under that same recipe from the same initial state, both reused directly from
replicate_noise_probe.py (SETUPS, DEFAULT_SEED_BASE, _gp_particle_rollout_constant_action) and
action_sensitivity_baseline.py (_rollout_base). Only the figure differs:

  1. `time` is dropped from the panel row (deterministic/uninformative here -- same CHANNELS
     convention the action_deafness_*.py scripts use).
  2. One shared legend at the bottom of the figure, instead of a legend on only the first panel.
  3. The title drops run name / trial / base_level -- keeps only n_replicates, for a cleaner,
     less run-specific caption.

Writes replicate_noise_vs_gp_band_simple.{png,csv} -- deliberately different filenames from
replicate_noise_probe.py's own outputs, so running both against the same run directory never
overwrites either one's output.
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

from evaluations.action_sensitivity_baseline import _rollout_base
from evaluations.replicate_noise_probe import (
    SETUPS, DEFAULT_SEED_BASE, _gp_particle_rollout_constant_action)
from mcpilco.pensim_wrapper import STATE_NAMES

CHANNELS = [c for c in STATE_NAMES if c != "time"]
CHANNEL_IDX = [STATE_NAMES.index(c) for c in CHANNELS]


def main(run_id, gp_trial=None, n_replicates=20, base_level=0.0, seed_base=DEFAULT_SEED_BASE,
        results_root=None, setup="single_phase_baseline"):
    if setup not in SETUPS:
        raise ValueError(f"--setup must be one of {list(SETUPS)}, got '{setup}'")
    lib, baseline_get_config, default_root = SETUPS[setup]
    results_root = default_root if results_root is None else results_root
    run = lib.load_run(run_id, get_config_fn=baseline_get_config, results_root=results_root)
    out_dir = run.dir
    gp_agent, gp_idx = lib.reconstruct_gp_agent(run, idx=gp_trial, get_config_fn=baseline_get_config)
    print(f"reconstructed GP model @ trial {gp_idx}")

    print(f"\n----- rolling {n_replicates} real-simulator replicates of the plain recipe "
         f"(base_level={base_level}, seeds {seed_base}..{seed_base + n_replicates - 1}) -----")
    real_trajs = np.stack([_rollout_base(seed_base + i, base_level) for i in range(n_replicates)])
    print(f"real_trajs shape: {real_trajs.shape}  (N, T, STATE_DIM)")

    init_state = real_trajs[:, 0, :].mean(axis=0)
    n_steps = real_trajs.shape[1] - 1
    n_part = run.cfg["reinforce_par"]["policy_optimization_dict"]["num_particles"]
    print(f"\n----- rolling {n_part} GP particles under the SAME fixed recipe from the same "
         f"initial state -----")
    gp_trajs = _gp_particle_rollout_constant_action(
        gp_agent, init_state, base_level, n_steps, N=n_part)

    t = lib.decision_time_grid(real_trajs.shape[1])
    rows = []
    fig, ax = plt.subplots(1, len(CHANNELS), figsize=(3.4 * len(CHANNELS), 3.8), squeeze=False)
    handles = labels = None
    for j, (k, name) in enumerate(zip(CHANNEL_IDX, CHANNELS)):
        a = ax[0, j]
        real_k = lib._denorm_phys(real_trajs[:, :, k], name)   # (N, T)
        gp_k = lib._denorm_phys(gp_trajs[:, :, k], name)       # (T, N_particles)

        real_mean = real_k.mean(axis=0)
        real_p10, real_p90 = np.percentile(real_k, [10, 90], axis=0)
        gp_mean = gp_k.mean(axis=1)
        gp_p10, gp_p90 = np.percentile(gp_k, [10, 90], axis=1)

        a.fill_between(t, real_p10, real_p90, color="0.3", alpha=.25,
                       label=f"real replicates 10-90% (n={n_replicates})")
        a.plot(t, real_mean, color="0.15", lw=1.8, label="real replicate mean")
        a.fill_between(t, gp_p10, gp_p90, color="C0", alpha=.25,
                       label=f"GP particles 10-90% (n={n_part})")
        a.plot(t, gp_mean, color="C0", lw=1.8, ls="--", label="GP particle mean")
        a.set_title(name, fontsize=9); a.set_xlabel("time (h)", fontsize=8)
        a.grid(alpha=.3); a.tick_params(labelsize=7)
        if handles is None:
            handles, labels = a.get_legend_handles_labels()

        real_halfwidth = float((real_p90 - real_p10).mean() / 2)
        gp_halfwidth = float((gp_p90 - gp_p10).mean() / 2)
        if real_halfwidth == 0.0:
            ratio = float("nan")
            verdict = ("deterministic (no real replicate variance)" if gp_halfwidth == 0.0 else
                      "GP disagrees: predicts variance on a deterministic channel")
        else:
            ratio = gp_halfwidth / real_halfwidth
            verdict = ("well-calibrated" if 0.7 <= ratio <= 1.4 else
                      "OVER-confident (GP band too narrow)" if ratio < 0.7 else
                      "UNDER-confident (GP band too wide)")
        print(f"{name:>10}: real replicate half-width={real_halfwidth:.4f}  "
             f"GP half-width={gp_halfwidth:.4f}  ratio={ratio:.2f}  -> {verdict}")
        rows.append({"channel": name, "real_halfwidth_mean": real_halfwidth,
                    "gp_halfwidth_mean": gp_halfwidth, "gp_over_real_ratio": ratio,
                    "verdict": verdict})

    fig.suptitle(f"Real replicate noise vs GP predicted band (n_replicates={n_replicates})")
    ax[0, 0].legend(handles, labels, loc="lower right", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_dir) / "replicate_noise_vs_gp_band_simple.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "replicate_noise_vs_gp_band_simple.csv", index=False)
    print(f"\n----- results written to {out_dir} -----")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", type=str,
                   help="run to probe, e.g. 'seed3_1' (resolved under results/<setup>/, see "
                        "--setup) or a full/relative path to a run folder")
    p.add_argument("--gp_trial", type=int, default=None,
                   help="which trial's GP model to compare against (default: last saved)")
    p.add_argument("--n_replicates", type=int, default=10,
                   help="number of real-simulator replicate rollouts of the fixed recipe")
    p.add_argument("--base_level", type=float, default=0.0,
                   help="constant Fs-scale action held for the whole batch (0.0 = plain "
                        "recipe/PID schedule, mathematically identical to pid_baseline=True)")
    p.add_argument("--seed_base", type=int, default=DEFAULT_SEED_BASE,
                   help="first seed for the N replicate rollouts (uses seed_base.."
                        "seed_base+n_replicates-1)")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under "
                        "(default: results/<setup>/)")
    p.add_argument("--setup", choices=list(SETUPS), default="single_phase_baseline",
                   help="which *_baseline variant this run was trained with")
    args = p.parse_args()
    main(run_id=args.run_id, gp_trial=args.gp_trial, n_replicates=args.n_replicates,
        base_level=args.base_level, seed_base=args.seed_base, results_root=args.results_root,
        setup=args.setup)

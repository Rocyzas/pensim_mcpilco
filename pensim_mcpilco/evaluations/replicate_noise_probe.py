"""
PYTHONPATH=.. python -m evaluations.replicate_noise_probe seedX_Y [--gp_trial k]
    [--n_replicates 20] [--base_level 0.0] [--seed_base 800000] [--results_root <path>]
    [--setup {single_phase_baseline,single_phase_baseline_time,dual_phase_baseline,dual_phase_baseline_time}]

The "clean" replicate-noise probe: repeats ONE fixed recipe (a constant Fs-scale action held
for the whole batch -- base_level=0.0 by default, which is mathematically identical to the
plain PID/recipe schedule) on the REAL simulator N times, varying only the seed that drives
PenSimWrapper.rollout's own process/batch-parameter randomness (alpha_kla, PAA_c, N_conc_paa,
...) -- noise=None throughout, so no measurement noise is layered on top. Whatever wiggle
survives averaging across replicates is real (input-driven) dynamics; whatever shrinks with N
was noise. That empirical replicate band is then overlaid against the GP's OWN predicted
uncertainty band (N particles propagated under the SAME fixed recipe from the SAME initial
state), answering a question none of the existing diagnostics ask: is the GP's predictive
variance actually calibrated against real replicate variance, or only against its own one-step
teacher-forced residuals on a single recorded trajectory (which is all C4b/C4b_calibration
checks)?

Covers all four *_baseline setups (single- and dual-phase, each with or without time as an
active GP input regressor) via --setup; non-baseline (config_single_phase/config_dual_phase)
is out of scope, matching this script's siblings (evaluations_*_baseline*.py). Reuses
action_sensitivity_baseline.py's _rollout_base for the real-simulator side -- PenSimWrapper.rollout
is identical regardless of single- vs dual-phase, so the same helper works unchanged for both --
and mirrors eval_single_phase_lib/eval_multi_phase_lib's own _particle_rollout for the GP side
(both expose the same helper names, see eval_multi_phase_lib.py's module docstring point about
mirroring eval_single_phase_lib -- this script picks whichever library the chosen --setup uses),
adapted to take an explicit fixed recipe/initial state instead of a historical recorded batch.
"""
import argparse
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import evaluations.eval_single_phase_lib as single_lib
import evaluations.eval_multi_phase_lib as multi_lib
from evaluations.action_sensitivity_baseline import _rollout_base
from mcpilco.config_single_phase_baseline import get_config as _single_no_time_get_config
from mcpilco.config_single_phase_baseline_time import get_config as _single_time_get_config
from mcpilco.config_dual_phase_baseline import get_config as _dual_no_time_get_config
from mcpilco.config_dual_phase_baseline_time import get_config as _dual_time_get_config
from mcpilco.pensim_wrapper import STATE_NAMES, STATE_DIM, ACTION_DIM

# --setup name -> (eval library module, get_config fn, default results root). Mirrors
# test_seed_policies.py's own SETUPS naming and its "both libraries expose the same helper
# names" pattern exactly, restricted to the four *_baseline variants (no/with time as an
# active GP input regressor -- see config_single_phase_baseline_time.py /
# config_dual_phase_baseline_time.py). Passing the WRONG get_config_fn for a run's actual
# active_dims either raises a shape mismatch in load_state_dict or silently reconstructs a GP
# the checkpoint was never fit against -- same hazard eval_single_phase_lib.reconstruct_gp_agent's
# docstring warns about for the Wt/Viscosity prior-mean case. Passing the WRONG library
# (single vs multi) for a dual-phase run fails even louder: DualPhaseModelLearning's gp_list is
# 2*STATE_DIM long, so reconstruct_gp_agent's slicing/state-dict loads mismatch immediately.
SETUPS = {
    "single_phase_baseline":      (single_lib, _single_no_time_get_config, Path(_ROOT) / "results" / "single_phase_baseline"),
    "single_phase_baseline_time": (single_lib, _single_time_get_config,    Path(_ROOT) / "results" / "single_phase_baseline_time"),
    "dual_phase_baseline":        (multi_lib,  _dual_no_time_get_config,  Path(_ROOT) / "results" / "dual_phase_baseline"),
    "dual_phase_baseline_time":   (multi_lib,  _dual_time_get_config,     Path(_ROOT) / "results" / "dual_phase_baseline_time"),
}

# Reserved seed block for this script's replicate rollouts -- clear of eval_base=700000 (A1
# held-out block), MEASUREMENT_SEED_BASE=900_000 (population-level measurement rollouts),
# action_sensitivity_baseline.py's own 424242+i block, and training's seed*1000.
DEFAULT_SEED_BASE = 800000


def _gp_particle_rollout_constant_action(agent, init_state, action_value, n_steps, N, seed=0):
    """N GP particles propagated under a CONSTANT action for n_steps, starting from
    init_state. Same particle-propagation loop as eval_single_phase_lib._particle_rollout /
    eval_multi_phase_lib._particle_rollout (ml.get_next_state(..., particle_pred=True)), but
    parameterised by an explicit fixed action/initial state instead of a historical recorded
    batch -- no existing helper accepts an arbitrary fixed recipe, since C2b's diagnostic only
    ever replays what was recorded.

    For dual-phase, ml is a DualPhaseModelLearning: it blends phase1/phase2 based on an
    internal decision-step counter that must be reset to 0 before a sequential rollout (see
    eval_multi_phase_lib._particle_rollout / PenSimMCPILCOMultiPhase's own docstring) -- single-
    phase's flat model_learning has no such counter, hence the hasattr guard."""
    ml = agent.model_learning
    ml.set_eval_mode()
    torch.manual_seed(seed)
    x = torch.tensor(np.tile(init_state, (N, 1)), dtype=agent.dtype, device=agent.device)
    D = x.shape[1]
    out = np.zeros((n_steps + 1, N, D))
    out[0] = x.detach().cpu().numpy()
    u = torch.full((N, ACTION_DIM), float(action_value), dtype=agent.dtype, device=agent.device)
    if hasattr(ml, "reset_step_counter"):
        ml.reset_step_counter()
    with torch.no_grad():
        for t in range(1, n_steps + 1):
            x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=True)
            out[t] = x.detach().cpu().numpy()
    return out


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
    fig, ax = plt.subplots(1, STATE_DIM, figsize=(3.4 * STATE_DIM, 3.6), squeeze=False)
    for k, name in enumerate(STATE_NAMES):
        a = ax[0, k]
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
        if k == 0:
            a.legend(fontsize=6.5, loc="upper left")

        real_halfwidth = float((real_p90 - real_p10).mean() / 2)
        gp_halfwidth = float((gp_p90 - gp_p10).mean() / 2)
        if real_halfwidth == 0.0:
            # deterministic channel (e.g. time): no real variance to compare against at all
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

    fig.suptitle(f"Real replicate noise vs GP predicted band - {run.dir.name} "
                f"(trial {gp_idx}, base_level={base_level}, n_replicates={n_replicates})")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "replicate_noise_vs_gp_band.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "replicate_noise_vs_gp_band.csv", index=False)
    print(f"\n----- results written to {out_dir} -----")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", type=str,
                   help="run to probe, e.g. 'seed3_1' (resolved under results/<setup>/, see "
                        "--setup) or a full/relative path to a run folder")
    p.add_argument("--gp_trial", type=int, default=None,
                   help="which trial's GP model to compare against (default: last saved)")
    p.add_argument("--n_replicates", type=int, default=20,
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
                   help="which *_baseline variant this run was trained with -- "
                        "'single_phase_baseline' (the default, matching this script's "
                        "original behaviour) / 'single_phase_baseline_time' differ only in "
                        "whether time is an active GP input regressor; 'dual_phase_baseline' / "
                        "'dual_phase_baseline_time' are the same distinction for dual-phase "
                        "runs. Picking the wrong single-vs-dual library fails immediately "
                        "(DualPhaseModelLearning's gp_list is 2x as long); picking the wrong "
                        "time variant either crashes on a GP active_dims/lengthscales shape "
                        "mismatch or silently reconstructs a GP the checkpoint was never fit "
                        "against.")
    args = p.parse_args()
    main(run_id=args.run_id, gp_trial=args.gp_trial, n_replicates=args.n_replicates,
        base_level=args.base_level, seed_base=args.seed_base, results_root=args.results_root,
        setup=args.setup)

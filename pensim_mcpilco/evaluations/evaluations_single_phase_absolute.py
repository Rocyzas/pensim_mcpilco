"""
PYTHONPATH=.. python -m evaluations.evaluations_single_phase_absolute seed0_1

Same report as evaluations_single_phase.py, but for config_single_phase_absolute runs --
the ABSOLUTE-ACTION arm (Fs commanded directly over [fs_abs_min, fs_abs_max], default 0-200 L/h,
instead of as a +/-FS_SCALE residual on the recipe; see pensim_wrapper.fs_from_action) on the
plain-RBF baseline model with `time` DROPPED from every GP's inputs. Regenerates every plot/table from just the run id
(e.g. "seed0_1", resolved under results/single_phase_absolute/) -- everything else is read
back from that run's own note.txt/log.pkl/monitor.pkl.

Only load_run/reconstruct_gp_agent/plot_fs_residual differ from evaluations_single_phase.py:

  * load_run/reconstruct_gp_agent are passed config_single_phase_absolute.get_config and this
    results root, so the rebuilt PenSimWrapper carries THIS run's action encoding. That matters
    more here than for the other baseline drivers: the wrapper is what turns a policy output into
    a feed rate, so a residual get_config would re-simulate every held-out batch under the wrong
    action semantics -- silently, and with plausible-looking numbers. note.txt now records
    action_mode/fs_abs_min/fs_abs_max, and eval_single_phase_lib._GET_CONFIG_KEYS forwards them,
    so the mismatch raises TypeError instead.
  * plot_fs_residual is told the action_mode, so its A7b "recovered action" panel inverts the
    ABSOLUTE map rather than the residual one (`a = 2*(Fs-lo)/(hi-lo) - 1`). Left at its default
    it would report a residual action this run never used.

Everything else -- held-out yield vs recipe, training progression, GP one-step/multi-step
diagnostics -- is encoding-agnostic and comes from eval_single_phase_lib unchanged, so there is
still exactly one implementation of every plot/table.
"""
import argparse
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import evaluations.eval_single_phase_lib as lib
from mcpilco.config_single_phase_absolute import get_config as absolute_get_config

ABSOLUTE_RESULTS_ROOT = Path(_ROOT) / "results" / "single_phase_absolute"


def main(run_id, gp_trial=None, n_eval_seeds=5, eval_base=700000, compare_seed=None,
        results_root=None):
    results_root = ABSOLUTE_RESULTS_ROOT if results_root is None else results_root
    run = lib.load_run(run_id, get_config_fn=absolute_get_config, results_root=results_root)
    out_dir = run.dir
    compare_seed = compare_seed if compare_seed is not None else eval_base

    print("\n----- A. RL policy vs same-seed PID baseline -----")
    policy_agent, eval_wrapper, np_policy, ref, ref_lbl = lib.build_policy_agent(run, trial_k=gp_trial)

    df, mons_rl, mons_recipe = lib.eval_held_out(
        run, np_policy, eval_wrapper, out_dir, n_eval_seeds=n_eval_seeds, eval_base=eval_base)
    lib.paired_stats(df, out_dir)
    lib.plot_paired_yield(df, mons_rl, mons_recipe, out_dir, show=False)
    lib.plot_total_yield(df, out_dir, show=False)
    lib.plot_seed_avg_vars(mons_rl, mons_recipe, out_dir, show=False)

    fig, m_rl_s, m_recipe_s = lib.plot_model_vs_recipe_single_seed(
        eval_wrapper, np_policy, compare_seed, out_dir, show=False)
    wp = run.cfg["wrapper_par"]
    lib.plot_fs_residual(m_rl_s, m_recipe_s, compare_seed, out_dir, show=False,
                         action_mode=wp["action_mode"], fs_abs_min=wp["fs_abs_min"],
                         fs_abs_max=wp["fs_abs_max"])

    eval_seeds = [eval_base + i for i in range(n_eval_seeds)]
    val_seeds = eval_seeds[:-2] if len(eval_seeds) > 2 else eval_seeds
    lib.validation_learning_curve(run, eval_wrapper, out_dir, val_seeds, show=False)

    print("\n----- B. Training progression -----")
    lib.plot_training_progression(run, ref, ref_lbl, out_dir, show=False)
    lib.plot_all_observations(run, ref, ref_lbl, out_dir, show=False)
    lib.plot_fs_all_episodes(run, out_dir, show=False)

    print("\n----- C. GP model diagnostics -----")
    gp_agent, gp_idx = lib.reconstruct_gp_agent(run, idx=gp_trial, get_config_fn=absolute_get_config)
    ho_idx = gp_idx + 1
    has_ho = ho_idx < len(gp_agent.state_samples_history)
    print(f"reconstructed GP model @ trial {gp_idx} | has held-out batch: {has_ho}")

    per_dim_mse, one_step_results = lib.one_step_fit(gp_agent, gp_idx, out_dir, show=False)
    lib.plot_multistep_rollout(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)
    # match the diagnostic's particle count to what this run actually trained with, rather
    # than plot_particle_bands' own hardcoded default
    train_n_particles = run.cfg["reinforce_par"]["policy_optimization_dict"]["num_particles"]
    lib.plot_particle_bands(gp_agent, gp_idx, ho_idx, has_ho, out_dir, n_part=train_n_particles, show=False)
    lib.plot_calibration(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)
    lib.plot_local_error(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)
    lib.plot_kstep_growth(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)

    if run.has_anchors:
        lib.plot_short_horizon(gp_agent, gp_idx, ho_idx, has_ho, out_dir,
                               run.optim_horizon, run.num_anchors, show=False)
    else:
        print(f"[C.7 short-horizon diagnostic skipped] run did not use anchors "
             f"(optim_horizon={run.optim_horizon}, num_anchors={run.num_anchors})")

    print(f"\n----- DONE: all plots/tables written to {out_dir} -----")
    print(f"held-out mean yield delta (RL-recipe): {df['delta'].mean():+.1f} kg "
         f"(n={len(df)} seeds)")
    print(f"one-step R^2: X={one_step_results['r2_x']:.3f} P={one_step_results['r2_p']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", type=str,
                   help="run to evaluate, e.g. 'seed0_1' (resolved under "
                        "results/single_phase_absolute/) or a full/relative path to a run folder")
    p.add_argument("--gp_trial", type=int, default=None,
                   help="which trial's policy (section A/B) and GP model (section C) to "
                        "evaluate, e.g. 4 out of 8 trained episodes (default: last saved)")
    p.add_argument("--n_eval_seeds", type=int, default=5,
                   help="number of held-out seeds for the section-A RL-vs-recipe comparison")
    p.add_argument("--eval_base", type=int, default=700000,
                   help="first held-out seed (held-out block is eval_base..eval_base+n_eval_seeds-1)")
    p.add_argument("--compare_seed", type=int, default=None,
                   help="single shared seed for A.6/A.7 (default: eval_base)")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under "
                        "(default: results/single_phase_absolute/)")
    args = p.parse_args()
    main(run_id=args.run_id, gp_trial=args.gp_trial, n_eval_seeds=args.n_eval_seeds,
        eval_base=args.eval_base, compare_seed=args.compare_seed,
        results_root=args.results_root)

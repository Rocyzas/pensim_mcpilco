"""
PYTHONPATH=.. python -m evaluations.evaluations_multi_phase seed3_1

Regenerates every plot/table for a dual-phase MC-PILCO run from just its run id (e.g.
"seed3_1", resolved under results/dual_phase/) -- everything else (SEED, NUM_TRIALS, FAST,
pivot_hours, the trained policy, the trained GPs) is read back from that run's own
note.txt/log.pkl/monitor.pkl. All output (PNG/CSV) is written flat into the run's own folder.

Same analysis as evaluations_multi_phase.ipynb -- both import from eval_multi_phase_lib, so
there is exactly one implementation of every plot/table.
"""
import argparse

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import evaluations.eval_multi_phase_lib as lib


def main(run_id, gp_trial=None, n_eval_seeds=5, eval_base=700000, compare_seed=None):
    run = lib.load_run(run_id)
    out_dir = run.dir
    compare_seed = compare_seed if compare_seed is not None else eval_base

    print("\n----- A. RL policy vs same-seed PID baseline -----")
    policy_agent, eval_wrapper, np_policy, ref, ref_lbl = lib.build_policy_agent(run)

    df, mons_rl, mons_recipe = lib.eval_held_out(
        run, np_policy, eval_wrapper, out_dir, n_eval_seeds=n_eval_seeds, eval_base=eval_base)
    lib.paired_stats(df, out_dir)
    lib.plot_paired_yield(df, mons_rl, mons_recipe, out_dir, show=False)
    lib.plot_total_yield(df, out_dir, show=False)
    lib.plot_seed_avg_vars(mons_rl, mons_recipe, out_dir, show=False)

    fig, m_rl_s, m_recipe_s = lib.plot_model_vs_recipe_single_seed(
        eval_wrapper, np_policy, compare_seed, out_dir, show=False)
    lib.plot_fs_residual(m_rl_s, m_recipe_s, compare_seed, out_dir, show=False)

    eval_seeds = [eval_base + i for i in range(n_eval_seeds)]
    val_seeds = eval_seeds[:-2] if len(eval_seeds) > 2 else eval_seeds
    lib.validation_learning_curve(run, eval_wrapper, out_dir, val_seeds, show=False)

    print("\n----- B. Training progression -----")
    lib.plot_training_progression(run, ref, ref_lbl, out_dir, show=False)
    lib.plot_all_observations(run, ref, ref_lbl, out_dir, show=False)
    lib.plot_fs_all_episodes(run, out_dir, show=False)

    print("\n----- C. GP model diagnostics -----")
    gp_agent, gp_idx = lib.reconstruct_gp_agent(run, idx=gp_trial)
    ho_idx = gp_idx + 1
    has_ho = ho_idx < len(gp_agent.state_samples_history)
    print(f"reconstructed GP model @ trial {gp_idx} | has held-out batch: {has_ho}")

    per_dim_mse, one_step_results = lib.one_step_fit(gp_agent, gp_idx, out_dir, show=False)
    lib.plot_multistep_rollout(gp_agent, gp_idx, ho_idx, has_ho, out_dir, run.pivot_hours,
                               blend_half_width_hours=run.blend_half_width_hours, show=False)
    lib.plot_particle_bands(gp_agent, gp_idx, ho_idx, has_ho, out_dir, run.pivot_hours,
                            blend_half_width_hours=run.blend_half_width_hours, show=False)
    lib.plot_calibration(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)
    lib.plot_local_error(gp_agent, gp_idx, ho_idx, has_ho, out_dir, run.pivot_hours,
                         blend_half_width_hours=run.blend_half_width_hours, show=False)
    lib.plot_kstep_growth(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False)

    print(f"\n----- DONE: all plots/tables written to {out_dir} -----")
    print(f"held-out mean yield delta (RL-recipe): {df['delta'].mean():+.1f} kg "
         f"(n={len(df)} seeds)")
    print(f"one-step R^2 -- phase1: X={one_step_results['phase1']['r2_x']:.3f} "
         f"P={one_step_results['phase1']['r2_p']:.3f} | "
         f"phase2: X={one_step_results['phase2']['r2_x']:.3f} "
         f"P={one_step_results['phase2']['r2_p']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", type=str,
                   help="run to evaluate, e.g. 'seed3_1' (resolved under results/dual_phase/) "
                        "or a full/relative path to a run folder")
    p.add_argument("--gp_trial", type=int, default=None,
                   help="which trial's GP model to diagnose in section C (default: last saved)")
    p.add_argument("--n_eval_seeds", type=int, default=5,
                   help="number of held-out seeds for the section-A RL-vs-recipe comparison")
    p.add_argument("--eval_base", type=int, default=700000,
                   help="first held-out seed (held-out block is eval_base..eval_base+n_eval_seeds-1)")
    p.add_argument("--compare_seed", type=int, default=None,
                   help="single shared seed for A.6/A.7 (default: eval_base)")
    args = p.parse_args()
    main(run_id=args.run_id, gp_trial=args.gp_trial, n_eval_seeds=args.n_eval_seeds,
        eval_base=args.eval_base, compare_seed=args.compare_seed)

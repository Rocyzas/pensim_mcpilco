"""
 PYTHONPATH=.. python -m experiments.03_mcpilco_dual_phase_baseline_S --seed 1 --num_trials 5 --fast

S-state ablation: identical in every respect to 03_mcpilco_dual_phase_baseline.py (same
config_dual_phase_baseline, same plain-RBF/time-dropped model in both phases, same cost, same
CLI) EXCEPT that the observed state gains a sixth channel -- S, the substrate concentration:

    ["Wt", "X", "P", "Viscosity", "time"]  ->  ["Wt", "X", "P", "Viscosity", "time", "S"]

Logs to results/dual_phase_baseline_S/ so it never collides with the 5-channel baseline's tree.
Comparing a seed here against the SAME seed under 03_mcpilco_dual_phase_baseline.py is the
intended A/B: everything else is held fixed. --no_S runs the plain 5-channel state instead, so
this driver can also produce that A/B control itself through identical code paths.

WHY S: model_learning_det_time.py's RBF_RecipeMean docstring flags S (alongside DO2) as the state
PenSimPy's own ODE actually needs to give P/X a closed-form mean -- neither is in this 4/5-channel
state, so both currently get only an empirical recipe-trajectory prior instead. S is also the
substrate-accumulation signal that precedes an overfeeding crash (see its STATE_RANGES comment in
pensim_wrapper.py) and is a free online channel (no lab assay, no delay), so it is a plausible
missing regressor for X/P's dynamics quite apart from any prior-mean use. Unlike CER (already
measured near-redundant with the existing state over 40 diagnostic batches, see
02_mcpilco_single_phase_baseline_cer.py's docstring), S has NOT been checked for redundancy here
-- that measurement (corr(S, existing channels), residual-variance-by-bin) would be worth doing
before or alongside this run if the result needs defending.

set_state_names() is APPEND-ONLY (see its docstring in pensim_wrapper.py): S is added at the end,
after `time`, not inserted near X/P where it would sit more naturally biologically -- reordering
would silently reinterpret every positional index baked into penicillin_cost/wt_mass_balance/
model_learning_det_time by other modules at THEIR import time.
"""
import argparse
import datetime
import pickle
import pprint
from pathlib import Path

import numpy as np

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

# NOTE: mcpilco.pensim_wrapper is imported here, but NOT PenSimWrapper/PenSimMCPILCOMultiPhaseDelayed
# -- see main()'s deferred imports. Importing the MODULE is safe (it defines STATE_NAMES rather
# than reading it from somewhere else); importing the names that BAKE IN channel indices is not.
_RESULTS_ROOT = Path(_ROOT) / "results" / "dual_phase_baseline_S"

# The state this driver runs with. Append-only: set_state_names() rejects any reordering, because
# every X_IDX-style lookup in penicillin_cost / wt_mass_balance / model_learning_det_time is
# positional at THEIR import time.
S_STATE_NAMES = ["Wt", "X", "P", "Viscosity", "time", "S"]

# Kept as a plain literal (not imported from mcpilco.config_dual_phase.COST_FUNCTIONS) so
# argparse's --cost_function choices= can be built at module level, BEFORE the deferred
# get_config import below runs -- see main()'s t_sampling/state_names handling for why that
# import can't happen at module level any more.
_COST_FUNCTION_NAMES = ("PeniConcentrationCost", "PeniConcentrationDenseCost",
                        "PeniConcentrationChangeCost", "PeniMassTerminalCost", "PeniMassChangeCost")

"""Auto-incrementing default log dir - not to overwrite:
seed{seed}_1, seed{seed}_2,etc

`results_root` overrides the PARENT only (default: results/dual_phase_baseline_S/), keeping the
seed{seed}_{n} auto-increment. That is the difference from --out_dir, which names the leaf
directory outright and so silently overwrites on a re-run."""
def _next_run_dir(seed, results_root=None):
    root = _RESULTS_ROOT if results_root is None else Path(results_root)
    n = 1
    while (root / f"seed{seed}_{n}").exists():
        n += 1
    return str(root / f"seed{seed}_{n}")


def _write_note(log_path, run_params, cfg):
    """Dump the parameters used for this run to a note.txt."""
    lines = [
        f"run timestamp : {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "== run parameters ==",
    ]
    lines += [f"{k} = {v}" for k, v in run_params.items()]
    lines += ["", "== resolved config =="]
    lines.append(pprint.pformat(cfg, width=100, sort_dicts=False))
    (Path(log_path) / "note.txt").write_text("\n".join(lines) + "\n")


def main(seed=1, num_trials=10, fast=False, out_dir=None, results_root=None, pivot_hours=100.0,
         blend_half_width_hours=50.0, pivot_mode="time", pivot_bm=None,
         on_each_rollout=False, blend_half_width_bm=None,
         risk_weight=0.0, visc_penalty=0.02, constraint_strength=0.75, harvest_reward=True,
         pms_visc_delay=False, use_offline_measurements=False,
         t_sampling=None, cost_function=None, num_explorations=None,
         num_high_feed_probes=0, state_names=None):
    if out_dir is not None and results_root is not None:
        raise ValueError(
            "pass --out_dir OR --results_root, not both: --out_dir sets the exact run directory "
            f"({out_dir!r}) while --results_root ({results_root!r}) only replaces the parent and "
            "keeps the auto-incrementing seed{seed}_{n} leaf name.")

    # Both overrides below mutate module-level constants that OTHER mcpilco modules snapshot at
    # THEIR import time (`from mcpilco.pensim_wrapper import T_SAMPLING` / `STATE_NAMES.index(...)`),
    # so both must run before config_dual_phase_baseline -- and everything it transitively imports
    # -- is EVER imported in this process. Hence get_config and the wrapper classes are imported
    # inside main(), deferred, rather than at module level. See set_t_sampling's and
    # set_state_names' own docstrings for the full rationale and the one-process-per-value caveat.
    import mcpilco.pensim_wrapper as _pw
    if t_sampling is not None:
        _pw.set_t_sampling(t_sampling)
    # This is the ONLY thing that differs from 03_mcpilco_dual_phase_baseline.py's behaviour.
    # state_names=None means "leave pensim_wrapper's default 5-channel STATE_NAMES alone", which
    # is what --no_S selects to produce the A/B control through these same code paths.
    if state_names is not None:
        _pw.set_state_names(state_names)

    from mcpilco.config_dual_phase_baseline import get_config
    from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCOMultiPhaseDelayed

    _pivot_kw = {"pivot_mode": pivot_mode, "on_each_rollout": on_each_rollout}
    if pivot_bm is not None:
        _pivot_kw["pivot_bm"] = pivot_bm
    if blend_half_width_bm is not None:
        _pivot_kw["blend_half_width_bm"] = blend_half_width_bm
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast, pivot_hours=pivot_hours,
                     blend_half_width_hours=blend_half_width_hours, **_pivot_kw,
                     risk_weight=risk_weight, visc_penalty=visc_penalty,
                     constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward, pms_visc_delay=pms_visc_delay,
                     use_offline_measurements=use_offline_measurements,
                     cost_function=cost_function, num_explorations=num_explorations,
                     num_high_feed_probes=num_high_feed_probes)
    log_path = out_dir if out_dir is not None else _next_run_dir(seed, results_root)
    cfg["mc_pilco_init"]["log_path"] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)

    run_params = {"seed": seed, "num_trials": num_trials, "fast": fast, "out_dir": out_dir,
                  "results_root": results_root,
                  "pivot_hours": pivot_hours, "blend_half_width_hours": blend_half_width_hours,
                  # Read back from cfg so note.txt records what was actually used, including
                  # pivot_bm's default when the flag was left off.
                  "pivot_mode": cfg["mc_pilco_init"]["model_learning_par"]["pivot_mode"],
                  "pivot_bm": cfg["mc_pilco_init"]["model_learning_par"]["pivot_bm"],
                  "on_each_rollout": cfg["mc_pilco_init"]["model_learning_par"]["on_each_rollout"],
                  "blend_half_width_bm":
                      cfg["mc_pilco_init"]["model_learning_par"]["blend_half_width_bm"],
                  "risk_weight": risk_weight,
                  "visc_penalty": visc_penalty, "constraint_strength": constraint_strength,
                  "harvest_reward": harvest_reward,
                  # Read back from cfg (not re-hardcoded here) so note.txt can never drift from
                  # what the wrapper actually used.
                  "pms_visc_delay": cfg["wrapper_par"]["pms_visc_delay"],
                  "use_offline_measurements": cfg["wrapper_par"]["use_offline_measurements"],
                  "t_sampling": cfg["mc_pilco_init"]["T_sampling"],
                  "cost_function": cfg["mc_pilco_init"]["f_cost_function"].__name__,
                  "num_explorations": cfg["reinforce_par"]["num_explorations"],
                  "num_high_feed_probes": num_high_feed_probes,
                  # Recorded so eval code can rebuild the right state layout: a run logged under
                  # this driver is NOT interchangeable with a 5-channel baseline run.
                  "state_names": _pw.STATE_NAMES,
                  "state_dim": _pw.STATE_DIM}
    _write_note(log_path, run_params, cfg)
    print(f"state ({_pw.STATE_DIM}ch): {_pw.STATE_NAMES}")

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCOMultiPhaseDelayed(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    # probe batches: excite the feed->response channels the exploration policy structurally
    # cannot (sustained, off-recipe, held across the production window), before training starts
    # (no-op if disabled). See PROBE_PLAN / setup_high_feed_probes docstring
    # -- unlike setup_recipe_anchors, this IS supported for the dual-phase agent.
    if num_high_feed_probes > 0:
        agent.setup_high_feed_probes(**cfg["probe_par"])
    agent.reinforce(**cfg["reinforce_par"])

    # constraint plots
    pickle.dump(wrapper.monitor, open(Path(log_path) / "monitor.pkl", "wb"))
    print(f"Saved monitor.pkl ({len(wrapper.monitor)} episodes) to {log_path}")

    # Where each trajectory's training split actually landed. Only DualPhaseModelLearning's
    # biomass mode populates this (it stays empty under pivot_mode="time", where the split is
    # the constant pivot_step and there is nothing to record), and it is the ONLY record of the
    # per-batch split -- add_data's choice is otherwise invisible once the GPs are fit. Read back
    # by eval_multi_phase_lib.check_split_distribution (C.0f).
    _split_log = getattr(agent.model_learning, "_split_log", [])
    if _split_log:
        pickle.dump(_split_log, open(Path(log_path) / "split_log.pkl", "wb"))
        _hrs = [h for _, h, _ in _split_log]
        _fallbacks = sum(1 for _, _, crossed in _split_log if not crossed)
        print(f"Saved split_log.pkl ({len(_split_log)} trajectories) to {log_path}: "
              f"split at median {np.median(_hrs):.0f}h, range {min(_hrs):.0f}-{max(_hrs):.0f}h, "
              f"{_fallbacks} never crossed pivot_bm (fell back to pivot_step)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_trials", type=int, default=10)
    p.add_argument("--fast", action="store_true", help="small particles/steps/epochs for quick debugging")
    p.add_argument("--out_dir", type=str, default=None,
                   help="exact run directory to write into, bypassing the seed{seed}_{n} naming "
                        "entirely (and so overwriting it if it already exists). Mutually "
                        "exclusive with --results_root.")
    p.add_argument("--results_root", type=str, default=None,
                   help="parent directory for this run, replacing the default "
                        "results/dual_phase_baseline_S/. The seed{seed}_{n} auto-increment is "
                        "KEPT, so repeated runs land in seed3_1, seed3_2, ... under your root "
                        "instead of overwriting. Use this (not --out_dir) for multi-seed sweeps "
                        "that should not clobber each other. Mutually exclusive with --out_dir.")
    p.add_argument("--pivot_hours", type=float, default=100.0,
                   help="dual-phase pivot: phase-1 GPs are TRAINED on decisions up to and "
                        "including this batch time, phase-2 GPs from it onwards (hard split); "
                        "also the CENTER of the sigmoid blend that combines their predictions "
                        "at rollout time (see --blend_half_width_hours)")
    p.add_argument("--pivot_mode", choices=("time", "biomass"), default="time",
                   help="coordinate for the HARD TRAINING SPLIT. 'time' (default) splits every "
                        "batch at --pivot_hours, the previous behaviour exactly. 'biomass' splits "
                        "each batch at ITS OWN biomass-progress crossing (X*Wt running max >= "
                        "--pivot_bm), so batches are cut at the same metabolic stage rather than "
                        "the same wall-clock hour. The ROLLOUT BLEND is the time sigmoid in both "
                        "modes -- only the split moves.")
    p.add_argument("--pivot_bm", type=float, default=None,
                   help="biomass-progress threshold X[g/L]*Wt[kg]/1000 for --pivot_mode=biomass "
                        "(default 1349 ~= 60%% of the median per-batch peak; all 40 diagnostic "
                        "batches cross it, at median 58h / range 39-91h). Ignored under "
                        "--pivot_mode=time.")
    p.add_argument("--onEachRollout", dest="on_each_rollout", action="store_true",
                   help="apply the biomass pivot to the ROLLOUT BLEND as well as the training "
                        "split: the phase weight becomes a per-particle sigmoid on each "
                        "rollout's own running-max X*Wt (centred on --pivot_bm, width "
                        "--blend_half_width_bm) instead of a shared sigmoid on batch time. "
                        "REQUIRES --pivot_mode biomass. The weight is detached, so this adds no "
                        "new gradient path through the policy. Default off = the time sigmoid, "
                        "i.e. exactly the behaviour of every run predating this flag.")
    p.add_argument("--blend_half_width_bm", type=float, default=None,
                   help="half-width of the --onEachRollout blend sigmoid, in BM units "
                        "(X[g/L]*Wt[kg]/1000), NOT hours: w~0.01 at pivot_bm - this and ~0.99 at "
                        "pivot_bm + this. Default 250 puts the ~99%% point at ~1600, below every "
                        "observed per-batch peak, so every rollout does reach phase 2. Ignored "
                        "without --onEachRollout.")
    p.add_argument("--blend_half_width_hours", type=float, default=50.0,
                   help="sigmoid blend half-width (hours): predictions are ~all phase-1 below "
                        "pivot_hours - this, ~all phase-2 above pivot_hours + this, and a smooth "
                        "sigmoid transition in between (default 50 => blend spans 50h-150h)")
    p.add_argument("--risk_weight", type=float, default=0.0,
                   help="PARTICLE SPREAD PENTALTY. weight on imagined-outcome spread in the objective (0 = disabled)")
    p.add_argument("--visc_penalty", type=float, default=0.02,
                   help="lambda_visc: viscosity-collapse constraint weight (0 = disabled)")
    p.add_argument("--constraint_strength", type=float, default=0.75,
                   help="global knob scaling soft_penalty/visc_penalty/risk_weight together "
                        "(see penicillin_cost.py; NOT used to scale rate_penalty)")
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false",
                   help="score only in-tank mass, ignoring penicillin drawn off by the discharge "
                        "pulses (~20%% of batch_yield_kg). Default is to credit it.")
    p.add_argument("--pms_visc_delay", action="store_true",
                   help="MC-PILCO4PMS-style asymmetric delay: GP/cost see the true Viscosity, only "
                        "the policy's input is the held/delayed value (mutually exclusive with "
                        "--use_offline_measurements)")
    p.add_argument("--use_offline_measurements", action="store_true",
                   help="simple 'delayed everywhere' Viscosity: GP/cost/policy all see the same held "
                        "value (mutually exclusive with --pms_visc_delay)")
    p.add_argument("--t_sampling", type=float, default=None,
                   help="override the global decision-step interval in hours (repo default: 5h). "
                        "Affects the WHOLE process (see pensim_wrapper.set_t_sampling) -- the "
                        "simulator's decision cadence, the dual-phase blend's hour<->step "
                        "conversion, the deterministic time channel's per-step delta, and every "
                        "other hour<->step conversion, not just this run's config.")
    p.add_argument("--cost_function", type=str, default=None, choices=_COST_FUNCTION_NAMES,
                   help="reward/cost class (see mcpilco/penicillin_cost.py); default is this "
                        "config's own default (PeniMassChangeCost)")
    p.add_argument("--num_explorations", type=int, default=None,
                   help="initial random-exploration episodes collected before the first trial "
                        "(default: 5)")
    p.add_argument("--num_high_feed_probes", type=int, nargs="?", const=4, default=0,
                   help="fixed-profile probe batches added to the GP training set before training "
                        "starts, ON TOP OF --num_explorations. Bare flag = 4 = one of each probe "
                        "in PROBE_PLAN (slow dither, fast dither, sustained +/- production-window "
                        "steps); higher counts replicate the set under fresh batch realisations "
                        "(0 = disabled)")
    p.add_argument("--no_S", dest="use_S", action="store_false",
                   help="run the plain 5-channel state instead, so this driver can produce the "
                        "A/B control itself under identical code paths (equivalent to "
                        "03_mcpilco_dual_phase_baseline.py, but logged under this tree)")
    args = p.parse_args()
    main(seed=args.seed, num_trials=args.num_trials, fast=args.fast, out_dir=args.out_dir,
         results_root=args.results_root,
         pivot_hours=args.pivot_hours, blend_half_width_hours=args.blend_half_width_hours,
         pivot_mode=args.pivot_mode, pivot_bm=args.pivot_bm,
         on_each_rollout=args.on_each_rollout, blend_half_width_bm=args.blend_half_width_bm,
         risk_weight=args.risk_weight,
         visc_penalty=args.visc_penalty, constraint_strength=args.constraint_strength,
         harvest_reward=args.harvest_reward, pms_visc_delay=args.pms_visc_delay,
         use_offline_measurements=args.use_offline_measurements,
         t_sampling=args.t_sampling, cost_function=args.cost_function,
         num_explorations=args.num_explorations, num_high_feed_probes=args.num_high_feed_probes,
         state_names=S_STATE_NAMES if args.use_S else None)

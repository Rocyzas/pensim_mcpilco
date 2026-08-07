"""
 PYTHONPATH=.. python -m experiments.03_mcpilco_dual_phase_baseline_priors --seed 1 --num_trials 5 --fast

Dual-phase plain-RBF baseline WITH recipe-trajectory prior means: identical driver to
03_mcpilco_dual_phase_baseline.py, but built on config_dual_phase_baseline_priors -- same plain-RBF,
time-dropped kernels as the baseline, PLUS a non-zero prior mean m(x,u) on every learned channel in
BOTH phases, measured from 10 pure-recipe simulator batches (see model_learning_priors.py /
recipe_trajectory_mean.py). Logs to its own results/dual_phase_baseline_priors/ tree so it never
collides with the config_dual_phase_baseline runs it is compared against.
"""
import argparse
import datetime
import pickle
import pprint
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import (PenSimWrapper, PenSimMCPILCOMultiPhaseDelayed,
                                    PIVOT_HOURS, BLEND_HALF_WIDTH_HOURS)

_RESULTS_ROOT = Path(_ROOT) / "results" / "dual_phase_baseline_priors"

# Kept as a plain literal (not imported from mcpilco.config_dual_phase.COST_FUNCTIONS) so
# argparse's --cost_function choices= can be built at module level, BEFORE the deferred
# get_config import below runs -- see main()'s t_sampling handling for why that import can't
# happen at module level any more. get_config still does the actual name->class lookup (and
# will KeyError on a real mismatch), so this list is just for CLI help/validation.
_COST_FUNCTION_NAMES = ("PeniConcentrationCost", "PeniConcentrationDenseCost",
                        "PeniConcentrationChangeCost", "PeniMassTerminalCost", "PeniMassChangeCost")

"""Auto-incrementing default log dir - not to overwrite:
seed{seed}_1, seed{seed}_2,etc"""
def _next_run_dir(seed):
    n = 1
    while (_RESULTS_ROOT / f"seed{seed}_{n}").exists():
        n += 1
    return str(_RESULTS_ROOT / f"seed{seed}_{n}")


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


def main(seed=1, num_trials=10, fast=False, out_dir=None, pivot_hours=100.0,
         blend_half_width_hours=50.0,
         risk_weight=0.0, visc_penalty=0.02, constraint_strength=0.75, harvest_reward=True,
         pms_visc_delay=False, use_offline_measurements=False,
         t_sampling=None, cost_function=None, num_explorations=None,
         num_high_feed_probes=0):
    # T_sampling is a module-level pensim_wrapper.py constant baked into several OTHER mcpilco
    # modules at THEIR OWN import time (model_learning_dual_phase.py -- including its
    # _blend_weight's own T_SAMPLING-derived hour<->step conversion --, model_learning_det_time's
    # TIME_DELTA_NORM, penicillin_cost.py, PenSimWrapper.rollout's own STEPS_PER_DECISION) via
    # `from mcpilco.pensim_wrapper import T_SAMPLING`-style statements, which snapshot the value
    # at THAT moment. So this override must run, via pensim_wrapper.set_t_sampling(), before
    # config_dual_phase_baseline (and everything it transitively imports) is EVER imported in
    # this process -- hence get_config is imported here, deferred, instead of at module level
    # like every other name in this file. See set_t_sampling's own docstring for the full
    # rationale and its one-process-per-value caveat.
    if t_sampling is not None:
        import mcpilco.pensim_wrapper as _pw
        _pw.set_t_sampling(t_sampling)
    from mcpilco.config_dual_phase_baseline_priors import get_config

    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast, pivot_hours=pivot_hours,
                     blend_half_width_hours=blend_half_width_hours,
                     risk_weight=risk_weight, visc_penalty=visc_penalty,
                     constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward, pms_visc_delay=pms_visc_delay,
                     use_offline_measurements=use_offline_measurements,
                     cost_function=cost_function, num_explorations=num_explorations,
                     num_high_feed_probes=num_high_feed_probes)
    log_path = out_dir if out_dir is not None else _next_run_dir(seed)
    cfg["mc_pilco_init"]["log_path"] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)

    run_params = {"seed": seed, "num_trials": num_trials, "fast": fast, "out_dir": out_dir,
                  "pivot_hours": pivot_hours, "blend_half_width_hours": blend_half_width_hours,
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
                  "num_high_feed_probes": num_high_feed_probes}
    _write_note(log_path, run_params, cfg)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_trials", type=int, default=10)
    p.add_argument("--fast", action="store_true", help="small particles/steps/epochs for quick debugging")
    p.add_argument("--out_dir", type=str, default=None, help="override log_path")
    # Default blend window is 50h->150h: pivot_hours=100 (center, w=0.5) with
    # blend_half_width_hours=50 puts w~0.01 at 100-50=50h and w~0.99 at 100+50=150h. NOTE
    # pivot_hours is ALSO the hard training split (phase-1 GPs <= pivot, phase-2 > pivot), so this
    # moves that split to 100h too -- the two are one parameter by design.
    p.add_argument("--pivot_hours", type=float, default=100.0,
                   help="dual-phase pivot: phase-1 GPs are TRAINED on decisions up to and "
                        "including this batch time, phase-2 GPs from it onwards (hard split); "
                        "also the CENTER of the sigmoid blend that combines their predictions "
                        "at rollout time (see --blend_half_width_hours)")
    p.add_argument("--blend_half_width_hours", type=float, default=50.0,
                   help="sigmoid blend half-width (hours): predictions are ~all phase-1 below "
                        "pivot_hours - this, ~all phase-2 above pivot_hours + this, and a smooth "
                        "sigmoid transition in between (default 50 => blend spans 50h-150h)")
    # cost-shaping terms (see penicillin_cost.PeniConcentrationCost). constraint_strength default is
    # 0.75, matching the single-phase drivers post sigma_n^2 fix (see config_single_phase.py's
    # constraint_strength comment) -- NOT validated for dual-phase specifically, override if sweeping.
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
    # Two mutually-exclusive ways to delay Viscosity (PenSimWrapper raises if both are set) -- see
    # config_dual_phase.get_config's docstring. Exposed here to match 02_mcpilco_single_phase_baseline.
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
    # targeted probing: fixed-profile batches added to the GP training set before trial 0, on top
    # of the exploration batches, so the GPs see sustained off-recipe feed (0 = disabled).
    # Unlike --num_anchor_batches (02_mcpilco_single_phase_baseline.py), this IS supported here
    # -- see PenSimMCPILCOMultiPhase.setup_high_feed_probes's docstring for why.
    p.add_argument("--num_high_feed_probes", type=int, nargs="?", const=4, default=0,
                   help="fixed-profile probe batches added to the GP training set before training "
                        "starts, ON TOP OF --num_explorations. Bare flag = 4 = one of each probe "
                        "in PROBE_PLAN (slow dither, fast dither, sustained +/- production-window "
                        "steps); higher counts replicate the set under fresh batch realisations "
                        "(0 = disabled)")
    args = p.parse_args()
    main(seed=args.seed, num_trials=args.num_trials, fast=args.fast, out_dir=args.out_dir,
         pivot_hours=args.pivot_hours, blend_half_width_hours=args.blend_half_width_hours,
         risk_weight=args.risk_weight,
         visc_penalty=args.visc_penalty, constraint_strength=args.constraint_strength,
         harvest_reward=args.harvest_reward, pms_visc_delay=args.pms_visc_delay,
         use_offline_measurements=args.use_offline_measurements,
         t_sampling=args.t_sampling, cost_function=args.cost_function,
         num_explorations=args.num_explorations, num_high_feed_probes=args.num_high_feed_probes)

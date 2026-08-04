"""
 PYTHONPATH=.. python -m experiments.03_mcpilco_dual_phase_baseline_time --seed 1 --num_trials 5 --fast

Plain-RBF dual-phase baseline WITH `time` kept as a GP regressor: identical driver to
03_mcpilco_dual_phase_baseline.py, but built on config_dual_phase_baseline_time (same plain-RBF swap
in both phases, no prior means, but `time` is NOT dropped from the GP inputs -- see
config_dual_phase_baseline_time.py) and logging to its own results/dual_phase_baseline_time/ tree so
it never collides with the config_dual_phase_baseline runs it is compared against. This is the
dual-phase analog of 02_mcpilco_single_phase_baseline_time.py.
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

from mcpilco.config_dual_phase_baseline_time import get_config
from mcpilco.pensim_wrapper import (PenSimWrapper, PenSimMCPILCOMultiPhaseDelayed,
                                    PIVOT_HOURS, BLEND_HALF_WIDTH_HOURS)

_RESULTS_ROOT = Path(_ROOT) / "results" / "dual_phase_baseline_time"

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
         pms_visc_delay=False, use_offline_measurements=False):
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast, pivot_hours=pivot_hours,
                     blend_half_width_hours=blend_half_width_hours,
                     risk_weight=risk_weight, visc_penalty=visc_penalty,
                     constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward, pms_visc_delay=pms_visc_delay,
                     use_offline_measurements=use_offline_measurements)
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
                  "use_offline_measurements": cfg["wrapper_par"]["use_offline_measurements"]}
    _write_note(log_path, run_params, cfg)

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCOMultiPhaseDelayed(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
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
    args = p.parse_args()
    main(seed=args.seed, num_trials=args.num_trials, fast=args.fast, out_dir=args.out_dir,
         pivot_hours=args.pivot_hours, blend_half_width_hours=args.blend_half_width_hours,
         risk_weight=args.risk_weight,
         visc_penalty=args.visc_penalty, constraint_strength=args.constraint_strength,
         harvest_reward=args.harvest_reward, pms_visc_delay=args.pms_visc_delay,
         use_offline_measurements=args.use_offline_measurements)

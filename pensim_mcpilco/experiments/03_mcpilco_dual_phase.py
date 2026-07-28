"""
 PYTHONPATH=.. python -m experiments.03_mcpilco_dual_phase --seed 1 --num_trials 5 --fast
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

from mcpilco.config_dual_phase import get_config
from mcpilco.pensim_wrapper import (PenSimWrapper, PenSimMCPILCOMultiPhase,
                                    PIVOT_HOURS, BLEND_HALF_WIDTH_HOURS)

_RESULTS_ROOT = Path(_ROOT) / "results" / "dual_phase"

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


def main(seed=1, num_trials=10, fast=False, out_dir=None, pivot_hours=PIVOT_HOURS,
         blend_half_width_hours=BLEND_HALF_WIDTH_HOURS,
         risk_weight=0.0, visc_penalty=0.02, constraint_strength=1.5, harvest_reward=True):
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast, pivot_hours=pivot_hours,
                     blend_half_width_hours=blend_half_width_hours,
                     risk_weight=risk_weight, visc_penalty=visc_penalty,
                     constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward)
    log_path = out_dir if out_dir is not None else _next_run_dir(seed)
    cfg["mc_pilco_init"]["log_path"] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)

    run_params = {"seed": seed, "num_trials": num_trials, "fast": fast, "out_dir": out_dir,
                  "pivot_hours": pivot_hours, "blend_half_width_hours": blend_half_width_hours,
                  "risk_weight": risk_weight,
                  "visc_penalty": visc_penalty, "constraint_strength": constraint_strength,
                  "harvest_reward": harvest_reward}
    _write_note(log_path, run_params, cfg)

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCOMultiPhase(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
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
    p.add_argument("--pivot_hours", type=float, default=PIVOT_HOURS,
                   help="dual-phase pivot: phase-1 GPs are TRAINED on decisions up to and "
                        "including this batch time, phase-2 GPs from it onwards (hard split); "
                        "also the CENTER of the sigmoid blend that combines their predictions "
                        "at rollout time (see --blend_half_width_hours)")
    p.add_argument("--blend_half_width_hours", type=float, default=BLEND_HALF_WIDTH_HOURS,
                   help="sigmoid blend half-width (hours): predictions are ~all phase-1 below "
                        "pivot_hours - this, ~all phase-2 above pivot_hours + this, and a smooth "
                        "sigmoid transition in between")
    # cost-shaping terms (see penicillin_cost.PeniConcentrationCost) -- defaults (0.02/1.5)
    # match single-phase's validated configuration (see evaluations/cost_term_report.py's
    # constraint_strength sweep) so single- vs dual-phase comparisons are apples-to-apples.
    p.add_argument("--risk_weight", type=float, default=0.0,
                   help="PARTICLE SPREAD PENTALTY. weight on imagined-outcome spread in the objective (0 = disabled)")
    p.add_argument("--visc_penalty", type=float, default=0.02,
                   help="lambda_visc: viscosity-collapse constraint weight (0 = disabled)")
    p.add_argument("--constraint_strength", type=float, default=1.5,
                   help="global knob scaling soft_penalty/visc_penalty/risk_weight together "
                        "(see penicillin_cost.py; NOT used to scale rate_penalty)")
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false",
                   help="score only in-tank mass, ignoring penicillin drawn off by the discharge "
                        "pulses (~20%% of batch_yield_kg). Default is to credit it.")
    args = p.parse_args()
    main(seed=args.seed, num_trials=args.num_trials, fast=args.fast, out_dir=args.out_dir,
         pivot_hours=args.pivot_hours, blend_half_width_hours=args.blend_half_width_hours,
         risk_weight=args.risk_weight,
         visc_penalty=args.visc_penalty, constraint_strength=args.constraint_strength,
         harvest_reward=args.harvest_reward)

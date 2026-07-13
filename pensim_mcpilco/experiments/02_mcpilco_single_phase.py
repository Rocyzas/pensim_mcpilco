"""
 PYTHONPATH=.. python -m experiments.02_mcpilco_single_phase --seed 1 --num_trials 5 --fast
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

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCO

_RESULTS_ROOT = Path(_ROOT) / "results" / "single_phase"

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


def main(seed=1, num_trials=10, fast=False, out_dir=None,
         optim_horizon=None, num_anchor_batches=0, num_anchors=None, anchor_var=0.01):
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast,
                     optim_horizon_steps=optim_horizon, num_anchor_batches=num_anchor_batches,
                     num_anchors=num_anchors, anchor_var=anchor_var)
    log_path = out_dir if out_dir is not None else _next_run_dir(seed)
    cfg["mc_pilco_init"]["log_path"] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)

    run_params = {"seed": seed, "num_trials": num_trials, "fast": fast, "out_dir": out_dir,
                  "optim_horizon": optim_horizon, "num_anchor_batches": num_anchor_batches,
                  "num_anchors": num_anchors, "anchor_var": anchor_var}
    _write_note(log_path, run_params, cfg)

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCO(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    # multi-origin short rollouts: build the fixed anchor set once, before training (no-op if disabled)
    if num_anchor_batches > 0:
        agent.setup_recipe_anchors(**cfg["anchor_par"])
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
    # multi-origin short-rollout optimisation (all optional; defaults reproduce stock MC-PILCO)
    p.add_argument("--optim_horizon", type=int, default=None,
                   help="cap the imagined GP-rollout to this many steps during policy optimisation")
    p.add_argument("--num_anchor_batches", type=int, default=0,
                   help="pure-recipe batches to launch short rollouts from (0 = disabled)")
    p.add_argument("--num_anchors", type=int, default=None, help="anchor launch states spread across the batch")
    p.add_argument("--anchor_var", type=float, default=0.01, help="per-anchor particle-init variance")
    args = p.parse_args()
    main(args.seed, args.num_trials, args.fast, args.out_dir,
         args.optim_horizon, args.num_anchor_batches, args.num_anchors, args.anchor_var)

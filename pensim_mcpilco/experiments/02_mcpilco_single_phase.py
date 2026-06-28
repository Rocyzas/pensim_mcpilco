"""
 PYTHONPATH=.. python -m experiments.02_mcpilco_single_phase --seed 1 --num_trials 5 --fast
"""
import argparse
import pickle
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCO


def main(seed=1, num_trials=10, fast=False, out_dir=None):
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast)
    if out_dir is not None:
        cfg["mc_pilco_init"]["log_path"] = out_dir
    log_path = cfg["mc_pilco_init"]["log_path"]
    Path(log_path).mkdir(parents=True, exist_ok=True)

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCO(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    agent.reinforce(**cfg["reinforce_par"])

    # constraint plots
    pickle.dump(wrapper.monitor, open(Path(log_path) / "monitor.pkl", "wb"))
    print(f"Saved monitor.pkl ({len(wrapper.monitor)} episodes) to {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_trials", type=int, default=10)
    p.add_argument("--fast", action="store_true", help="small particles/steps/epochs for quick debugging")
    p.add_argument("--out_dir", type=str, default=None, help="override log_path")
    args = p.parse_args()
    main(args.seed, args.num_trials, args.fast, args.out_dir)

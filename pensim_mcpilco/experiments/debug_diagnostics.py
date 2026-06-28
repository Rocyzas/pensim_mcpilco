"""
Capability diagnostics for a single-phase MC-PILCO run — answers "why is yield flat?"

For one seed's log.pkl it plots:
  (1) policy action magnitude |delta(Fs,Fg)| per collected batch — is the policy
      moving, stuck at 0, or saturating at +-1?
  (2) the optimiser's PREDICTED cost per trial vs the REAL collected-batch yield —
      if predicted improves while real stays flat -> model-reality gap; if both flat
      -> no headroom / optimiser stuck.

Run from inside pensim_mcpilco:
    PYTHONPATH=.. python -m experiments.debug_diagnostics --seed 1
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make this script runnable directly (no -m / PYTHONPATH) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

from mcpilco.pensim_wrapper import STATE_RANGES, STATE_NAMES
from utils.recipe import Recipe
from PenSimPy.pensimpy.data.constants import DISCHARGE, DISCHARGE_DEFAULT_PROFILE

P_IDX = STATE_NAMES.index("P")
V_IDX = STATE_NAMES.index("V")
T_SAMPLING = 5.0
_DISCH = Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE)


def _denorm(x, lo, hi):
    return lo + (x + 1.0) * (hi - lo) / 2.0


def batch_yield_kg(s):
    P = _denorm(s[:, P_IDX], *STATE_RANGES["P"])
    V = _denorm(s[:, V_IDX], *STATE_RANGES["V"])
    mass = P * V / 1000.0
    incr = mass - np.concatenate([mass[:1], mass[:-1]])
    fd = np.array([_DISCH.get_value_at(t * T_SAMPLING) for t in range(len(P))])
    return float((incr + P * fd * T_SAMPLING / 1000.0).sum())


def _scalar(x):
    """Pull a python float out of a tensor/array/scalar."""
    try:
        return float(np.asarray(x).reshape(-1)[-1])   # last opt step
    except Exception:
        return float("nan")


def main(seed=1, results_dir="results/single_phase"):
    d = pickle.load(open(Path(results_dir) / f"seed{seed}" / "log.pkl", "rb"))
    states = d["state_samples_history"]      # [expl, tr0, tr1, ...]  normalised
    inputs = d["input_samples_history"]      # same length; normalised actions [-1,1]
    costs  = d.get("cost_trial_list", [])    # per trial: opt-step cost curve

    # (1) policy action magnitude per collected batch
    mean_abs = [float(np.abs(u).mean()) for u in inputs]
    max_abs  = [float(np.abs(u).max()) for u in inputs]
    # (2) real yield per batch + predicted final cost per trial
    yld = [batch_yield_kg(s) for s in states]
    pred_cost = [_scalar(c) for c in costs]   # trial 0.. ; one per trial

    print("batch  |delta|_mean  |delta|_max   yield(kg)")
    for i,(m,mx,y) in enumerate(zip(mean_abs,max_abs,yld)):
        tag = "expl" if i==0 else f"tr{i-1}"
        print(f"{tag:5}  {m:10.3f}  {mx:10.3f}   {y:9.1f}")
    print("\npredicted final cost per trial:", [round(c,3) for c in pred_cost])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bx = np.arange(len(inputs))
    ax[0].plot(bx, mean_abs, "o-", label="mean |δ|")
    ax[0].plot(bx, max_abs, "s--", label="max |δ|", alpha=0.6)
    ax[0].axhline(1.0, color="red", ls=":", label="saturation (±1)")
    ax[0].set_xlabel("batch (0=expl)"); ax[0].set_ylabel("normalised action |δ|")
    ax[0].set_title(f"seed {seed}: policy action magnitude"); ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(range(1, len(yld)), yld[1:], "o-", color="purple", label="real yield (kg)")
    ax[1].set_xlabel("trial"); ax[1].set_ylabel("real yield (kg)", color="purple")
    ax[1].grid(alpha=.3)
    if pred_cost:
        axt = ax[1].twinx()
        axt.plot(range(len(pred_cost)), pred_cost, "x--", color="orange", label="predicted cost")
        axt.set_ylabel("predicted cost (lower=better)", color="orange")
    ax[1].set_title(f"seed {seed}: predicted cost vs real yield")

    out = Path(results_dir) / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / f"diagnostics_seed{seed}.png", dpi=150)
    print(f"\nSaved {out}/diagnostics_seed{seed}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    main(args.seed)

"""
GP-vs-simulator diagnostic for the single-phase MC-PILCO run (item #4).

Answers: "is the learned GP model actually right about PAA dynamics, or is the policy
optimising against a wrong model?" Rebuilds the trained GPs OFFLINE from a run's
log.pkl (no re-run) and reuses MC-PILCO's own checks:
  - get_model_learning_performance : one-step GP delta-prediction vs the recorded target;
  - get_rollout_prediction_performance : multi-step open-loop GP rollout (fed the recorded
    action sequence) vs the actual simulator trajectory.

Produces a 2x2 figure (focus on PAA conc = state dim "PAA", plus P):
  (0,0) one-step predicted dPAA vs target dPAA scatter (mg/L per step) + y=x, coloured by
        current PAA level; prints MSE and R^2.
  (0,1) per-dim one-step MSE (normalised delta space) — model quality across all 8 states.
  (1,0) multi-step PAA(t): GP rollout vs actual, with recorded Fpaa on a twin axis.
  (1,1) multi-step P(t):   GP rollout vs actual.

The GP outputs are deltas in NORMALISED state space (Model_learning.data_to_gp_output),
so a delta is denormalised as d_phys = d_norm * (hi-lo)/2.

NOTE: predictions are evaluated on the same trajectory the model was trained on (the
built-in MC-PILCO check is in-sample); read it as "can the model even fit what it saw",
which is exactly what we need to rule the model in or out as the bottleneck.

Usage (must match the run that produced the log so dims/config line up):
    PYTHONPATH=.. python -m experiments.diagnose_gp --seed 1 --fast
    PYTHONPATH=.. python -m experiments.diagnose_gp --seed 1            # non-fast run
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import (PenSimWrapper, PenSimMCPILCO, STATE_NAMES,
                                     STATE_RANGES, WARMUP_H, T_SAMPLING, PAA_BAND)

PAA_IDX = STATE_NAMES.index("PAA")
P_IDX = STATE_NAMES.index("P")


def _denorm(x, lo, hi):
    return lo + (x + 1.0) * (hi - lo) / 2.0


def _denorm_delta(d, lo, hi):
    return d * (hi - lo) / 2.0


def _resolve_trial(log, trial):
    """Default to the last trial that has a saved GP model."""
    avail = sorted(int(k.split("_")[-1]) for k in log if k.startswith("parameters_gp_"))
    if not avail:
        raise RuntimeError("log.pkl has no parameters_gp_<i> (no trained GP to load)")
    if trial is None:
        return avail[-1]
    if trial not in avail:
        raise RuntimeError(f"trial {trial} not in saved GP trials {avail}")
    return trial


def reconstruct(seed, num_trials, fast, log, idx):
    """Build a PenSimMCPILCO and load the trial-`idx` GP model from `log` (no training)."""
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast)
    cfg["mc_pilco_init"]["log_path"] = None        # never write during diagnosis
    agent = PenSimMCPILCO(pensim_wrapper=PenSimWrapper(**cfg["wrapper_par"]),
                          **cfg["mc_pilco_init"])
    agent.state_samples_history = log["state_samples_history"]
    agent.input_samples_history = log["input_samples_history"]
    agent.noiseless_states_history = log.get("noiseless_states_history",
                                             log["state_samples_history"])

    ml = agent.model_learning
    ml.gp_inputs = log[f"gp_inputs_{idx}"]
    ml.gp_output_list = log[f"gp_output_list_{idx}"]
    ml.num_samples = ml.gp_inputs.shape[0]
    ml.dim_state = len(STATE_NAMES)                # normally set by add_data()
    ml.init_gp_models()                            # fresh modules (init hyperparameters)
    params = log[f"parameters_gp_{idx}"]
    for k in range(ml.num_gp):
        ml.gp_list[k].load_state_dict(params[k])   # restore TRAINED hyperparameters
        ml.norm_list[k] = torch.max(torch.abs(ml.gp_output_list[k]))  # train-time output norm
    with torch.no_grad():
        for k in range(ml.num_gp):
            ml.pretrain_gp(k)                       # rebuild alpha / SOD caches (prints MSE)
    ml.set_eval_mode()
    return agent


def main(seed=1, trial=None, num_trials=10, fast=False,
         results_dir="results/single_phase"):
    d = Path(results_dir) / f"seed{seed}"
    log = pickle.load(open(d / "log.pkl", "rb"))
    idx = _resolve_trial(log, trial)
    print(f"[diagnose_gp] seed {seed}, trial {idx} "
          f"({len(log['state_samples_history'])} episodes, state dim {len(STATE_NAMES)})")

    print("\n--- rebuild GP (offline) — these MSEs should match the training run ---")
    agent = reconstruct(seed, num_trials, fast, log, idx)

    print("\n--- one-step GP prediction performance ---")
    with torch.no_grad():
        _, targets, means, _ = agent.get_model_learning_performance(idx)
        print("\n--- multi-step rollout vs actual ---")
        pred, true, _ = agent.get_rollout_prediction_performance(idx)

    # per-dim one-step MSE (normalised delta space)
    per_dim_mse = [float(((targets[k] - means[k]) ** 2).mean()) for k in range(len(targets))]

    # PAA one-step, denormalised to mg/L per step
    lo, hi = STATE_RANGES["PAA"]
    tgt_paa = _denorm_delta(targets[PAA_IDX].ravel(), lo, hi)
    prd_paa = _denorm_delta(means[PAA_IDX].ravel(), lo, hi)
    ss_res = float(((tgt_paa - prd_paa) ** 2).sum())
    ss_tot = float(((tgt_paa - tgt_paa.mean()) ** 2).sum()) or 1.0
    r2_paa = 1.0 - ss_res / ss_tot
    # current PAA level at each one-step sample (gp_inputs = [states, action]; states first)
    gp_inputs = agent.model_learning.data_to_gp_input(
        torch.tensor(agent.state_samples_history[idx]),
        torch.tensor(agent.input_samples_history[idx]))[:-1, :].detach().cpu().numpy()
    paa_level = _denorm(gp_inputs[:, PAA_IDX], lo, hi)

    # multi-step trajectories, denormalised
    t = WARMUP_H + np.arange(pred.shape[0]) * T_SAMPLING
    paa_pred = _denorm(pred[:, PAA_IDX], lo, hi);  paa_true = _denorm(true[:, PAA_IDX], lo, hi)
    p_pred = _denorm(pred[:, P_IDX], *STATE_RANGES["P"])
    p_true = _denorm(true[:, P_IDX], *STATE_RANGES["P"])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) one-step dPAA scatter
    sc = ax[0, 0].scatter(tgt_paa, prd_paa, c=paa_level, cmap="viridis", s=14, alpha=.8)
    lim = [min(tgt_paa.min(), prd_paa.min()), max(tgt_paa.max(), prd_paa.max())]
    ax[0, 0].plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
    fig.colorbar(sc, ax=ax[0, 0], label="current PAA (mg/L)")
    ax[0, 0].set_title(f"One-step dPAA: GP vs actual  (R^2={r2_paa:.3f})")
    ax[0, 0].set_xlabel("actual dPAA (mg/L per step)")
    ax[0, 0].set_ylabel("GP predicted dPAA (mg/L per step)")
    ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=8)

    # (0,1) per-dim MSE
    ax[0, 1].bar(range(len(per_dim_mse)), per_dim_mse, color="steelblue")
    ax[0, 1].bar([PAA_IDX], [per_dim_mse[PAA_IDX]], color="crimson", label="PAA")
    ax[0, 1].set_xticks(range(len(STATE_NAMES))); ax[0, 1].set_xticklabels(STATE_NAMES, rotation=45)
    ax[0, 1].set_title("Per-dim one-step MSE (normalised delta)")
    ax[0, 1].set_ylabel("MSE"); ax[0, 1].grid(alpha=.3, axis="y"); ax[0, 1].legend(fontsize=8)

    # (1,0) multi-step PAA + Fpaa overlay
    ax[1, 0].plot(t, paa_true, "k-", lw=2, label="actual (simulator)")
    ax[1, 0].plot(t, paa_pred, "C1--", lw=2, label="GP rollout")
    ax[1, 0].axhspan(*PAA_BAND, color="green", alpha=.10, label="band")
    ax[1, 0].set_title("Multi-step PAA: GP rollout vs simulator")
    ax[1, 0].set_xlabel("time (h)"); ax[1, 0].set_ylabel("PAA (mg/L)"); ax[1, 0].grid(alpha=.3)
    try:
        mon = pickle.load(open(d / "monitor.pkl", "rb"))[idx]
        axt = ax[1, 0].twinx()
        axt.plot(mon["t"], mon["Fpaa"], color="purple", lw=1, alpha=.5, label="Fpaa (action)")
        axt.set_ylabel("Fpaa (L/h)", color="purple")
        axt.set_xlim(t[0], t[-1])
    except (FileNotFoundError, IndexError, KeyError):
        pass
    ax[1, 0].legend(fontsize=8, loc="upper left")

    # (1,1) multi-step P
    ax[1, 1].plot(t, p_true, "k-", lw=2, label="actual (simulator)")
    ax[1, 1].plot(t, p_pred, "C1--", lw=2, label="GP rollout")
    ax[1, 1].set_title("Multi-step P: GP rollout vs simulator")
    ax[1, 1].set_xlabel("time (h)"); ax[1, 1].set_ylabel("P (g/L)")
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"GP-vs-simulator diagnostic — seed {seed}, trial {idx}")
    fig.tight_layout()
    out = Path(results_dir) / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"gp_diag_seed{seed}_trial_{idx}.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)

    print(f"\nPAA one-step: R^2={r2_paa:.3f}, MSE(norm delta)={per_dim_mse[PAA_IDX]:.4f}")
    print(f"PAA rollout : final actual={paa_true[-1]:.0f} mg/L, GP pred={paa_pred[-1]:.0f} mg/L")
    print(f"P   rollout : final actual={p_true[-1]:.2f} g/L,  GP pred={p_pred[-1]:.2f} g/L")
    print(f"Saved {fpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trial", type=int, default=None, help="trial index (default: last saved)")
    p.add_argument("--num_trials", type=int, default=10, help="must match the run's config")
    p.add_argument("--fast", action="store_true", help="must match the run's config")
    p.add_argument("--results_dir", type=str, default="results/single_phase")
    args = p.parse_args()
    main(args.seed, args.trial, args.num_trials, args.fast, args.results_dir)

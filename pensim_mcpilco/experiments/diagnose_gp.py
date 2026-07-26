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
                                     STATE_RANGES, STATE_DIM, WARMUP_H, T_SAMPLING, FS)

# The RL action is the Fs (sugar-feed) residual, so this diagnostic focuses on the
# state dims Fs actually drives -- biomass X and penicillin P -- not PAA.
P_IDX = STATE_NAMES.index("P")
X_IDX = STATE_NAMES.index("X")
ACTION_IDX = STATE_DIM          # Fs-residual column in gp_inputs = [state(STATE_DIM), action]

# Horizons (steps ahead) at which to measure k-step prediction error growth.
KSTEP_HORIZONS = (1, 5, 10, 20, 50)


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
        # Only re-derive the training-time output norm if the model was actually trained with
        # flg_norm=True -- otherwise it was trained (and its variance calibrated) against norm=1,
        # and imposing a non-unity norm here would corrupt every predicted-variance number by
        # norm_list[k]**2 without ever having been used during training.
        ml.norm_list[k] = (torch.max(torch.abs(ml.gp_output_list[k]))
                          if getattr(ml, "flg_norm", False)
                          else torch.tensor(1.0, dtype=agent.dtype, device=agent.device))
    with torch.no_grad():
        for k in range(ml.num_gp):
            ml.pretrain_gp(k)                       # rebuild alpha / SOD caches (prints MSE)
    ml.set_eval_mode()
    return agent


def _kstep_errors(agent, batch_idx, horizons, target_origins=60):
    """Sliding-origin k-step prediction error for one batch (open-loop, recorded actions).

    For every start time t0 (strided to ~`target_origins` samples), initialise the GP at the
    TRUE state s(t0), roll the model forward on the batch's recorded actions, and record the
    per-dim squared error of the predicted state s_hat(t0+k) vs the true s(t0+k), for each k in
    `horizons`. Averaging over origins (rather than reading one trajectory at k) is what makes
    error(k) a robust horizon curve -- it decouples "how far ahead" from "which start point".
    Returns rmse[len(horizons), STATE_DIM] in NORMALISED state space (NaN where k unreachable).
    """
    ml = agent.model_learning
    true = torch.tensor(agent.state_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    inp = torch.tensor(agent.input_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    T = true.shape[0]
    kmax = max(horizons)
    stride = max(1, int(np.ceil((T - 1) / target_origins)))
    sq = {k: [] for k in horizons}                 # per-horizon list of per-dim squared errors
    for t0 in range(0, T - 1, stride):
        steps = min(kmax, T - 1 - t0)              # can't predict past the last recorded state
        cur = true[t0:t0 + 1, :]
        for j in range(1, steps + 1):
            cur, _, _ = ml.get_next_state(current_state=cur,
                                          current_input=inp[t0 + j - 1:t0 + j, :],
                                          particle_pred=False)   # deterministic mean, as in rollout()
            if j in sq:                           # only harvest the requested horizons
                err = (cur - true[t0 + j:t0 + j + 1, :]).ravel()
                sq[j].append((err ** 2).detach().cpu().numpy())
    rmse = np.full((len(horizons), STATE_DIM), np.nan)
    for r, k in enumerate(horizons):
        if sq[k]:
            rmse[r] = np.sqrt(np.mean(np.stack(sq[k], 0), axis=0))
    return rmse


def kstep_error_growth(agent, idx, ho_idx, has_ho, horizons, out, seed):
    """Extra diagnostic figure + table: k-step-ahead RMSE growth for X and P (held-out).

    Complements the fixed multi-step rollout (which shows one trajectory) by quantifying HOW
    FAST error compounds with horizon -- i.e. the horizon over which the learned model can be
    trusted, which is exactly what MC-PILCO's policy optimisation rolls the model over.
    """
    horizons = np.asarray(horizons)
    rmse_in = _kstep_errors(agent, idx, horizons)                       # in-sample reference
    rmse_ho = _kstep_errors(agent, ho_idx, horizons) if has_ho else None

    print("\n--- k-step-ahead RMSE growth (sliding-origin, open-loop; physical units) ---")
    print(f"{'k':>4} {'h_ahead':>8} | {'X in':>8} {'X held-out':>11} | {'P in':>8} {'P held-out':>11}   (g/L)")
    for r, k in enumerate(horizons):
        xin = _denorm_delta(rmse_in[r, X_IDX], *STATE_RANGES["X"])
        pin = _denorm_delta(rmse_in[r, P_IDX], *STATE_RANGES["P"])
        xho = _denorm_delta(rmse_ho[r, X_IDX], *STATE_RANGES["X"]) if has_ho else float("nan")
        pho = _denorm_delta(rmse_ho[r, P_IDX], *STATE_RANGES["P"]) if has_ho else float("nan")
        print(f"{k:>4d} {k * T_SAMPLING:>8.1f} | {xin:>8.3f} {xho:>11.3f} | {pin:>8.3f} {pho:>11.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for dim, name, color in [(X_IDX, "X", "crimson"), (P_IDX, "P", "steelblue")]:
        lo, hi = STATE_RANGES[name]
        if has_ho:
            y = _denorm_delta(rmse_ho[:, dim], lo, hi); m = np.isfinite(y)
            ax.plot(horizons[m], y[m], "-o", color=color, lw=2, label=f"{name} held-out (batch {ho_idx})")
        y = _denorm_delta(rmse_in[:, dim], lo, hi); m = np.isfinite(y)
        ax.plot(horizons[m], y[m], "--o", color=color, lw=1.5, alpha=.45,
                label=f"{name} in-sample (batch {idx})")
    ax.set_xscale("log"); ax.set_yscale("linear")
    ax.set_xticks(horizons); ax.set_xticklabels([str(k) for k in horizons])
    ax.set_xlabel(f"prediction horizon k (steps ahead)   [1 step = {T_SAMPLING:g} h]")
    ax.set_ylabel("RMSE (g/L, linear scale)")
    ax.set_title("k-step-ahead prediction error growth\n(sliding-origin, open-loop, held-out vs in-sample)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fpath = out / f"gp_kstep_seed{seed}_trial_{idx}_updated_full07.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)
    print(f"Saved {fpath}")


def main(seed=0, trial=None, num_trials=10, fast=False, results_dir="results/single_phase/seed0_2"):
    base = Path(results_dir)
    if not base.is_absolute():
        base = Path(_ROOT) / base
    direct = base / "log.pkl"
    if direct.exists():
        log_file = direct
    else:
        raise FileNotFoundError(
            f"no log.pkl found (tried {direct}); "
            f"pass --results_dir pointing at the run folder"
        )
    d = log_file.parent
    log = pickle.load(open(log_file, "rb"))
    idx = _resolve_trial(log, trial)
    print(f"[diagnose_gp] seed {seed}, trial {idx} "
          f"({len(log['state_samples_history'])} episodes, state dim {len(STATE_NAMES)})")

    print("\n--- rebuild GP (offline) — these MSEs should match the training run ---")
    agent = reconstruct(seed, num_trials, fast, log, idx)

    print("\n--- one-step GP prediction performance (in-sample) ---")
    with torch.no_grad():
        _, targets, means, _ = agent.get_model_learning_performance(idx)
        print("\n--- multi-step rollout (IN-SAMPLE: batch idx is in the GP training set) ---")
        pred, true, _ = agent.get_rollout_prediction_performance(idx)

    # OUT-OF-SAMPLE test (the supervisor's harder diagnostic): model@idx was trained on
    # episodes 0..idx, so grading it on the *next* batch (idx+1) -- collected under the
    # trial-idx control policy, i.e. a trajectory that deviates from the recipe -- is a
    # genuine held-out rollout. In-sample (idx) only shows the GP reproduced its own
    # training data; the held-out rollout is what can expose model error in the regions
    # the policy explores (the likely source of the decreasing-yield bug).
    ho_idx = idx + 1
    has_ho = ho_idx < len(agent.state_samples_history)
    if has_ho:
        print(f"\n--- multi-step rollout (HELD-OUT: model@{idx} predicting unseen batch {ho_idx}) ---")
        with torch.no_grad():
            pred_ho, true_ho, _ = agent.get_rollout_prediction_performance(ho_idx)
    else:
        print(f"\n[warn] no held-out batch {ho_idx} in history "
              f"({len(agent.state_samples_history)} episodes); showing in-sample only")

    # per-dim one-step MSE (normalised delta space)
    per_dim_mse = [float(((targets[k] - means[k]) ** 2).mean()) for k in range(len(targets))]

    # One-step biomass delta (dX): the state Fs most directly drives.
    lo, hi = STATE_RANGES["X"]
    tgt_dx = _denorm_delta(targets[X_IDX].ravel(), lo, hi)
    prd_dx = _denorm_delta(means[X_IDX].ravel(), lo, hi)
    ss_res = float(((tgt_dx - prd_dx) ** 2).sum())
    ss_tot = float(((tgt_dx - tgt_dx.mean()) ** 2).sum()) or 1.0
    r2_x = 1.0 - ss_res / ss_tot
    # colour the scatter by the Fs action (residual in [-1, 1]) applied at each step,
    # to see whether the GP captures the action -> dynamics effect.
    gp_inputs = agent.model_learning.data_to_gp_input(
        torch.tensor(agent.state_samples_history[idx]),
        torch.tensor(agent.input_samples_history[idx]))[:-1, :].detach().cpu().numpy()
    action_col = gp_inputs[:, ACTION_IDX]

    # multi-step trajectories, denormalised (in-sample: batch idx)
    t = WARMUP_H + np.arange(pred.shape[0]) * T_SAMPLING
    x_pred = _denorm(pred[:, X_IDX], lo, hi);  x_true = _denorm(true[:, X_IDX], lo, hi)
    p_pred = _denorm(pred[:, P_IDX], *STATE_RANGES["P"])
    p_true = _denorm(true[:, P_IDX], *STATE_RANGES["P"])

    # held-out trajectories (batch idx+1), denormalised
    if has_ho:
        t_ho = WARMUP_H + np.arange(pred_ho.shape[0]) * T_SAMPLING
        x_pred_ho = _denorm(pred_ho[:, X_IDX], lo, hi); x_true_ho = _denorm(true_ho[:, X_IDX], lo, hi)
        p_pred_ho = _denorm(pred_ho[:, P_IDX], *STATE_RANGES["P"])
        p_true_ho = _denorm(true_ho[:, P_IDX], *STATE_RANGES["P"])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) one-step dX scatter, coloured by the Fs action applied at each step
    sc = ax[0, 0].scatter(tgt_dx, prd_dx, c=action_col, cmap="viridis", s=14, alpha=.8)
    lim = [min(tgt_dx.min(), prd_dx.min()), max(tgt_dx.max(), prd_dx.max())]
    ax[0, 0].plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
    fig.colorbar(sc, ax=ax[0, 0], label="Fs action (residual -1..+1)")
    ax[0, 0].set_title(f"One-step dX: GP vs actual  (R^2={r2_x:.3f}, in-sample)")
    ax[0, 0].set_xlabel("actual dX (g/L per step)")
    ax[0, 0].set_ylabel("GP predicted dX (g/L per step)")
    ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=8)

    # (0,1) per-dim MSE
    ax[0, 1].bar(range(len(per_dim_mse)), per_dim_mse, color="steelblue")
    ax[0, 1].bar([X_IDX], [per_dim_mse[X_IDX]], color="crimson", label="X")
    ax[0, 1].set_xticks(range(len(STATE_NAMES))); ax[0, 1].set_xticklabels(STATE_NAMES, rotation=45)
    ax[0, 1].set_title("Per-dim one-step MSE (in-sample, normalised delta)")
    ax[0, 1].set_ylabel("MSE"); ax[0, 1].grid(alpha=.3, axis="y"); ax[0, 1].legend(fontsize=8)

    # (1,0) multi-step X (biomass): in-sample (batch idx) vs held-out (batch idx+1).
    # If the GP tracks the black/orange (in-sample) pair but the red diverges from the
    # blue (held-out), that gap is the model failing where the policy operates.
    ax[1, 0].plot(t, x_true, "k-", lw=2, label=f"in-sample true (batch {idx})")
    ax[1, 0].plot(t, x_pred, "C1--", lw=2, label="in-sample GP")
    if has_ho:
        ax[1, 0].plot(t_ho, x_true_ho, "-", color="steelblue", lw=2, label=f"held-out true (batch {ho_idx})")
        ax[1, 0].plot(t_ho, x_pred_ho, "--", color="crimson", lw=2, label="held-out GP")
    ax[1, 0].set_title("Multi-step X (biomass): in-sample vs held-out")
    ax[1, 0].set_xlabel("time (h)"); ax[1, 0].set_ylabel("X (g/L)"); ax[1, 0].grid(alpha=.3)
    # Fs overlays on a secondary axis: the recipe baseline (the a=0 reference the residual
    # action perturbs) plus the actual feed the policy applied in each rollout's batch.
    # The gap between a batch's Fs and the recipe shows how hard the policy fed off-baseline,
    # so an X divergence above can be attributed to the feed that produced it.
    axt = ax[1, 0].twinx()
    axt.set_ylabel("Fs (L/h)", color="purple")
    axt.set_xlim(t[0], t[-1])
    try:
        monitor = pickle.load(open(d / "monitor.pkl", "rb"))
    except FileNotFoundError:
        monitor = None
    # recipe Fs is deterministic (identical across batches) and comes from the wrapper, so it
    # is drawable even without monitor.pkl; sample it over the in-sample batch's time grid.
    grid = monitor[idx]["t"] if monitor is not None else t
    recipe_fs = [agent.system._recipe.get_values_dict_at(time=float(tt))[FS] for tt in grid]
    axt.plot(grid, recipe_fs, color="gray", ls="--", lw=1.2, alpha=.6, label="recipe Fs (a=0)")
    if monitor is not None:
        try:
            axt.plot(monitor[idx]["t"], monitor[idx]["Fs"], color="C1", ls=":", lw=1.2, alpha=.6,
                     label=f"Fs in-sample (batch {idx})")
            if has_ho:
                axt.plot(monitor[ho_idx]["t"], monitor[ho_idx]["Fs"], color="crimson", ls=":", lw=1.2,
                         alpha=.6, label=f"Fs held-out (batch {ho_idx})")
        except (IndexError, KeyError):
            pass
    # merged legend: ax[1,0].legend() alone drops the twin-axis handles, so combine both
    # so the X curves AND every Fs overlay appear.
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    h2, l2 = axt.get_legend_handles_labels()
    ax[1, 0].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # (1,1) multi-step P: in-sample vs held-out
    ax[1, 1].plot(t, p_true, "k-", lw=2, label=f"in-sample true (batch {idx})")
    ax[1, 1].plot(t, p_pred, "C1--", lw=2, label="in-sample GP")
    if has_ho:
        ax[1, 1].plot(t_ho, p_true_ho, "-", color="steelblue", lw=2, label=f"held-out true (batch {ho_idx})")
        ax[1, 1].plot(t_ho, p_pred_ho, "--", color="crimson", lw=2, label="held-out GP")
    ax[1, 1].set_title("Multi-step P: in-sample vs held-out")
    ax[1, 1].set_xlabel("time (h)"); ax[1, 1].set_ylabel("P (g/L)")
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"GP-vs-simulator diagnostic — seed {seed}, model@trial {idx}"
                 + (f" (held-out batch {ho_idx})" if has_ho else " (in-sample only)"))
    fig.tight_layout()
    out = d.parent / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"gp_diag_seed{seed}_trial_{idx}_updated_full07.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)

    print(f"\nX one-step (in-sample): R^2={r2_x:.3f}, MSE(norm delta)={per_dim_mse[X_IDX]:.4f}")
    print(f"X rollout  in-sample : final actual={x_true[-1]:.2f} g/L, GP pred={x_pred[-1]:.2f} g/L")
    print(f"P rollout  in-sample : final actual={p_true[-1]:.2f} g/L, GP pred={p_pred[-1]:.2f} g/L")
    if has_ho:
        print(f"X rollout  HELD-OUT  : final actual={x_true_ho[-1]:.2f} g/L, GP pred={x_pred_ho[-1]:.2f} g/L")
        print(f"P rollout  HELD-OUT  : final actual={p_true_ho[-1]:.2f} g/L, GP pred={p_pred_ho[-1]:.2f} g/L")
    print(f"Saved {fpath}")

    # extra, self-contained diagnostic: k-step-ahead error growth (guarded so it can never
    # break the 2x2 figure above). Held-out is the primary curve; in-sample is a reference.
    try:
        with torch.no_grad():
            kstep_error_growth(agent, idx, ho_idx, has_ho, KSTEP_HORIZONS, out, seed)
    except Exception as e:
        print(f"[warn] k-step error-growth plot skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trial", type=int, default=None, help="trial index (default: last saved)")
    p.add_argument("--num_trials", type=int, default=4, help="must match the run's config")
    p.add_argument("--fast", action="store_true", help="must match the run's config")
    p.add_argument("--results_dir", type=str, default="results/single_phase/seed0_2")
    args = p.parse_args()
    main(args.seed, args.trial, args.num_trials, args.fast, args.results_dir)

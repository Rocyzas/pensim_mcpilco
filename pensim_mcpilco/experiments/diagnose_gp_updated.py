"""GP diagnostic TAILORED to the multi-origin short-horizon update.

The training change (setup_recipe_anchors + optim_horizon_steps) means the policy is no longer
optimised over the full 114-step imagined rollout. Instead it is optimised over many SHORT rollouts
of length H = `optim_horizon_steps` (default 25 steps = 50 h), each LAUNCHED FROM A RECIPE ANCHOR
state spread across the batch. So the question the original diagnose_gp.py answers ("is the full
rollout accurate?") is no longer the relevant one. What matters now is:

  1. Is the GP trustworthy INSIDE the H-step optimiser window (and how bad is it BEYOND H)?
  2. Over the exact H-step windows the optimiser uses -- launched from recipe anchors across the
     batch -- how accurate is the GP, and how does that accuracy DIFFER by batch phase
     (early growth vs late production)?

This script does NOT modify diagnose_gp.py; it imports its stable helpers and adds the tailored
analysis. Run it against a multi-origin run folder (matching --num_trials / --fast), passing the
--optim_horizon you trained with so the optimiser window is marked correctly.

    python experiments/diagnose_gp_updated.py --seed 0 --num_trials 2 --fast \
        --optim_horizon 25 --results_dir results/single_phase/_mo_on
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # experiments/ -> import diagnose_gp

# reuse the stable machinery from the original diagnostic (nothing there is changed)
from diagnose_gp import (reconstruct, _resolve_trial, _kstep_errors,
                         _denorm, _denorm_delta, P_IDX, X_IDX)
from mcpilco.pensim_wrapper import (PenSimWrapper, STATE_NAMES, STATE_RANGES, STATE_DIM,
                                    WARMUP_H, T_SAMPLING, CONTROL_H, initial_state_norm)


def recipe_batch(agent, seed):
    """Roll ONE pure-recipe (a=0) batch on a FRESH wrapper -> the deterministic recipe trajectory
    the optimiser's anchors are drawn from (same seed_offset family as the run). Returns the
    normalised (states, inputs); does not touch the reconstructed GP or its data."""
    w = PenSimWrapper(seed_offset=agent.system.seed_offset)
    a0 = lambda state, decision_idx: np.array([0.0])
    states, inputs, _ = w.rollout(s0=initial_state_norm(), policy=a0, T=CONTROL_H,
                                  dt=agent.T_sampling, noise=agent.std_meas_noise, seed=seed)
    return states, inputs


def window_rollouts(agent, true, inp, origins, H):
    """From each origin t0 in `origins`, roll the GP H steps open-loop on the recorded `inp` actions
    (pure GP; truth only seeds t0). Returns:
      preds[o] : [h+1, STATE_DIM] normalised GP trajectory (index 0 = the true anchor state)
      rmseH[o, dim] : per-dim RMSE over that H-step window (normalised state space)
    This is exactly what the optimiser's short imagined rollout looks like from each anchor."""
    ml = agent.model_learning
    true_t = torch.tensor(true, dtype=agent.dtype, device=agent.device)
    inp_t = torch.tensor(inp, dtype=agent.dtype, device=agent.device)
    T = true_t.shape[0]
    preds, rmseH = [], []
    for t0 in origins:
        h = int(min(H, T - 1 - t0))
        cur = true_t[t0:t0 + 1, :]
        traj, sq = [cur], []
        for j in range(1, h + 1):
            cur, _, _ = ml.get_next_state(current_state=cur,
                                          current_input=inp_t[t0 + j - 1:t0 + j, :],
                                          particle_pred=False)
            traj.append(cur)
            sq.append(((cur - true_t[t0 + j:t0 + j + 1, :]).ravel() ** 2).detach().cpu().numpy())
        preds.append(torch.cat(traj, 0).detach().cpu().numpy())
        rmseH.append(np.sqrt(np.mean(np.stack(sq, 0), axis=0)) if sq else np.full(STATE_DIM, np.nan))
    return preds, np.stack(rmseH)


def main(seed=0, trial=None, num_trials=10, fast=False, optim_horizon=25, num_anchors=12,
         results_dir="results/single_phase/_mo_on"):
    base = Path(results_dir)
    if not base.is_absolute():
        base = Path(_ROOT) / base
    log_file = base / "log.pkl"
    if not log_file.exists():
        raise FileNotFoundError(f"no log.pkl found (tried {log_file}); pass --results_dir at the run folder")
    d = log_file.parent
    log = pickle.load(open(log_file, "rb"))
    idx = _resolve_trial(log, trial)
    H = int(optim_horizon)
    print(f"[diagnose_gp_updated] seed {seed}, trial {idx}, optimiser horizon H={H} steps "
          f"({H * T_SAMPLING:g} h); {len(log['state_samples_history'])} episodes")

    print("\n--- rebuild GP (offline) ---")
    agent = reconstruct(seed, num_trials, fast, log, idx)
    ho_idx = idx + 1
    has_ho = ho_idx < len(agent.state_samples_history)

    # ---- (A) k-step error growth: inside vs BEYOND the optimiser window H -------------------
    Tb = agent.state_samples_history[idx].shape[0]
    horizons = np.array(sorted({k for k in (1, 5, 10, 15, 20, H, int(1.5 * H), 2 * H, 50)
                                if 1 <= k <= Tb - 1}))
    print("\n--- k-step RMSE growth (sliding-origin, open-loop; g/L). H marks the optimiser window ---")
    rmse_in = _kstep_errors(agent, idx, horizons)
    rmse_ho = _kstep_errors(agent, ho_idx, horizons) if has_ho else None
    print(f"{'k':>4} {'h':>7} {'in/out':>7} | {'X in':>8} {'X held':>9} | {'P in':>8} {'P held':>9}")
    for r, k in enumerate(horizons):
        tag = "IN" if k <= H else "beyond"
        xi = _denorm_delta(rmse_in[r, X_IDX], *STATE_RANGES["X"]); pi = _denorm_delta(rmse_in[r, P_IDX], *STATE_RANGES["P"])
        xh = _denorm_delta(rmse_ho[r, X_IDX], *STATE_RANGES["X"]) if has_ho else np.nan
        ph = _denorm_delta(rmse_ho[r, P_IDX], *STATE_RANGES["P"]) if has_ho else np.nan
        print(f"{k:>4d} {k*T_SAMPLING:>7.1f} {tag:>7} | {xi:>8.3f} {xh:>9.3f} | {pi:>8.3f} {ph:>9.3f}")

    # ---- (B) per-anchor H-step windows on the recipe trajectory (the optimiser's launch points) --
    have_anchors = False
    try:
        rs, ri = recipe_batch(agent, agent.system.seed_offset)
        print(f"[seed] recipe anchors rolled with sim seed={agent.system.seed_offset} "
              f"(== training anchor batch-0 seed; matches the run so long as --seed matches)")
        Tr = rs.shape[0]
        # launch every window so it has the FULL H steps and the set spans the whole batch
        # (first window 0..H, last window ends at the batch end) -> uniform lengths, no "cut off".
        anchor_t0 = np.linspace(0, max(1, Tr - 1 - H), num_anchors).round().astype(int)
        preds, rmseH = window_rollouts(agent, rs, ri, anchor_t0, H)
        anchor_hours = WARMUP_H + anchor_t0 * T_SAMPLING
        xH = _denorm_delta(rmseH[:, X_IDX], *STATE_RANGES["X"])
        pH = _denorm_delta(rmseH[:, P_IDX], *STATE_RANGES["P"])
        have_anchors = True
        print(f"\n--- per-anchor {H}-step-window RMSE on the recipe trajectory (how it differs by phase) ---")
        print(f"{'anchor_h':>9} | {'X RMSE':>8} | {'P RMSE':>8}   (g/L over the window)")
        for a in range(len(anchor_t0)):
            print(f"{anchor_hours[a]:>9.1f} | {xH[a]:>8.3f} | {pH[a]:>8.3f}")
        print(f"window mean: X={np.nanmean(xH):.3f} g/L, P={np.nanmean(pH):.3f} g/L")
    except Exception as e:
        print(f"[warn] recipe-anchor window analysis skipped: {e}")

    # ---- payoff number: GP error over the optimiser window vs the full-batch rollout (held-out) --
    if has_ho:
        full_k = np.array([H, Tb - 1])
        rmse_cmp = _kstep_errors(agent, ho_idx, full_k)
        pH_win = _denorm_delta(rmse_cmp[0, P_IDX], *STATE_RANGES["P"])
        pH_full = _denorm_delta(rmse_cmp[1, P_IDX], *STATE_RANGES["P"])
        print(f"\n[payoff] held-out P RMSE: over optimiser window H={H} -> {pH_win:.2f} g/L | "
              f"over full {Tb-1} steps -> {pH_full:.2f} g/L  (the old horizon the optimiser used)")

    # ================================ FIGURE (2x2) ================================
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    hcol = "purple"

    # (0,0) error growth, optimiser window shaded
    a00 = ax[0, 0]
    for dim, name, color in [(X_IDX, "X", "crimson"), (P_IDX, "P", "steelblue")]:
        if has_ho:
            y = _denorm_delta(rmse_ho[:, dim], *STATE_RANGES[name]); m = np.isfinite(y)
            a00.plot(horizons[m], y[m], "-o", color=color, lw=2, label=f"{name} held-out")
        y = _denorm_delta(rmse_in[:, dim], *STATE_RANGES[name]); m = np.isfinite(y)
        a00.plot(horizons[m], y[m], "--o", color=color, lw=1.4, alpha=.45, label=f"{name} in-sample")
    a00.axvspan(horizons.min(), H, color="green", alpha=.06)
    a00.axvline(H, color=hcol, ls=":", lw=1.6, label=f"optimiser horizon H={H}")
    a00.set_yscale("linear"); a00.set_xlabel(f"horizon k (steps)   [1 step = {T_SAMPLING:g} h]")
    a00.set_ylabel("RMSE (g/L, linear)")
    a00.set_title(f"Error growth: INSIDE (green) vs BEYOND the H={H} optimiser window")
    a00.grid(alpha=.3, which="both"); a00.legend(fontsize=7)

    # (0,1) per-anchor H-step RMSE vs batch phase
    a01 = ax[0, 1]
    if have_anchors:
        a01.plot(anchor_hours, xH, "-o", color="crimson", lw=2, label="X")
        a01.plot(anchor_hours, pH, "-o", color="steelblue", lw=2, label="P")
        a01.set_xlabel("anchor launch time in batch (h)")
        a01.set_ylabel(f"{H}-step-window RMSE (g/L)")
        a01.set_title(f"Per-anchor {H}-step accuracy across the batch\n(the windows the policy is optimised on)")
        a01.grid(alpha=.3); a01.legend(fontsize=8)
    else:
        a01.set_axis_off(); a01.set_title("per-anchor windows unavailable")

    # (1,0)/(1,1) short GP rollouts launched from each anchor, over the true recipe trajectory.
    # Each colored curve is ONE anchor's H-step pure-GP rollout; the dot + vertical line at its
    # start is the ONLY point truth is injected (the true recipe state) -- everything after is GP.
    if have_anchors:
        norm = plt.Normalize(vmin=float(anchor_hours.min()), vmax=float(anchor_hours.max()))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
    for a1, dim, name in [(ax[1, 0], X_IDX, "X"), (ax[1, 1], P_IDX, "P")]:
        if have_anchors:
            lo, hi = STATE_RANGES[name]
            t_full = WARMUP_H + np.arange(Tr) * T_SAMPLING
            true_phys = _denorm(rs[:, dim], lo, hi)
            (rl,) = a1.plot(t_full, true_phys, "k-", lw=2, label="recipe true", zorder=3)
            for a, t0 in enumerate(anchor_t0):
                c = plt.cm.viridis(norm(anchor_hours[a]))
                tw = WARMUP_H + (t0 + np.arange(preds[a].shape[0])) * T_SAMPLING
                a1.plot(tw, _denorm(preds[a][:, dim], lo, hi), "-", color=c, lw=1.6, alpha=.9)
                a1.axvline(anchor_hours[a], color="0.7", ls=":", lw=.6, alpha=.5, zorder=0)   # truth injected here
                a1.plot(anchor_hours[a], true_phys[t0], "o", color=c, ms=5, mec="k", mew=.5, zorder=4)
            a1.set_xlabel("time (h)"); a1.set_ylabel(f"{name} (g/L)")
            a1.set_title(f"{H}-step GP rollouts from recipe anchors — {name}\n"
                         f"(each curve = one anchor; dot/line = true state seeded, pure GP after)")
            a1.grid(alpha=.3)
            fig.colorbar(sm, ax=a1).set_label("anchor launch time (h)")
            marker_proxy = Line2D([0], [0], marker="o", color="w", mec="k", mfc="0.6", ms=6,
                                  label="anchor launch (truth injected)")
            a1.legend(handles=[rl, marker_proxy], fontsize=7, loc="upper left")
        else:
            a1.set_axis_off()

    fig.suptitle(f"Short-horizon GP diagnostic — seed {seed}, model@trial {idx}, optimiser H={H} steps "
                 f"({H*T_SAMPLING:g} h)")
    fig.tight_layout()
    out = d.parent / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"gp_shorthorizon_seed{seed}_trial_{idx}_20hor.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)
    print(f"\nSaved {fpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=None, help="trial index (default: last saved)")
    p.add_argument("--num_trials", type=int, default=2, help="must match the run's config")
    p.add_argument("--fast", action="store_true", help="must match the run's config")
    p.add_argument("--optim_horizon", type=int, default=25, help="H used in training (marks the optimiser window)")
    p.add_argument("--num_anchors", type=int, default=12, help="recipe anchor launch points to probe")
    p.add_argument("--results_dir", type=str, default="results/single_phase/_mo_on")
    args = p.parse_args()
    main(args.seed, args.trial, args.num_trials, args.fast, args.optim_horizon,
         args.num_anchors, args.results_dir)

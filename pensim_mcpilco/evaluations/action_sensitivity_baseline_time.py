"""
PYTHONPATH=.. python evaluations/action_sensitivity_baseline_time.py seed0_1

Same diagnostic as action_sensitivity.py, but for config_single_phase_baseline_time runs (plain
RBF on every channel, WITH time kept as a GP input regressor -- see
config_single_phase_baseline_time.py / model_learning_baseline.py). Only load_agent differs: it
resolves run_id under results/single_phase_baseline_time/ and reconstructs the GP model via
config_single_phase_baseline_time.get_config -- using config_single_phase.get_config or
config_single_phase_baseline.get_config here would silently rebuild the wrong architecture (their
parameter sets can overlap enough that load_state_dict succeeds without error -- see
eval_single_phase_lib.reconstruct_gp_agent's docstring).

Takes only a run id (resolved under results/single_phase_baseline_time/, or a full/relative
path) -- seed/num_trials/fast and the trained GPs are read back from that run's own
note.txt/log.pkl via eval_single_phase_lib.load_run/reconstruct_gp_agent.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if _os.path.dirname(_ROOT) not in _sys.path:
    _sys.path.insert(0, _os.path.dirname(_ROOT))

import evaluations.eval_single_phase_lib as lib
from mcpilco.config_single_phase_baseline import get_config as _no_time_get_config
from mcpilco.config_single_phase_baseline_time import get_config as _time_get_config
from mcpilco.pensim_wrapper import (PenSimWrapper, STATE_NAMES, STATE_DIM, ACTION_DIM,
                                    CONTROL_H, T_SAMPLING, TIME_IDX)

# --setup name -> get_config fn. Both siblings share the exact same processing logic below
# (flat model_learning, no phase routing) -- only the config/active_dims differ, so either file
# can now diagnose either variant. Does NOT extend to the dual-phase scripts: see
# action_sensitivity_baseline.py's own SETUPS comment for why.
SETUPS = {
    "single_phase_baseline":      _no_time_get_config,
    "single_phase_baseline_time": _time_get_config,
}
DEFAULT_RESULTS_ROOT = {
    "single_phase_baseline":      Path(_ROOT) / "results" / "single_phase_baseline",
    "single_phase_baseline_time": Path(_ROOT) / "results" / "single_phase_baseline_time",
}
DEFAULT_SETUP = "single_phase_baseline_time"

CHANNELS = [c for c in STATE_NAMES if c != "time"]
CHANNEL_IDX = [STATE_NAMES.index(c) for c in CHANNELS]
INPUT_NAMES = STATE_NAMES + ["action"]
# Relative sign threshold: a one-step delta smaller than SIGN_REL_EPS * (that channel's own
# GP-training-target std) counts as noise-level, not a real signed effect. Replaces a flat
# absolute epsilon that was 4x the size of typical late-phase P deltas (see channel_scales()).
SIGN_REL_EPS = 0.02
PHASE_COLORS = {"early": "C0", "mid": "C1", "late": "C2"}
# Keep textually identical to action_sensitivity_multi-phase.py's J_GRID_SHARED -- any cross-
# script "mid" comparison is only valid if both scripts probe the exact same j's.
J_GRID_SHARED = [2, 8, 15, 22, 30, 38, 44]


def load_agent(run_id, trial, setup=DEFAULT_SETUP, results_root=None):
    """Resolves run_id under results/<setup>/ (or as a full/relative path; --results_root
    overrides the default root for the chosen setup), reads seed/num_trials/fast back from
    that run's own note.txt, and reconstructs the trained GP model from log.pkl via the
    matching get_config for `setup` -- no more separately hand-typing SEED/NUM_TRIALS/FAST.
    Passing the wrong setup for a run's actual active_dims either crashes reconstruct_gp_agent
    on a load_state_dict shape mismatch or silently reconstructs a GP the checkpoint was never
    fit against -- see eval_single_phase_lib.reconstruct_gp_agent's docstring."""
    if setup not in SETUPS:
        raise ValueError(f"--setup must be one of {list(SETUPS)}, got '{setup}'")
    get_config_fn = SETUPS[setup]
    root = DEFAULT_RESULTS_ROOT[setup] if results_root is None else results_root
    run = lib.load_run(run_id, get_config_fn=get_config_fn, results_root=root)
    agent, idx = lib.reconstruct_gp_agent(run, idx=trial, get_config_fn=get_config_fn)
    print(f"[load_agent] {run.dir} trial {idx}  (setup={setup})")
    return agent, run, idx


def lengthscale_report(agent):
    """Per-GP lengthscales, named by looking up each GP's OWN `active_dims` rather than
    assuming every GP shares the same (STATE_DIM + ACTION_DIM)-length input -- kept generic
    even though every channel in this baseline uses the same (full, time-included) active_dims,
    so this stays a drop-in match for action_sensitivity.py's reports/plots.

    Reports action_ls_ratio = action_lengthscale / geomean(that GP's OTHER lengthscales) instead
    of a hardcoded absolute threshold: a fixed cutoff (e.g. >= 2.0) turned out to be true for
    every GP in every run regardless of architecture, carrying no information. The ratio is
    comparable across runs/architectures since it's relative to that GP's own other inputs."""
    ml = agent.model_learning
    rows = []
    for k in range(ml.num_gp):
        gp = ml.gp_list[k]
        if gp.flg_ARD:
            ls = torch.exp(gp.log_lengthscales_par)
        else:
            ls = torch.exp(gp.log_lengthscales_par) * torch.ones(
                gp.num_features, dtype=gp.dtype, device=gp.device)
        ls = ls.detach().cpu().numpy()
        active = gp.active_dims.detach().cpu().numpy()
        names = [INPUT_NAMES[i] for i in active]
        ls_by_name = dict(zip(names, ls))
        a_val = ls_by_name["action"]
        other_vals = [v for n, v in ls_by_name.items() if n != "action"]
        if other_vals:
            geomean_other = float(np.exp(np.mean(np.log(np.abs(other_vals)))))
            action_ls_ratio = a_val / geomean_other if geomean_other else float("inf")
        else:
            action_ls_ratio = float("nan")
        vals = " ".join(f"{n}={v:.3f}" for n, v in zip(names, ls))
        dropped = [n for n in INPUT_NAMES if n not in ls_by_name]
        drop_note = f"  (no {', '.join(dropped)} input)" if dropped else ""
        print(f"[lengthscales] {STATE_NAMES[k]:>10}: {vals}  action_ls_ratio={action_ls_ratio:.2f}"
              f"{drop_note}")
        row = {"gp": STATE_NAMES[k]}
        row.update({f"ls_{n}": ls_by_name.get(n, float("nan")) for n in INPUT_NAMES})
        row["action_lengthscale"] = a_val
        row["action_ls_ratio"] = action_ls_ratio
        rows.append(row)
    return rows


def action_deafness_report(agent, n_sweep=21):
    """Sweeps the action at THREE reference states -- early/mid/late time terciles of the GP's
    own training data -- instead of one state pooled over the whole batch. `time` is already a
    normalized state channel in gp_inputs (STATE_NAMES includes it), so terciles are read
    straight off it via TIME_IDX, mirroring the early/mid/late thirds used elsewhere in this file
    (phase_of) rather than collapsing very different points in the batch trajectory into one
    average state that resembles none of them.

    Reports sd_over_noise_ratio instead of a boolean: `spread` is peak-to-peak of the swept mean
    prediction, `sigma_n` is a 1-SD noise floor -- different units. For a monotone response swept
    uniformly over [-1, 1], the implied SD of that response is spread / (2*sqrt(3)) (SD of a
    uniform distribution), so the ratio implied_sd / sigma_n is the actual comparable quantity."""
    ml = agent.model_learning
    time_vals = ml.gp_inputs[:, TIME_IDX]
    buckets = {
        "early": time_vals < -1.0 / 3.0,
        "mid": (time_vals >= -1.0 / 3.0) & (time_vals < 1.0 / 3.0),
        "late": time_vals >= 1.0 / 3.0,
    }
    a_grid = torch.linspace(-1.0, 1.0, n_sweep, dtype=ml.dtype, device=ml.device)
    rows = []
    with torch.no_grad():
        for bucket, mask in buckets.items():
            if int(mask.sum()) == 0:
                print(f"[action-sweep] {bucket}: no training rows in this time tercile, skipped")
                continue
            ref_state = ml.gp_inputs[mask, :STATE_DIM].mean(dim=0, keepdim=True)
            for k in range(ml.num_gp):
                gp = ml.gp_list[k]
                deltas = []
                for a in a_grid:
                    u = torch.full((1, ACTION_DIM), float(a), dtype=ml.dtype, device=ml.device)
                    ns, _, _ = ml.get_next_state(current_state=ref_state, current_input=u,
                                                 particle_pred=False)
                    deltas.append(float((ns - ref_state)[0, k]))
                deltas = np.array(deltas)
                spread = float(deltas.max() - deltas.min())
                sigma_n = float(torch.sqrt(gp.get_sigma_n_2()).detach().cpu())
                implied_sd = spread / (2.0 * np.sqrt(3.0))
                ratio = implied_sd / sigma_n if sigma_n else float("inf")
                print(f"[action-sweep] {bucket}:{STATE_NAMES[k]:>10}: spread={spread:.5f} "
                      f"sigma_n={sigma_n:.5f} sd/noise={ratio:.2f}")
                rows.append({"bucket": bucket, "gp": STATE_NAMES[k], "spread": spread,
                            "sigma_n": sigma_n, "implied_sd": implied_sd,
                            "sd_over_noise_ratio": ratio})
    return rows


def channel_scales(agent):
    """Per-channel scale (std of that channel's GP training targets) used to make SIGN_REL_EPS a
    relative threshold instead of an absolute one applied identically across channels of very
    different typical magnitude (e.g. Wt deltas ~5e-3 vs late-phase P deltas ~4e-4)."""
    ml = agent.model_learning
    return {c: float(ml.gp_output_list[idx].std().detach().cpu())
           for c, idx in zip(CHANNELS, CHANNEL_IDX)}


def _rollout_base(seed, base_level):
    base = PenSimWrapper(seed_offset=0)
    base_policy = lambda state, decision_idx: np.array([base_level])
    s_base, _, _ = base.rollout(s0=None, policy=base_policy, T=CONTROL_H, dt=T_SAMPLING,
                                noise=None, seed=seed)
    return s_base


def _rollout_pert(seed, j, delta, base_level):
    pert = PenSimWrapper(seed_offset=0)
    pert_policy = lambda state, decision_idx: np.array(
        [base_level + delta if decision_idx >= j else base_level])
    s_pert, _, _ = pert.rollout(s0=None, policy=pert_policy, T=CONTROL_H, dt=T_SAMPLING,
                                noise=None, seed=seed)
    return s_pert


def model_action_effect(agent, state_j, delta, base_level=0.0):
    """Clamps both queried actions to [-1, 1], matching the simulator's own
    np.clip(action_norm, -1.0, 1.0) (pensim_wrapper.py). Without this, an off-manifold probe at
    e.g. base_level=0.6, delta=1.0 queried the GP at a=1.6 while the simulator (clipped) only
    ever reached a=1.0 -- comparing the model at a point the simulator can't."""
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    a0 = float(np.clip(base_level, -1.0, 1.0))
    ad = float(np.clip(base_level + delta, -1.0, 1.0))
    u0 = torch.full((1, ACTION_DIM), a0, dtype=ml.dtype, device=ml.device)
    ud = torch.full((1, ACTION_DIM), ad, dtype=ml.dtype, device=ml.device)
    with torch.no_grad():
        ns0, _, _ = ml.get_next_state(current_state=s, current_input=u0, particle_pred=False)
        nsd, _, _ = ml.get_next_state(current_state=s, current_input=ud, particle_pred=False)
    return (nsd - ns0)[0].detach().cpu().numpy()


def model_rollout(agent, state_j, action_level, n_steps):
    """Clamped to [-1, 1] -- see model_action_effect."""
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    a = float(np.clip(action_level, -1.0, 1.0))
    u = torch.full((1, ACTION_DIM), a, dtype=ml.dtype, device=ml.device)
    traj = [s]
    with torch.no_grad():
        for _ in range(n_steps):
            s, _, _ = ml.get_next_state(current_state=s, current_input=u, particle_pred=False)
            traj.append(s)
    return torch.cat(traj, dim=0).detach().cpu().numpy()


def phase_of(j, n_decisions):
    frac = j / max(1, n_decisions - 1)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def signed(x, scale):
    thresh = SIGN_REL_EPS * scale if scale > 0 else 0.0
    return 0.0 if abs(x) < thresh else float(np.sign(x))


def action_effect_table(agent, sim_seeds, js, deltas, horizons, base_level, channel_scale,
                        j_grid_shared=None):
    """Merged one-step (formerly sensitivity_table) + k-step sustained-feed (formerly
    sustained_divergence_table) action-effect table, averaged over `sim_seeds` instead of one
    ground-truth realisation.

    The two original functions both called true_action_effect for the identical (seed, j, delta,
    base_level) grid -- confirmed the k=1 rows of the old multistep CSV were bit-identical to the
    one-step table. Here each (seed, j, delta) combo runs exactly ONE perturbed simulator rollout
    (s_pert), which already contains the full trajectory to CONTROL_H, so every horizon's true
    k-step delta is read off it directly (s_pert[j+k] - s_base[j+k]) with no re-simulation. The
    baseline rollout s_base doesn't depend on j/delta at all (base_policy is constant), so it's
    cached once per seed for this base_level rather than recomputed per (j, delta).

    Returns (rows, rows_by_seed, ms_rows, loss_rows, phase_channel_stats, agree, total).
    `phase_channel_stats`/`agree`/`total` only aggregate rows where j is in `j_grid_shared` (the
    grid common to both single- and dual-phase scripts), so cross-script "mid" comparisons stay
    valid even when a script (dual-phase) probes extra pivot-adjacent j's.
    """
    n_decisions = int(CONTROL_H / T_SAMPLING)
    horizons_sorted = sorted(horizons)
    max_h = max(horizons_sorted)
    j_grid_shared = set(js) if j_grid_shared is None else set(j_grid_shared)

    base_cache = {}

    def get_base(seed):
        if seed not in base_cache:
            base_cache[seed] = _rollout_base(seed, base_level)
        return base_cache[seed]

    print(f"base_level={base_level}  sim_seeds={sim_seeds}")
    header = (f"{'phase':>6} {'j':>4} {'delta':>6} {'chan':>10} {'true_mean':>10} "
             f"{'model_mean':>10} {'sign_frac':>9}")
    print(header)

    rows, rows_by_seed, ms_rows = [], [], []
    first_loss = {}

    for j in js:
        phase = phase_of(j, n_decisions)
        in_shared = j in j_grid_shared
        for delta in deltas:
            per_true = {c: [] for c in CHANNELS}
            per_model = {c: [] for c in CHANNELS}
            per_true_k = {c: {k: [] for k in horizons_sorted} for c in CHANNELS}
            per_model_k = {c: {k: [] for k in horizons_sorted} for c in CHANNELS}

            for seed in sim_seeds:
                s_base = get_base(seed)
                s_pert = _rollout_pert(seed, j, delta, base_level)
                state_j = s_base[j]

                model_delta = model_action_effect(agent, state_j, delta, base_level)
                model_base_traj = model_rollout(agent, state_j, base_level, max_h)
                model_pert_traj = model_rollout(agent, state_j, base_level + delta, max_h)

                for c, idx in zip(CHANNELS, CHANNEL_IDX):
                    td1 = float(s_pert[j + 1, idx] - s_base[j + 1, idx])
                    md1 = float(model_delta[idx])
                    per_true[c].append(td1)
                    per_model[c].append(md1)
                    rows_by_seed.append({"seed": seed, "phase": phase, "j": j, "delta": delta,
                                         "channel": c, "true_delta": td1, "model_delta": md1,
                                         "base_level": base_level, "in_shared_grid": in_shared})

                    for k in horizons_sorted:
                        if j + k >= s_base.shape[0]:
                            continue
                        tdk = float(s_pert[j + k, idx] - s_base[j + k, idx])
                        mdk = float(model_pert_traj[k, idx] - model_base_traj[k, idx])
                        per_true_k[c][k].append(tdk)
                        per_model_k[c][k].append(mdk)

            for c in CHANNELS:
                scale = channel_scale[c]
                t_arr, m_arr = np.array(per_true[c]), np.array(per_model[c])
                sign_frac = float(np.mean([signed(t, scale) == signed(m, scale)
                                          for t, m in zip(t_arr, m_arr)]))
                rows.append({
                    "phase": phase, "j": j, "delta": delta, "channel": c,
                    "true_delta_mean": float(t_arr.mean()), "true_delta_std": float(t_arr.std()),
                    "model_delta_mean": float(m_arr.mean()), "model_delta_std": float(m_arr.std()),
                    "sign_match_frac": sign_frac, "n_seeds": len(sim_seeds),
                    "base_level": base_level, "in_shared_grid": in_shared,
                })
                print(f"{phase:>6} {j:>4d} {delta:>6.2f} {c:>10} {t_arr.mean():>10.5f} "
                     f"{m_arr.mean():>10.5f} {sign_frac:>9.2f}")

                for k in horizons_sorted:
                    tk, mk = np.array(per_true_k[c][k]), np.array(per_model_k[c][k])
                    if tk.size == 0:
                        continue
                    sign_frac_k = float(np.mean([signed(t, scale) == signed(m, scale)
                                                for t, m in zip(tk, mk)]))
                    ms_rows.append({
                        "phase": phase, "j": j, "delta": delta, "k": k, "channel": c,
                        "true_delta_mean": float(tk.mean()), "model_delta_mean": float(mk.mean()),
                        "abs_error": float(np.mean(np.abs(mk - tk))), "sign_match_frac": sign_frac_k,
                        "n_seeds": len(sim_seeds), "in_shared_grid": in_shared,
                    })
                    key = (j, delta, c)
                    # "lost" = majority of seeds now disagree in sign with the model
                    if sign_frac_k < 0.5 and key not in first_loss:
                        first_loss[key] = k

    shared_rows = [r for r in rows if r["in_shared_grid"]]
    agree = sum(r["sign_match_frac"] for r in shared_rows)
    total = len(shared_rows)
    phase_channel_stats = {}
    for r in shared_rows:
        key = (r["phase"], r["channel"])
        a, t = phase_channel_stats.get(key, (0.0, 0))
        phase_channel_stats[key] = (a + r["sign_match_frac"], t + 1)

    loss_rows = []
    for j in js:
        for delta in deltas:
            for c in CHANNELS:
                loss_rows.append({"j": j, "delta": delta, "channel": c,
                                  "first_sign_loss_k": first_loss.get((j, delta, c), -1)})

    print()
    print(f"{'phase':>6} {'chan':>10} {'mean sign-agreement':>20}")
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        print(f"{phase:>6} {c:>10} {a / t:.2f}  (n={t} combos)")
    print()
    print(f"HEADLINE mean sign-agreement fraction (shared grid only): {agree / total:.3f}  "
         f"(n={total} combos x {len(sim_seeds)} seeds each)")
    return rows, rows_by_seed, ms_rows, loss_rows, phase_channel_stats, agree, total


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {path}")


def plot_gp_diagnostics(ls_rows, deaf_rows, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    mat = np.array([[r[f"ls_{n}"] for n in INPUT_NAMES] for r in ls_rows])
    im = ax1.imshow(mat, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(INPUT_NAMES)))
    ax1.set_xticklabels(INPUT_NAMES, rotation=45, ha="right")
    ax1.set_yticks(range(len(ls_rows)))
    ax1.set_yticklabels([r["gp"] for r in ls_rows])
    vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax1.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat[i, j] < vmax * 0.6 else "black", fontsize=8)
    ax1.set_title("decoded lengthscales per GP\n(action_ls_ratio in the CSV: action lengthscale / "
                 "geomean of that GP's other lengthscales)")
    fig.colorbar(im, ax=ax1)

    labels_d = [f"{r['bucket']}:{r['gp']}" for r in deaf_rows]
    x = np.arange(len(deaf_rows))
    width = 0.35
    spreads = [r["spread"] for r in deaf_rows]
    sigmas = [r["sigma_n"] for r in deaf_rows]
    ax2.bar(x - width / 2, spreads, width, color="C0", label="action-sweep spread")
    ax2.bar(x + width / 2, sigmas, width, color="0.6", label="sigma_n")
    for xi, r in zip(x, deaf_rows):
        ax2.text(xi, max(r["spread"], r["sigma_n"]) * 1.02, f"{r['sd_over_noise_ratio']:.2f}x",
                 ha="center", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_d, rotation=45, ha="right", fontsize=8)
    ax2.set_title("action-sweep spread vs noise floor\n(label = implied-SD / sigma_n ratio, "
                 "swept at early/mid/late time terciles)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_sensitivity(rows, out_path, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, c in zip(axes.ravel(), CHANNELS):
        crows = [r for r in rows if r["channel"] == c]
        td = np.array([r["true_delta_mean"] for r in crows])
        td_err = np.array([r["true_delta_std"] for r in crows])
        md = np.array([r["model_delta_mean"] for r in crows])
        md_err = np.array([r["model_delta_std"] for r in crows])
        for ph, col in PHASE_COLORS.items():
            m = np.array([r["phase"] == ph for r in crows])
            if not m.any():
                continue
            ax.errorbar(td[m], md[m], xerr=td_err[m], yerr=md_err[m], fmt="o", color=col,
                       label=ph, alpha=0.8, zorder=3, capsize=2, ms=5, lw=1)
        lim = [min(td.min(), md.min()), max(td.max(), md.max())]
        ax.plot(lim, lim, "k--", lw=1, label="y=x", zorder=2)
        ax.axhline(0, color="0.7", lw=0.8, zorder=1)
        ax.axvline(0, color="0.7", lw=0.8, zorder=1)
        mean_sign_frac = float(np.mean([r["sign_match_frac"] for r in crows])) if crows else float("nan")
        n_seeds = crows[0]["n_seeds"] if crows else 0
        rho, p_value = stats.spearmanr(td, md) if len(td) > 1 else (float("nan"), float("nan"))
        ax.set_title(f"{c}: mean sign-agreement {mean_sign_frac:.2f} "
                    f"(n={len(crows)} combos x {n_seeds} seeds), rho={rho:+.2f} (p={p_value:.3f})")
        ax.set_xlabel("true delta (mean +/- SD over seeds)")
        ax.set_ylabel("model delta (mean +/- SD over seeds)")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"true vs model one-step action sensitivity{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_first_sign_loss(loss_rows, horizons, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CHANNELS))
    bins = horizons + [-1]
    n_bins = len(bins)
    width = 0.8 / n_bins
    for i, k in enumerate(bins):
        counts = [sum(1 for r in loss_rows if r["channel"] == c and r["first_sign_loss_k"] == k)
                 for c in CHANNELS]
        label = "never lost" if k == -1 else f"lost by k={k}"
        color = "0.3" if k == -1 else None
        ax.bar(x + (i - (n_bins - 1) / 2) * width, counts, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("count of (j, delta) combos")
    ax.set_title(f"first horizon at which the majority-of-seeds sign flips{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def main(run_id, trial=None, setup=DEFAULT_SETUP, results_root=None):
    N_SIM_SEEDS = 5
    SIM_SEEDS = [424242 + i for i in range(N_SIM_SEEDS)]
    J_GRID = J_GRID_SHARED
    DELTAS = [-1.0, -0.2, 0.2, 1.0]
    HORIZONS = [1, 20]
    OFFMANIFOLD_LEVEL = 0.6

    agent, run, _idx = load_agent(run_id, trial, setup=setup, results_root=results_root)
    out_dir = run.dir

    ml = agent.model_learning
    X = ml.gp_inputs          # [N, STATE_DIM + ACTION_DIM], shared across all GPs
    for name, m, s in zip(INPUT_NAMES, X.mean(dim=0).tolist(), X.std(dim=0).tolist()):
        print(f"{name:>10}: mean={m:8.4f}  std={s:8.4f}")

    print("\n--- lengthscale report ---")
    ls_rows = lengthscale_report(agent)
    save_csv(ls_rows, out_dir / "action_sensitivity_lengthscales.csv",
             ["gp"] + [f"ls_{n}" for n in INPUT_NAMES] + ["action_lengthscale", "action_ls_ratio"])

    print("\n--- action-deafness report ---")
    deaf_rows = action_deafness_report(agent)
    save_csv(deaf_rows, out_dir / "action_sensitivity_deafness.csv",
             ["bucket", "gp", "spread", "sigma_n", "implied_sd", "sd_over_noise_ratio"])

    channel_scale = channel_scales(agent)
    print("\n--- per-channel sign threshold scale (std of GP training targets) ---")
    for c in CHANNELS:
        print(f"{c:>10}: scale={channel_scale[c]:.5f}  eps={SIGN_REL_EPS * channel_scale[c]:.6f}")

    print(f"\n--- true vs model action sensitivity + sustained-feed divergence "
         f"(on-manifold, a=0 baseline, {N_SIM_SEEDS} sim seeds) ---")
    (rows, rows_by_seed, ms_rows, loss_rows,
     phase_channel_stats, agree, total) = action_effect_table(
        agent, SIM_SEEDS, J_GRID, DELTAS, HORIZONS, base_level=0.0,
        channel_scale=channel_scale, j_grid_shared=set(J_GRID))
    save_csv(rows, out_dir / "action_sensitivity_table.csv",
             ["phase", "j", "delta", "channel", "true_delta_mean", "true_delta_std",
              "model_delta_mean", "model_delta_std", "sign_match_frac", "n_seeds", "base_level",
              "in_shared_grid"])
    save_csv(rows_by_seed, out_dir / "action_sensitivity_table_by_seed.csv",
             ["seed", "phase", "j", "delta", "channel", "true_delta", "model_delta", "base_level",
              "in_shared_grid"])

    summary_rows = []
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        summary_rows.append({"phase": phase, "channel": c, "mean_sign_agreement_frac": a / t,
                             "n_combos": t})
    summary_rows.append({"phase": "ALL", "channel": "ALL",
                         "mean_sign_agreement_frac": agree / total, "n_combos": total})
    save_csv(summary_rows, out_dir / "action_sensitivity_summary.csv",
             ["phase", "channel", "mean_sign_agreement_frac", "n_combos"])

    plot_gp_diagnostics(ls_rows, deaf_rows, out_dir / "action_sensitivity_gp_diagnostics.png")
    plot_sensitivity(rows, out_dir / "action_sensitivity_scatter.png")

    save_csv(ms_rows, out_dir / "action_sensitivity_multistep.csv",
             ["phase", "j", "delta", "k", "channel", "true_delta_mean", "model_delta_mean",
              "abs_error", "sign_match_frac", "n_seeds", "in_shared_grid"])
    save_csv(loss_rows, out_dir / "action_sensitivity_multistep_first_loss.csv",
             ["j", "delta", "channel", "first_sign_loss_k"])
    plot_first_sign_loss(loss_rows, HORIZONS, out_dir / "action_sensitivity_multistep_first_loss.png")

    print(f"\n--- true vs model action sensitivity + sustained-feed divergence "
         f"(off-manifold, a={OFFMANIFOLD_LEVEL} baseline) ---")
    (om_rows, om_rows_by_seed, om_ms_rows, om_loss_rows,
     om_phase_channel_stats, om_agree, om_total) = action_effect_table(
        agent, SIM_SEEDS, J_GRID, DELTAS, HORIZONS, base_level=OFFMANIFOLD_LEVEL,
        channel_scale=channel_scale, j_grid_shared=set(J_GRID))
    save_csv(om_rows, out_dir / "action_sensitivity_offmanifold_table.csv",
             ["phase", "j", "delta", "channel", "true_delta_mean", "true_delta_std",
              "model_delta_mean", "model_delta_std", "sign_match_frac", "n_seeds", "base_level",
              "in_shared_grid"])
    save_csv(om_rows_by_seed, out_dir / "action_sensitivity_offmanifold_table_by_seed.csv",
             ["seed", "phase", "j", "delta", "channel", "true_delta", "model_delta", "base_level",
              "in_shared_grid"])
    save_csv(om_ms_rows, out_dir / "action_sensitivity_offmanifold_multistep.csv",
             ["phase", "j", "delta", "k", "channel", "true_delta_mean", "model_delta_mean",
              "abs_error", "sign_match_frac", "n_seeds", "in_shared_grid"])
    save_csv(om_loss_rows, out_dir / "action_sensitivity_offmanifold_multistep_first_loss.csv",
             ["j", "delta", "channel", "first_sign_loss_k"])

    om_summary_rows = []
    for (phase, c), (a, t) in sorted(om_phase_channel_stats.items()):
        om_summary_rows.append({"phase": phase, "channel": c, "mean_sign_agreement_frac": a / t,
                                "n_combos": t})
    om_summary_rows.append({"phase": "ALL", "channel": "ALL",
                            "mean_sign_agreement_frac": om_agree / om_total, "n_combos": om_total})
    save_csv(om_summary_rows, out_dir / "action_sensitivity_offmanifold_summary.csv",
             ["phase", "channel", "mean_sign_agreement_frac", "n_combos"])
    plot_sensitivity(om_rows, out_dir / "action_sensitivity_offmanifold_scatter.png",
                     title_suffix=f" (off-manifold, a={OFFMANIFOLD_LEVEL} baseline)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", type=str,
                   help="run to evaluate, e.g. 'seed0_1' (resolved under results/<setup>/, see "
                        "--setup) or a full/relative path to a run folder")
    p.add_argument("--trial", type=int, default=None,
                   help="which trial's GP model to diagnose (default: last saved)")
    p.add_argument("--setup", choices=list(SETUPS), default=DEFAULT_SETUP,
                   help="which single-phase-baseline variant this run was trained with -- "
                        "'single_phase_baseline_time' (time re-added as a GP input, this "
                        "script's own default) or 'single_phase_baseline' (time dropped). "
                        "Picking the wrong one either crashes on a GP "
                        "active_dims/lengthscales shape mismatch or silently reconstructs a "
                        "GP the checkpoint was never fit against.")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under (default: "
                        "results/<setup>/)")
    args = p.parse_args()
    main(run_id=args.run_id, trial=args.trial, setup=args.setup, results_root=args.results_root)

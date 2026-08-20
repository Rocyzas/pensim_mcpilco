"""
PYTHONPATH=.. python "evaluations/action_sensitivity_multi-phase_baseline_time.py" seed1_1

(Direct file execution, not `-m` -- the hyphen in this filename isn't a valid Python module
identifier, so it can't be imported or run as `-m evaluations.action_sensitivity_multi-phase_baseline_time`.)

Same diagnostic as action_sensitivity_multi-phase.py, but for config_dual_phase_baseline_time
runs (plain RBF on every channel in BOTH phase1/phase2, WITH time kept as a GP input regressor
-- see config_dual_phase_baseline_time.py / model_learning_baseline.py). Only load_agent
differs: it resolves run_id under results/dual_phase_baseline_time/ and reconstructs the GP
model via config_dual_phase_baseline_time.get_config -- using config_dual_phase.get_config or
config_dual_phase_baseline.get_config here would silently rebuild the wrong architecture
(their parameter sets can overlap enough that load_state_dict succeeds without error -- see
eval_multi_phase_lib.reconstruct_gp_agent's docstring; a full active_dims mismatch instead
raises a shape-mismatch RuntimeError, which is how this gap was originally found).

Takes only a run id (resolved under results/dual_phase_baseline_time/, or a full/relative
path) -- seed/num_trials/fast/pivot_hours and the trained GPs are read back from that run's
own note.txt/log.pkl via eval_multi_phase_lib.load_run/reconstruct_gp_agent.

Dual-phase counterpart to evaluations/action_sensitivity_baseline_time.py: same tests (GP
lengthscale report, action-deafness sweep, true-vs-model one-step and sustained-feed action
sensitivity), adapted for DualPhaseModelLearning -- see action_sensitivity_multi-phase.py's
module docstring for the three real (not just renamed) differences from the single-phase
script.
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

import evaluations.eval_multi_phase_lib as lib
from mcpilco.config_dual_phase_baseline import get_config as _no_time_get_config
from mcpilco.config_dual_phase_baseline_time import get_config as _time_get_config
from mcpilco.pensim_wrapper import (PenSimWrapper, STATE_NAMES, STATE_DIM, ACTION_DIM,
                                    CONTROL_H, T_SAMPLING)

# --setup name -> get_config fn. Both siblings share the exact same DualPhaseModelLearning
# processing logic below -- only the config/active_dims differ, so either file can now
# diagnose either dual-phase-baseline variant. Does NOT extend to the single-phase scripts:
# see action_sensitivity_baseline.py's own SETUPS comment for why.
SETUPS = {
    "dual_phase_baseline":      _no_time_get_config,
    "dual_phase_baseline_time": _time_get_config,
}
DEFAULT_RESULTS_ROOT = {
    "dual_phase_baseline":      Path(_ROOT) / "results" / "dual_phase_baseline",
    "dual_phase_baseline_time": Path(_ROOT) / "results" / "dual_phase_baseline_time",
}
DEFAULT_SETUP = "dual_phase_baseline_time"

CHANNELS = [c for c in STATE_NAMES if c != "time"]
CHANNEL_IDX = [STATE_NAMES.index(c) for c in CHANNELS]
INPUT_NAMES = STATE_NAMES + ["action"]
# Relative sign threshold: a one-step delta smaller than SIGN_REL_EPS * (that channel's own
# GP-training-target std, in the phase whose GP actually answered the probe) counts as
# noise-level, not a real signed effect. Replaces a flat absolute epsilon.
SIGN_REL_EPS = 0.02
PHASE_COLORS = {"early": "C0", "mid": "C1", "late": "C2"}
PHASE_LABELS = ["phase1", "phase2"]
# Keep textually identical to action_sensitivity.py's J_GRID_SHARED -- any cross-script "mid"/
# batch_phase comparison is only valid over the exact same j's.
J_GRID_SHARED = [2, 8, 15, 22, 30, 38, 44]


def gp_phase_channel(k, state_dim=STATE_DIM):
    """Composite GP index k (0..2*STATE_DIM-1) -> (phase label, channel name). gp_list is
    ordered [phase1 x STATE_DIM, phase2 x STATE_DIM] -- see DualPhaseModelLearning.gp_list."""
    phase_idx, ch_idx = divmod(k, state_dim)
    return PHASE_LABELS[phase_idx], STATE_NAMES[ch_idx]


def pivot_step_of(run):
    """Decision index at which this run's DualPhaseModelLearning switches phases, read from
    the run's OWN pivot_hours (note.txt) rather than assuming the module default -- a run may
    have used a non-default --pivot_hours."""
    return int(round(run.pivot_hours / T_SAMPLING))


def model_phase(j, pivot_step):
    """Coarse "which phase dominates" label for decision j -- exact far from the pivot (the
    composite is ~100% one phase there), but only an approximation inside the sigmoid blend
    window (see DualPhaseModelLearning._blend_weight): DualPhaseModelLearning.get_next_state no
    longer routes to exactly one phase, it BLENDS both with a smoothly-varying weight, so a
    probe at e.g. j==pivot_step is really a ~50/50 mix, not a hard cutover. Still useful for
    reporting/bucketing (this is what action_effect_table groups by), just don't read it as
    "the model that answered this probe" near the boundary."""
    return "phase1" if j < pivot_step else "phase2"


def load_agent(run_id, trial, setup=DEFAULT_SETUP, results_root=None):
    """Resolves run_id under results/<setup>/ (or as a full/relative path; --results_root
    overrides the default root for the chosen setup), reads seed/num_trials/fast/pivot_hours
    back from that run's own note.txt, and reconstructs the trained GP model from log.pkl via
    the matching get_config for `setup` (plain RBF for every channel on both phases, see
    model_learning_baseline.py) -- NOT the regular config_dual_phase, which would silently
    reattach the Wt/Viscosity prior means this run was trained without. Passing the wrong
    setup (no-time vs time) for a run's actual active_dims crashes reconstruct_gp_agent on a
    load_state_dict shape mismatch -- see eval_multi_phase_lib.reconstruct_gp_agent's
    docstring; this is literally how the gap this script closes was originally found."""
    if setup not in SETUPS:
        raise ValueError(f"--setup must be one of {list(SETUPS)}, got '{setup}'")
    get_config_fn = SETUPS[setup]
    root = DEFAULT_RESULTS_ROOT[setup] if results_root is None else results_root
    run = lib.load_run(run_id, get_config_fn=get_config_fn, results_root=root)
    agent, idx = lib.reconstruct_gp_agent(run, idx=trial, get_config_fn=get_config_fn)
    print(f"[load_agent] {run.dir} trial {idx}  pivot_hours={run.pivot_hours:g}  (setup={setup})")
    return agent, run, idx


def lengthscale_report(agent):
    """Per-GP lengthscales, named by looking up each GP's OWN `active_dims` AND by which phase
    it belongs to (gp index k -> (phase, channel) via gp_phase_channel). Every GP in this
    baseline uses the same (full, time-included) active_dims -- kept generic anyway so this
    stays a drop-in match for action_sensitivity_multi-phase.py's reports/plots.

    Reports action_ls_ratio = action_lengthscale / geomean(that GP's OTHER lengthscales) instead
    of a hardcoded absolute threshold -- see action_sensitivity.py's lengthscale_report for why
    (a fixed cutoff turned out to be true for every GP in every run, carrying no information)."""
    ml = agent.model_learning
    rows = []
    for k in range(ml.num_gp):
        phase, channel = gp_phase_channel(k)
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
        print(f"[lengthscales] {phase}:{channel:>10}: {vals}  action_ls_ratio={action_ls_ratio:.2f}"
              f"{drop_note}")
        row = {"phase": phase, "gp": channel}
        row.update({f"ls_{n}": ls_by_name.get(n, float("nan")) for n in INPUT_NAMES})
        row["action_lengthscale"] = a_val
        row["action_ls_ratio"] = action_ls_ratio
        rows.append(row)
    return rows


def action_deafness_report(agent, n_sweep=21):
    """Per-phase: sweep the action at THAT phase's own mean training state (NOT a state
    averaged across both phases -- they cover different batch-time distributions), calling
    `sub.get_next_state` directly on phase1/phase2 (bypassing the composite's step counter
    entirely, since we already know exactly which phase we mean to probe).

    Reports sd_over_noise_ratio instead of a boolean -- see action_sensitivity.py's
    action_deafness_report for the derivation (implied_sd = spread / (2*sqrt(3))."""
    ml = agent.model_learning
    a_grid = torch.linspace(-1.0, 1.0, n_sweep, dtype=ml.dtype, device=ml.device)
    rows = []
    with torch.no_grad():
        for phase, sub in [("phase1", ml.phase1), ("phase2", ml.phase2)]:
            ref_state = sub.gp_inputs[:, :STATE_DIM].mean(dim=0, keepdim=True)
            for k in range(sub.num_gp):
                gp = sub.gp_list[k]
                deltas = []
                for a in a_grid:
                    u = torch.full((1, ACTION_DIM), float(a), dtype=sub.dtype, device=sub.device)
                    ns, _, _ = sub.get_next_state(current_state=ref_state, current_input=u,
                                                  particle_pred=False)
                    deltas.append(float((ns - ref_state)[0, k]))
                deltas = np.array(deltas)
                spread = float(deltas.max() - deltas.min())
                sigma_n = float(torch.sqrt(gp.get_sigma_n_2()).detach().cpu())
                implied_sd = spread / (2.0 * np.sqrt(3.0))
                ratio = implied_sd / sigma_n if sigma_n else float("inf")
                channel = STATE_NAMES[k]
                print(f"[action-sweep] {phase}:{channel:>10}: spread={spread:.5f} "
                      f"sigma_n={sigma_n:.5f} sd/noise={ratio:.2f}")
                rows.append({"phase": phase, "gp": channel, "spread": spread, "sigma_n": sigma_n,
                            "implied_sd": implied_sd, "sd_over_noise_ratio": ratio})
    return rows


def channel_scales(agent):
    """Per-(phase, channel) scale (std of that phase's own GP training targets), used to make
    SIGN_REL_EPS a relative threshold. The deployed model at decision j uses model_phase(j,
    pivot_step)'s own GP, so the sign threshold for a probe at j uses that same phase's scale."""
    ml = agent.model_learning
    scales = {}
    for phase, sub in [("phase1", ml.phase1), ("phase2", ml.phase2)]:
        for c, idx in zip(CHANNELS, CHANNEL_IDX):
            scales[(phase, c)] = float(sub.gp_output_list[idx].std().detach().cpu())
    return scales


def _rollout_base(seed, base_level):
    """Pure real-simulator comparison -- unaffected by dual-phase (the physics don't know or
    care how the LEARNED model is structured)."""
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


def _bm_max0_at(ml, s_base, j):
    """Running-max biomass over the REAL trajectory up to decision j, for seeding a jump.

    Under --onEachRollout the blend weight is a function of biomass ACCUMULATED SINCE THE START
    OF THE BATCH, so probing decision j via reset_step_counter(j) -- which is what every helper
    below does, deliberately, to avoid replaying the whole batch -- starts with an empty running
    max and would under-weight phase 2 at exactly the late-batch decisions these probes target.
    DualPhaseModelLearning._blend_weight raises rather than return that silently wrong weight,
    so the value has to be supplied here. s_base is the real episode's normalised state
    trajectory, which is exactly what the running max is defined over.

    Returns None (and reset_step_counter then behaves as before) whenever the flag is off, so
    the time-sigmoid path is untouched."""
    if not getattr(ml, "on_each_rollout", False):
        return None
    from mcpilco.model_learning_dual_phase import _bm_from_states
    bm = _bm_from_states(np.asarray(s_base)[:j + 1])
    return torch.tensor([float(np.max(bm))], dtype=ml.dtype, device=ml.device)


def model_action_effect(agent, state_j, delta, j, base_level=0.0, bm_max0=None):
    """Queries the DEPLOYED composite at decision j -- reset_step_counter(j) before EACH
    one-off call (baseline and perturbed are independent queries at the same j, not a
    sequential pair) so the phase router answers with whichever phase real rollouts would
    actually use at that decision. Both queried actions are clamped to [-1, 1], matching the
    simulator's own np.clip (pensim_wrapper.py) -- without this an off-manifold probe queries
    the GP at a point the simulator can never reach."""
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    a0 = float(np.clip(base_level, -1.0, 1.0))
    ad = float(np.clip(base_level + delta, -1.0, 1.0))
    u0 = torch.full((1, ACTION_DIM), a0, dtype=ml.dtype, device=ml.device)
    ud = torch.full((1, ACTION_DIM), ad, dtype=ml.dtype, device=ml.device)
    with torch.no_grad():
        ml.reset_step_counter(j, bm_max0=bm_max0)
        ns0, _, _ = ml.get_next_state(current_state=s, current_input=u0, particle_pred=False)
        ml.reset_step_counter(j, bm_max0=bm_max0)
        nsd, _, _ = ml.get_next_state(current_state=s, current_input=ud, particle_pred=False)
    return (nsd - ns0)[0].detach().cpu().numpy()


def model_rollout(agent, state_j, action_level, n_steps, j, bm_max0=None):
    """Sequential walk starting at decision j -- reset_step_counter(j) ONCE before the loop;
    the composite's own auto-increment then correctly crosses the pivot mid-rollout in exactly
    the way a real deployment rollout would. Clamped to [-1, 1] -- see model_action_effect."""
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    a = float(np.clip(action_level, -1.0, 1.0))
    u = torch.full((1, ACTION_DIM), a, dtype=ml.dtype, device=ml.device)
    traj = [s]
    ml.reset_step_counter(j, bm_max0=bm_max0)
    with torch.no_grad():
        for _ in range(n_steps):
            s, _, _ = ml.get_next_state(current_state=s, current_input=u, particle_pred=False)
            traj.append(s)
    return torch.cat(traj, dim=0).detach().cpu().numpy()


def phase_of(j, n_decisions):
    """Coarse early/mid/late batch-thirds bucketing -- kept alongside model_phase() since the
    two partitions differ (the pivot need not land on a batch third)."""
    frac = j / max(1, n_decisions - 1)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def signed(x, scale):
    thresh = SIGN_REL_EPS * scale if scale > 0 else 0.0
    return 0.0 if abs(x) < thresh else float(np.sign(x))


def action_effect_table(agent, sim_seeds, js, deltas, horizons, pivot_step, base_level,
                        channel_scale, j_grid_shared=None):
    """Merged one-step (formerly sensitivity_table) + k-step sustained-feed (formerly
    sustained_divergence_table) action-effect table, averaged over `sim_seeds`. See
    action_sensitivity.py's action_effect_table docstring for the full rationale (both original
    functions ran an identical (seed, j, delta, base_level) grid of simulator rollouts; here each
    combo runs exactly ONE perturbed rollout, whose full trajectory already covers every horizon,
    plus one cached baseline rollout per seed).

    Returns (rows, rows_by_seed, ms_rows, loss_rows, phase_channel_stats,
    model_phase_channel_stats, agree, total). `phase_channel_stats`/`agree`/`total` (the
    batch_phase view) only aggregate rows where j is in `j_grid_shared`, so cross-script "mid"
    comparisons with action_sensitivity.py stay valid despite the extra pivot-adjacent j's this
    script probes. `model_phase_channel_stats` (the dual-phase-routing view) has no single-phase
    counterpart, so it aggregates every row.
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
    header = (f"{'bphase':>6} {'mphase':>6} {'j':>4} {'delta':>6} {'chan':>10} {'true_mean':>10} "
             f"{'model_mean':>10} {'sign_frac':>9}")
    print(header)

    rows, rows_by_seed, ms_rows = [], [], []
    first_loss = {}

    for j in js:
        bphase = phase_of(j, n_decisions)
        mphase = model_phase(j, pivot_step)
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
                # Seed the running max from the REAL trajectory: these three calls all jump
                # straight to decision j (see _bm_max0_at). No-op unless --onEachRollout.
                bm0 = _bm_max0_at(agent.model_learning, s_base, j)

                model_delta = model_action_effect(agent, state_j, delta, j, base_level, bm_max0=bm0)
                model_base_traj = model_rollout(agent, state_j, base_level, max_h, j, bm_max0=bm0)
                model_pert_traj = model_rollout(agent, state_j, base_level + delta, max_h, j, bm_max0=bm0)

                for c, idx in zip(CHANNELS, CHANNEL_IDX):
                    td1 = float(s_pert[j + 1, idx] - s_base[j + 1, idx])
                    md1 = float(model_delta[idx])
                    per_true[c].append(td1)
                    per_model[c].append(md1)
                    rows_by_seed.append({"seed": seed, "batch_phase": bphase,
                                         "model_phase": mphase, "j": j, "delta": delta,
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
                scale = channel_scale[(mphase, c)]
                t_arr, m_arr = np.array(per_true[c]), np.array(per_model[c])
                sign_frac = float(np.mean([signed(t, scale) == signed(m, scale)
                                          for t, m in zip(t_arr, m_arr)]))
                rows.append({
                    "batch_phase": bphase, "model_phase": mphase, "j": j, "delta": delta,
                    "channel": c, "true_delta_mean": float(t_arr.mean()),
                    "true_delta_std": float(t_arr.std()), "model_delta_mean": float(m_arr.mean()),
                    "model_delta_std": float(m_arr.std()), "sign_match_frac": sign_frac,
                    "n_seeds": len(sim_seeds), "base_level": base_level,
                    "in_shared_grid": in_shared,
                })
                print(f"{bphase:>6} {mphase:>6} {j:>4d} {delta:>6.2f} {c:>10} "
                     f"{t_arr.mean():>10.5f} {m_arr.mean():>10.5f} {sign_frac:>9.2f}")

                for k in horizons_sorted:
                    tk, mk = np.array(per_true_k[c][k]), np.array(per_model_k[c][k])
                    if tk.size == 0:
                        continue
                    sign_frac_k = float(np.mean([signed(t, scale) == signed(m, scale)
                                                for t, m in zip(tk, mk)]))
                    ms_rows.append({
                        "model_phase": mphase, "j": j, "delta": delta, "k": k, "channel": c,
                        "true_delta_mean": float(tk.mean()), "model_delta_mean": float(mk.mean()),
                        "abs_error": float(np.mean(np.abs(mk - tk))),
                        "sign_match_frac": sign_frac_k, "n_seeds": len(sim_seeds),
                        "in_shared_grid": in_shared,
                    })
                    key = (j, delta, c)
                    # "lost" = majority of seeds now disagree in sign with the model
                    if sign_frac_k < 0.5 and key not in first_loss:
                        first_loss[key] = k

    shared_rows = [r for r in rows if r["in_shared_grid"]]
    agree = sum(r["sign_match_frac"] for r in shared_rows)
    total = len(shared_rows)
    phase_channel_stats = {}
    model_phase_channel_stats = {}
    for r in rows:
        mkey = (r["model_phase"], r["channel"])
        a2, t2 = model_phase_channel_stats.get(mkey, (0.0, 0))
        model_phase_channel_stats[mkey] = (a2 + r["sign_match_frac"], t2 + 1)
    for r in shared_rows:
        key = (r["batch_phase"], r["channel"])
        a, t = phase_channel_stats.get(key, (0.0, 0))
        phase_channel_stats[key] = (a + r["sign_match_frac"], t + 1)

    loss_rows = []
    for j in js:
        mphase = model_phase(j, pivot_step)
        for delta in deltas:
            for c in CHANNELS:
                loss_rows.append({"model_phase": mphase, "j": j, "delta": delta, "channel": c,
                                  "first_sign_loss_k": first_loss.get((j, delta, c), -1)})

    print()
    print(f"{'bphase':>6} {'chan':>10} {'mean sign-agreement':>20}   <- shared-grid only")
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        print(f"{phase:>6} {c:>10} {a / t:.2f}  (n={t} combos)")

    print()
    print(f"{'mphase':>6} {'chan':>10} {'mean sign-agreement':>20}   <- dual-phase model routing")
    for (phase, c), (a, t) in sorted(model_phase_channel_stats.items()):
        print(f"{phase:>6} {c:>10} {a / t:.2f}  (n={t} combos)")

    print()
    print(f"HEADLINE mean sign-agreement fraction (shared grid only): {agree / total:.3f}  "
         f"(n={total} combos x {len(sim_seeds)} seeds each)")
    return (rows, rows_by_seed, ms_rows, loss_rows, phase_channel_stats,
           model_phase_channel_stats, agree, total)


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {path}")


def plot_gp_diagnostics(ls_rows, deaf_rows, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    labels = [f"{r['phase']}:{r['gp']}" for r in ls_rows]
    mat = np.array([[r[f"ls_{n}"] for n in INPUT_NAMES] for r in ls_rows])
    im = ax1.imshow(mat, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(INPUT_NAMES)))
    ax1.set_xticklabels(INPUT_NAMES, rotation=45, ha="right")
    ax1.set_yticks(range(len(ls_rows)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.axhline(STATE_DIM - 0.5, color="white", lw=2)
    vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax1.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat[i, j] < vmax * 0.6 else "black", fontsize=7)
    ax1.set_title("decoded lengthscales per (phase, GP)\n(action_ls_ratio in the CSV: action "
                 "lengthscale / geomean of that GP's other lengthscales)")
    fig.colorbar(im, ax=ax1)

    labels_d = [f"{r['phase']}:{r['gp']}" for r in deaf_rows]
    x = np.arange(len(deaf_rows))
    width = 0.35
    spreads = [r["spread"] for r in deaf_rows]
    sigmas = [r["sigma_n"] for r in deaf_rows]
    ax2.bar(x - width / 2, spreads, width, color="C0", label="action-sweep spread")
    ax2.bar(x + width / 2, sigmas, width, color="0.6", label="sigma_n")
    for xi, r in zip(x, deaf_rows):
        ax2.text(xi, max(r["spread"], r["sigma_n"]) * 1.02, f"{r['sd_over_noise_ratio']:.2f}x",
                 ha="center", fontsize=7)
    ax2.axvline(STATE_DIM - 0.5, color="k", ls=":", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_d, rotation=45, ha="right", fontsize=8)
    ax2.set_title("action-sweep spread vs noise floor\n(label = implied-SD / sigma_n ratio)")
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
            m = np.array([r["batch_phase"] == ph for r in crows])
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


def plot_model_phase_summary(model_phase_channel_stats, out_path, title_suffix=""):
    """The question action_sensitivity.py structurally cannot ask: does true-vs-model sign
    agreement differ between phase1 and phase2 -- i.e. does accuracy change at the pivot?"""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CHANNELS))
    width = 0.35
    for i, ph in enumerate(PHASE_LABELS):
        vals = []
        for c in CHANNELS:
            a, t = model_phase_channel_stats.get((ph, c), (0, 0))
            vals.append(a / t if t else float("nan"))
        ax.bar(x + (i - 0.5) * width, vals, width, label=ph)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean sign-agreement fraction (true vs model)")
    ax.set_title(f"Sign agreement by dual-phase model routing{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
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
    PIVOT_MARGIN = 1                 # extra J_GRID points on each side of the pivot -- each
                                     # extra J value costs 2 real PenSimPy ODE rollouts x
                                     # len(DELTAS), so keep this small
    DELTAS = [-1.0, -0.2, 0.2, 1.0]
    HORIZONS = [1, 20]
    OFFMANIFOLD_LEVEL = 0.6

    agent, run, _idx = load_agent(run_id, trial, setup=setup, results_root=results_root)
    pivot_step = pivot_step_of(run)
    out_dir = run.dir
    print(f"pivot_step={pivot_step} (pivot_hours={run.pivot_hours:g})")

    # J_GRID_SHARED (module constant) must stay textually identical to action_sensitivity.py's --
    # pivot-adjacent points are probed too (full data collected) but tagged in_shared_grid=False
    # and excluded from the batch_phase summary that gets compared across scripts.
    J_GRID_PIVOT = sorted(
        set(range(max(0, pivot_step - PIVOT_MARGIN), pivot_step + PIVOT_MARGIN + 1))
        - set(J_GRID_SHARED))
    J_GRID = sorted(set(J_GRID_SHARED) | set(J_GRID_PIVOT))

    ml = agent.model_learning
    for phase, sub in [("phase1", ml.phase1), ("phase2", ml.phase2)]:
        X = sub.gp_inputs
        print(f"\n--- {phase} GP training input stats ---")
        for name, m, s in zip(INPUT_NAMES, X.mean(dim=0).tolist(), X.std(dim=0).tolist()):
            print(f"{name:>10}: mean={m:8.4f}  std={s:8.4f}")

    print("\n--- lengthscale report ---")
    ls_rows = lengthscale_report(agent)
    save_csv(ls_rows, out_dir / "action_sensitivity_lengthscales.csv",
             ["phase", "gp"] + [f"ls_{n}" for n in INPUT_NAMES] +
             ["action_lengthscale", "action_ls_ratio"])

    print("\n--- action-deafness report ---")
    deaf_rows = action_deafness_report(agent)
    save_csv(deaf_rows, out_dir / "action_sensitivity_deafness.csv",
             ["phase", "gp", "spread", "sigma_n", "implied_sd", "sd_over_noise_ratio"])

    channel_scale = channel_scales(agent)
    print("\n--- per-(phase, channel) sign threshold scale (std of GP training targets) ---")
    for (phase, c), scale in sorted(channel_scale.items()):
        print(f"{phase}:{c:>10}: scale={scale:.5f}  eps={SIGN_REL_EPS * scale:.6f}")

    print(f"\n--- true vs model action sensitivity + sustained-feed divergence "
         f"(on-manifold, a=0 baseline, {N_SIM_SEEDS} sim seeds) ---")
    (rows, rows_by_seed, ms_rows, loss_rows, phase_channel_stats, model_phase_channel_stats,
     agree, total) = action_effect_table(
        agent, SIM_SEEDS, J_GRID, DELTAS, HORIZONS, pivot_step, base_level=0.0,
        channel_scale=channel_scale, j_grid_shared=set(J_GRID_SHARED))
    save_csv(rows, out_dir / "action_sensitivity_table.csv",
             ["batch_phase", "model_phase", "j", "delta", "channel", "true_delta_mean",
              "true_delta_std", "model_delta_mean", "model_delta_std", "sign_match_frac",
              "n_seeds", "base_level", "in_shared_grid"])
    save_csv(rows_by_seed, out_dir / "action_sensitivity_table_by_seed.csv",
             ["seed", "batch_phase", "model_phase", "j", "delta", "channel", "true_delta",
              "model_delta", "base_level", "in_shared_grid"])

    summary_rows = []
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        summary_rows.append({"grouping": "batch_phase", "phase": phase, "channel": c,
                             "mean_sign_agreement_frac": a / t, "n_combos": t})
    for (phase, c), (a, t) in sorted(model_phase_channel_stats.items()):
        summary_rows.append({"grouping": "model_phase", "phase": phase, "channel": c,
                             "mean_sign_agreement_frac": a / t, "n_combos": t})
    summary_rows.append({"grouping": "ALL", "phase": "ALL", "channel": "ALL",
                         "mean_sign_agreement_frac": agree / total, "n_combos": total})
    save_csv(summary_rows, out_dir / "action_sensitivity_summary.csv",
             ["grouping", "phase", "channel", "mean_sign_agreement_frac", "n_combos"])

    plot_gp_diagnostics(ls_rows, deaf_rows, out_dir / "action_sensitivity_gp_diagnostics.png")
    plot_sensitivity(rows, out_dir / "action_sensitivity_scatter.png")
    plot_model_phase_summary(model_phase_channel_stats,
                             out_dir / "action_sensitivity_model_phase_summary.png")

    save_csv(ms_rows, out_dir / "action_sensitivity_multistep.csv",
             ["model_phase", "j", "delta", "k", "channel", "true_delta_mean", "model_delta_mean",
              "abs_error", "sign_match_frac", "n_seeds", "in_shared_grid"])
    save_csv(loss_rows, out_dir / "action_sensitivity_multistep_first_loss.csv",
             ["model_phase", "j", "delta", "channel", "first_sign_loss_k"])
    plot_first_sign_loss(loss_rows, HORIZONS, out_dir / "action_sensitivity_multistep_first_loss.png")

    print(f"\n--- true vs model action sensitivity + sustained-feed divergence "
         f"(off-manifold, a={OFFMANIFOLD_LEVEL} baseline) ---")
    (om_rows, om_rows_by_seed, om_ms_rows, om_loss_rows, om_phase_channel_stats,
     om_model_phase_channel_stats, om_agree, om_total) = action_effect_table(
        agent, SIM_SEEDS, J_GRID, DELTAS, HORIZONS, pivot_step, base_level=OFFMANIFOLD_LEVEL,
        channel_scale=channel_scale, j_grid_shared=set(J_GRID_SHARED))
    save_csv(om_rows, out_dir / "action_sensitivity_offmanifold_table.csv",
             ["batch_phase", "model_phase", "j", "delta", "channel", "true_delta_mean",
              "true_delta_std", "model_delta_mean", "model_delta_std", "sign_match_frac",
              "n_seeds", "base_level", "in_shared_grid"])
    save_csv(om_rows_by_seed, out_dir / "action_sensitivity_offmanifold_table_by_seed.csv",
             ["seed", "batch_phase", "model_phase", "j", "delta", "channel", "true_delta",
              "model_delta", "base_level", "in_shared_grid"])
    save_csv(om_ms_rows, out_dir / "action_sensitivity_offmanifold_multistep.csv",
             ["model_phase", "j", "delta", "k", "channel", "true_delta_mean", "model_delta_mean",
              "abs_error", "sign_match_frac", "n_seeds", "in_shared_grid"])
    save_csv(om_loss_rows, out_dir / "action_sensitivity_offmanifold_multistep_first_loss.csv",
             ["model_phase", "j", "delta", "channel", "first_sign_loss_k"])

    om_summary_rows = []
    for (phase, c), (a, t) in sorted(om_phase_channel_stats.items()):
        om_summary_rows.append({"grouping": "batch_phase", "phase": phase, "channel": c,
                                "mean_sign_agreement_frac": a / t, "n_combos": t})
    for (phase, c), (a, t) in sorted(om_model_phase_channel_stats.items()):
        om_summary_rows.append({"grouping": "model_phase", "phase": phase, "channel": c,
                                "mean_sign_agreement_frac": a / t, "n_combos": t})
    om_summary_rows.append({"grouping": "ALL", "phase": "ALL", "channel": "ALL",
                            "mean_sign_agreement_frac": om_agree / om_total, "n_combos": om_total})
    save_csv(om_summary_rows, out_dir / "action_sensitivity_offmanifold_summary.csv",
             ["grouping", "phase", "channel", "mean_sign_agreement_frac", "n_combos"])
    plot_sensitivity(om_rows, out_dir / "action_sensitivity_offmanifold_scatter.png",
                     title_suffix=f" (off-manifold, a={OFFMANIFOLD_LEVEL} baseline)")
    plot_model_phase_summary(om_model_phase_channel_stats,
                             out_dir / "action_sensitivity_offmanifold_model_phase_summary.png",
                             title_suffix=f" (off-manifold, a={OFFMANIFOLD_LEVEL} baseline)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", type=str,
                   help="run to evaluate, e.g. 'seed1_1' (resolved under results/<setup>/, see "
                        "--setup) or a full/relative path to a run folder")
    p.add_argument("--trial", type=int, default=None,
                   help="which trial's GP model to diagnose (default: last saved)")
    p.add_argument("--setup", choices=list(SETUPS), default=DEFAULT_SETUP,
                   help="which dual-phase-baseline variant this run was trained with -- "
                        "'dual_phase_baseline_time' (time re-added as a GP input, this "
                        "script's own default) or 'dual_phase_baseline' (time dropped). "
                        "Picking the wrong one crashes on a GP active_dims/lengthscales shape "
                        "mismatch.")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root run_id is resolved under (default: "
                        "results/<setup>/)")
    args = p.parse_args()
    main(run_id=args.run_id, trial=args.trial, setup=args.setup, results_root=args.results_root)

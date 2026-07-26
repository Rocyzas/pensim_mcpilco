import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from diagnose_gp import reconstruct, _resolve_trial
from mcpilco.pensim_wrapper import (PenSimWrapper, STATE_NAMES, STATE_DIM, ACTION_DIM,
                                    T_SAMPLING, CONTROL_H)

CHANNELS = [c for c in STATE_NAMES if c != "time"]
CHANNEL_IDX = [STATE_NAMES.index(c) for c in CHANNELS]
INPUT_NAMES = STATE_NAMES + ["action"]
SIGN_EPS = 1e-4
PHASE_COLORS = {"early": "C0", "mid": "C1", "late": "C2"}
LARGE_ACTION_LENGTHSCALE = 2.0


def load_agent(results_dir, seed, num_trials, fast, trial):
    base = Path(results_dir)
    if not base.is_absolute():
        base = Path(_ROOT) / base
    log_file = base / "log.pkl"
    log = pickle.load(open(log_file, "rb"))
    idx = _resolve_trial(log, trial)
    agent = reconstruct(seed, num_trials, fast, log, idx)
    print(f"[load_agent] {results_dir} trial {idx}")
    return agent


def lengthscale_report(agent):
    """Per-GP lengthscales, named by looking up each GP's OWN `active_dims` rather than
    assuming every GP shares the same (STATE_DIM + ACTION_DIM)-length input: the Viscosity GP
    drops `time` from its active_dims (see model_learning_det_time.VISC_ACTIVE_DIMS), so its
    lengthscale array is one entry shorter and "action" sits at a different LOCAL position than
    for every other GP."""
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
        large = a_val >= LARGE_ACTION_LENGTHSCALE
        flag = " <-- ACTION LENGTHSCALE LARGE (kernel may be ignoring the action)" if large else ""
        vals = " ".join(f"{n}={v:.3f}" for n, v in zip(names, ls))
        dropped = [n for n in INPUT_NAMES if n not in ls_by_name]
        drop_note = f"  (no {', '.join(dropped)} input)" if dropped else ""
        print(f"[lengthscales] {STATE_NAMES[k]:>10}: {vals}{flag}{drop_note}")
        row = {"gp": STATE_NAMES[k]}
        row.update({f"ls_{n}": ls_by_name.get(n, float("nan")) for n in INPUT_NAMES})
        row["action_lengthscale"] = a_val
        row["action_ls_large"] = large
        rows.append(row)
    return rows


def action_deafness_report(agent, n_sweep=21):
    ml = agent.model_learning
    ref_state = ml.gp_inputs[:, :STATE_DIM].mean(dim=0, keepdim=True)
    a_grid = torch.linspace(-1.0, 1.0, n_sweep, dtype=ml.dtype, device=ml.device)
    rows = []
    with torch.no_grad():
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
            deaf = spread < sigma_n
            flag = " <-- DEAF (spread < sigma_n)" if deaf else ""
            print(f"[action-sweep] {STATE_NAMES[k]:>10}: spread={spread:.5f} sigma_n={sigma_n:.5f}{flag}")
            rows.append({"gp": STATE_NAMES[k], "spread": spread, "sigma_n": sigma_n, "deaf": deaf})
    return rows


def signal_noise_report(agent):
    ml = agent.model_learning
    rows = []
    for k in range(ml.num_gp):
        gp = ml.gp_list[k]
        signal_sd = float(torch.sqrt(torch.exp(gp.log_lambda_par)).detach().cpu())
        noise_sd = float(torch.sqrt(gp.get_sigma_n_2()).detach().cpu())
        ratio = signal_sd / noise_sd if noise_sd else float("inf")
        print(f"[signal-noise] {STATE_NAMES[k]:>10}: signal_sd={signal_sd:.5f} "
              f"noise_sd={noise_sd:.5f} ratio={ratio:.2f}")
        rows.append({"gp": STATE_NAMES[k], "signal_sd": signal_sd, "noise_sd": noise_sd,
                     "ratio": ratio})
    return rows


def true_action_effect(seed, j, delta, base_level=0.0):
    base = PenSimWrapper(seed_offset=0)
    pert = PenSimWrapper(seed_offset=0)
    base_policy = lambda state, decision_idx: np.array([base_level])
    pert_policy = lambda state, decision_idx: np.array(
        [base_level + delta if decision_idx >= j else base_level])
    s_base, _, _ = base.rollout(s0=None, policy=base_policy, T=CONTROL_H, dt=T_SAMPLING,
                                noise=None, seed=seed)
    s_pert, _, _ = pert.rollout(s0=None, policy=pert_policy, T=CONTROL_H, dt=T_SAMPLING,
                                noise=None, seed=seed)
    true_delta = s_pert[j + 1] - s_base[j + 1]
    return s_base[j], true_delta, s_base, s_pert


def model_action_effect(agent, state_j, delta, base_level=0.0):
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    u0 = torch.full((1, ACTION_DIM), float(base_level), dtype=ml.dtype, device=ml.device)
    ud = torch.full((1, ACTION_DIM), float(base_level + delta), dtype=ml.dtype, device=ml.device)
    with torch.no_grad():
        ns0, _, _ = ml.get_next_state(current_state=s, current_input=u0, particle_pred=False)
        nsd, _, _ = ml.get_next_state(current_state=s, current_input=ud, particle_pred=False)
    return (nsd - ns0)[0].detach().cpu().numpy()


def model_rollout(agent, state_j, action_level, n_steps):
    ml = agent.model_learning
    s = torch.tensor(state_j, dtype=ml.dtype, device=ml.device).unsqueeze(0)
    u = torch.full((1, ACTION_DIM), float(action_level), dtype=ml.dtype, device=ml.device)
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


def signed(x):
    return 0.0 if abs(x) < SIGN_EPS else float(np.sign(x))


def sensitivity_table(agent, sim_seed, js, deltas, base_level=0.0):
    n_decisions = int(CONTROL_H / T_SAMPLING)
    header = f"{'phase':>6} {'j':>4} {'delta':>6} {'chan':>10} {'true_d':>10} {'model_d':>10} {'sign':>6}"
    print(f"base_level={base_level}")
    print(header)

    rows = []
    for j in js:
        for delta in deltas:
            state_j, true_delta, _, _ = true_action_effect(sim_seed, j, delta, base_level)
            model_delta = model_action_effect(agent, state_j, delta, base_level)
            phase = phase_of(j, n_decisions)
            for c, idx in zip(CHANNELS, CHANNEL_IDX):
                td, md = float(true_delta[idx]), float(model_delta[idx])
                match = signed(td) == signed(md)
                rows.append({"phase": phase, "j": j, "delta": delta, "channel": c,
                             "true_delta": td, "model_delta": md, "sign_match": match,
                             "base_level": base_level})
                print(f"{phase:>6} {j:>4d} {delta:>6.2f} {c:>10} {td:>10.5f} {md:>10.5f} "
                      f"{'OK' if match else 'X':>6}")

    agree = sum(r["sign_match"] for r in rows)
    total = len(rows)
    phase_channel_stats = {}
    for r in rows:
        key = (r["phase"], r["channel"])
        a, t = phase_channel_stats.get(key, (0, 0))
        phase_channel_stats[key] = (a + int(r["sign_match"]), t + 1)

    print()
    print(f"{'phase':>6} {'chan':>10} {'sign agreement':>15}")
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        print(f"{phase:>6} {c:>10} {a}/{t} = {a / t:.2f}")

    print()
    print(f"HEADLINE sign agreement: {agree}/{total} = {agree / total:.3f}")
    return rows, phase_channel_stats, agree, total


def sustained_divergence_table(agent, sim_seed, js, deltas, horizons, base_level=0.0):
    horizons = sorted(horizons)
    max_h = max(horizons)
    header = f"{'j':>4} {'delta':>6} {'k':>4} {'chan':>10} {'true_d':>10} {'model_d':>10} {'abs_err':>9} {'sign':>6}"
    print(f"base_level={base_level}")
    print(header)

    rows = []
    first_loss = {}
    for j in js:
        for delta in deltas:
            state_j, _, s_base, s_pert = true_action_effect(sim_seed, j, delta, base_level)
            model_base_traj = model_rollout(agent, state_j, base_level, max_h)
            model_pert_traj = model_rollout(agent, state_j, base_level + delta, max_h)
            for k in horizons:
                if j + k >= s_base.shape[0]:
                    continue
                for c, idx in zip(CHANNELS, CHANNEL_IDX):
                    td = float(s_pert[j + k, idx] - s_base[j + k, idx])
                    md = float(model_pert_traj[k, idx] - model_base_traj[k, idx])
                    err = abs(md - td)
                    match = signed(td) == signed(md)
                    rows.append({"j": j, "delta": delta, "k": k, "channel": c,
                                 "true_delta": td, "model_delta": md, "abs_error": err,
                                 "sign_match": match})
                    print(f"{j:>4d} {delta:>6.2f} {k:>4d} {c:>10} {td:>10.5f} {md:>10.5f} "
                          f"{err:>9.5f} {'OK' if match else 'X':>6}")
                    key = (j, delta, c)
                    if not match and key not in first_loss:
                        first_loss[key] = k

    loss_rows = []
    for j in js:
        for delta in deltas:
            for c in CHANNELS:
                k_loss = first_loss.get((j, delta, c), -1)
                loss_rows.append({"j": j, "delta": delta, "channel": c,
                                  "first_sign_loss_k": k_loss})

    print()
    never = sum(1 for r in loss_rows if r["first_sign_loss_k"] == -1)
    print(f"sign held through k={max_h} in {never}/{len(loss_rows)} (j, delta, channel) combos")
    return rows, loss_rows


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {path}")


def plot_gp_diagnostics(ls_rows, deaf_rows, sn_rows, out_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

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
    ax1.set_title(f"decoded lengthscales per GP\n(action col flagged large above {LARGE_ACTION_LENGTHSCALE})")
    fig.colorbar(im, ax=ax1)

    x = np.arange(len(deaf_rows))
    width = 0.35
    spreads = [r["spread"] for r in deaf_rows]
    sigmas = [r["sigma_n"] for r in deaf_rows]
    colors = ["crimson" if r["deaf"] else "C0" for r in deaf_rows]
    ax2.bar(x - width / 2, spreads, width, color=colors, label="action-sweep spread")
    ax2.bar(x + width / 2, sigmas, width, color="0.6", label="sigma_n")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["gp"] for r in deaf_rows])
    ax2.set_title("action-sweep spread vs noise floor (red = deaf)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    x3 = np.arange(len(sn_rows))
    signal_sds = [r["signal_sd"] for r in sn_rows]
    noise_sds = [r["noise_sd"] for r in sn_rows]
    ax3.bar(x3 - width / 2, signal_sds, width, color="C0", label="signal sd (sqrt lambda)")
    ax3.bar(x3 + width / 2, noise_sds, width, color="C3", label="noise sd (sigma_n)")
    ax3.set_yscale("log")
    ax3.set_xticks(x3)
    ax3.set_xticklabels([r["gp"] for r in sn_rows])
    ax3.set_title("signal vs noise per channel\nnoise near signal = channel not being explained")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_sensitivity(rows, out_path, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, c in zip(axes.ravel(), CHANNELS):
        crows = [r for r in rows if r["channel"] == c]
        td = np.array([r["true_delta"] for r in crows])
        md = np.array([r["model_delta"] for r in crows])
        for ph, col in PHASE_COLORS.items():
            m = np.array([r["phase"] == ph for r in crows])
            ax.scatter(td[m], md[m], color=col, label=ph, alpha=0.8, zorder=3)
        lim = [min(td.min(), md.min()), max(td.max(), md.max())]
        ax.plot(lim, lim, "k--", lw=1, label="y=x", zorder=2)
        ax.axhline(0, color="0.7", lw=0.8, zorder=1)
        ax.axvline(0, color="0.7", lw=0.8, zorder=1)
        agree = sum(r["sign_match"] for r in crows)
        rho, p_value = stats.spearmanr(td, md) if len(td) > 1 else (float("nan"), float("nan"))
        ax.set_title(f"{c}: sign agree {agree}/{len(crows)}, rho={rho:+.2f} (p={p_value:.3f})")
        ax.set_xlabel("true delta")
        ax.set_ylabel("model delta")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"true vs model one-step action sensitivity{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_multistep_divergence(rows, horizons, out_path, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    deltas_all = sorted({r["delta"] for r in rows})
    norm = plt.Normalize(vmin=min(deltas_all), vmax=max(deltas_all))
    cmap = plt.cm.coolwarm
    for ax, c in zip(axes.ravel(), CHANNELS):
        for j in sorted({r["j"] for r in rows}):
            for delta in deltas_all:
                crows = sorted(
                    [r for r in rows if r["channel"] == c and r["j"] == j and r["delta"] == delta],
                    key=lambda r: r["k"])
                if not crows:
                    continue
                ks = [r["k"] for r in crows]
                td = [r["true_delta"] for r in crows]
                md = [r["model_delta"] for r in crows]
                color = cmap(norm(delta))
                ax.plot(ks, td, "-o", color=color, alpha=0.5, ms=3, lw=1)
                ax.plot(ks, md, "--x", color=color, alpha=0.5, ms=4, lw=1)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{c} (solid=true, dashed=model)")
        ax.set_xlabel("horizon k (decisions ahead)")
        ax.set_ylabel("cumulative delta (normalised)")
        ax.set_xticks(horizons)
        ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="delta", shrink=0.8)
    fig.suptitle(f"multi-step sustained-feed divergence: true vs model{title_suffix}")
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
    ax.set_title(f"first horizon at which the GP loses the true sign{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    RESULTS_DIR = "results/single_phase/seed3_33"
    SEED = 3
    NUM_TRIALS = 5
    FAST = False
    TRIAL = None
    SIM_SEED = 424242
    J_GRID = [2, 8, 15, 22, 30, 38, 44]
    DELTAS = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]
    HORIZONS = [1, 5, 10, 20]
    OFFMANIFOLD_LEVEL = 0.6

    out_dir = Path(_ROOT) / RESULTS_DIR
    agent = load_agent(RESULTS_DIR, SEED, NUM_TRIALS, FAST, TRIAL)

    ml = agent.model_learning
    X = ml.gp_inputs          # [N, STATE_DIM + ACTION_DIM], shared across all GPs
    for name, m, s in zip(INPUT_NAMES, X.mean(dim=0).tolist(), X.std(dim=0).tolist()):
        print(f"{name:>10}: mean={m:8.4f}  std={s:8.4f}")


    print("\n--- lengthscale report ---")
    ls_rows = lengthscale_report(agent)
    save_csv(ls_rows, out_dir / "action_sensitivity_lengthscales.csv",
             ["gp"] + [f"ls_{n}" for n in INPUT_NAMES] + ["action_lengthscale", "action_ls_large"])

    print("\n--- action-deafness report ---")
    deaf_rows = action_deafness_report(agent)
    save_csv(deaf_rows, out_dir / "action_sensitivity_deafness.csv",
             ["gp", "spread", "sigma_n", "deaf"])

    print("\n--- signal vs noise report ---")
    sn_rows = signal_noise_report(agent)
    save_csv(sn_rows, out_dir / "action_sensitivity_signal_noise.csv",
             ["gp", "signal_sd", "noise_sd", "ratio"])

    print("\n--- true vs model action sensitivity (on-manifold, a=0 baseline) ---")
    rows, phase_channel_stats, agree, total = sensitivity_table(agent, SIM_SEED, J_GRID, DELTAS)
    save_csv(rows, out_dir / "action_sensitivity_table.csv",
             ["phase", "j", "delta", "channel", "true_delta", "model_delta", "sign_match", "base_level"])

    summary_rows = []
    for (phase, c), (a, t) in sorted(phase_channel_stats.items()):
        summary_rows.append({"phase": phase, "channel": c, "agree": a, "total": t,
                             "fraction": a / t})
    summary_rows.append({"phase": "ALL", "channel": "ALL", "agree": agree, "total": total,
                         "fraction": agree / total})
    save_csv(summary_rows, out_dir / "action_sensitivity_summary.csv",
             ["phase", "channel", "agree", "total", "fraction"])

    plot_gp_diagnostics(ls_rows, deaf_rows, sn_rows, out_dir / "action_sensitivity_gp_diagnostics.png")
    plot_sensitivity(rows, out_dir / "action_sensitivity_scatter.png")

    print("\n--- multi-step sustained-feed divergence (on-manifold, a=0 baseline) ---")
    ms_rows, ms_loss_rows = sustained_divergence_table(agent, SIM_SEED, J_GRID, DELTAS, HORIZONS)
    save_csv(ms_rows, out_dir / "action_sensitivity_multistep.csv",
             ["j", "delta", "k", "channel", "true_delta", "model_delta", "abs_error", "sign_match"])
    save_csv(ms_loss_rows, out_dir / "action_sensitivity_multistep_first_loss.csv",
             ["j", "delta", "channel", "first_sign_loss_k"])
    plot_multistep_divergence(ms_rows, HORIZONS,
                              out_dir / "action_sensitivity_multistep.png")
    plot_first_sign_loss(ms_loss_rows, HORIZONS,
                         out_dir / "action_sensitivity_multistep_first_loss.png")

    print(f"\n--- true vs model action sensitivity (off-manifold, a={OFFMANIFOLD_LEVEL} baseline) ---")
    om_rows, om_phase_channel_stats, om_agree, om_total = sensitivity_table(
        agent, SIM_SEED, J_GRID, DELTAS, base_level=OFFMANIFOLD_LEVEL)
    save_csv(om_rows, out_dir / "action_sensitivity_offmanifold_table.csv",
             ["phase", "j", "delta", "channel", "true_delta", "model_delta", "sign_match", "base_level"])

    om_summary_rows = []
    for (phase, c), (a, t) in sorted(om_phase_channel_stats.items()):
        om_summary_rows.append({"phase": phase, "channel": c, "agree": a, "total": t,
                                "fraction": a / t})
    om_summary_rows.append({"phase": "ALL", "channel": "ALL", "agree": om_agree, "total": om_total,
                            "fraction": om_agree / om_total})
    save_csv(om_summary_rows, out_dir / "action_sensitivity_offmanifold_summary.csv",
             ["phase", "channel", "agree", "total", "fraction"])
    plot_sensitivity(om_rows, out_dir / "action_sensitivity_offmanifold_scatter.png",
                     title_suffix=f" (off-manifold, a={OFFMANIFOLD_LEVEL} baseline)")

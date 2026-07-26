"""Imagined cost vs real yield -- the one diagnostic that ends the "is my model good or bad?" loop.

For each trial's saved policy we measure two things and plot them against each other:

  * imagined cost -- the policy rolled through THAT trial's GP model (exactly the objective the
    policy optimiser minimised: `apply_policy` -> `cost_function`, particles launched from a fresh
    reactor, 45-step imagined rollout, no dropout so it reflects the deployed policy). Lower = better.

  * real yield  -- the SAME policy run on the PenSimPy simulator on a small block of FIXED held-out
    seeds (mean over seeds). Higher = better.

The question this replaces "is the model accurate?" (unanswerable) with is:

    does LOW imagined cost predict HIGH real yield?

A trustworthy model gives a downward-sloping cloud (cost down => yield up, negative correlation).
Where imagined cost keeps falling while real yield craters, the optimiser is exploiting the model --
that divergence is what this plot makes visible.

Runs both single-phase runs (seed2_10, seed3_10). Writes a figure + CSV under
`results/single_phase/imagined_vs_real/` and prints the per-trial table.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE \
    /Users/rokaspranevicius/miniforge3/envs/mcpilco/bin/python experiments/imagined_vs_real.py
(from the pensim_mcpilco dir). Options: --real-seeds N, --imag-repeats N, --num-particles N, --fast.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conda libomp double-link guard

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

# --- repo paths (mirror the evaluations notebooks) -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)              # pensim_mcpilco/
_OUTER = os.path.dirname(_ROOT)             # repo root (for PenSimPy)
for p in (_ROOT, _OUTER):
    if p not in sys.path:
        sys.path.insert(0, p)

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import (
    PenSimWrapper, PenSimMCPILCO, STATE_NAMES, STATE_RANGES, STATE_DIM,
    T_SAMPLING, CONTROL_H, TIME_IDX, TIME_INIT_VAR, initial_state_norm,
    decode_state_value,
)
from experiments.eval_utils import yield_kg

P_IDX = STATE_NAMES.index("P")
P_PHYS_MAX = STATE_RANGES["P"][1]     # declared physical ceiling for penicillin conc (g/L)

# --- runs to evaluate -----------------------------------------------------------------------------
# (label, seed, run_suffix). Policy stage k's GP index is read from the log itself (see
# evaluate_run), not assumed from an exploration-count offset -- a run with a different exploration
# count or num_trials would otherwise silently mis-pair stage and GP (this is what broke here:
# seed3_3 predates the VISC_ACTIVE_DIMS fix, so its saved GPs don't even match today's architecture).
RUNS = [
    # ("seed 2", 2, 11),
    ("seed 3", 3, 31),
]

OUT_DIR = Path(_ROOT) / "results" / "single_phase" / f"seed{RUNS[0][1]}_{RUNS[0][2]}" / "imagined_vs_real"


def _quiet(fn, *a, **k):
    """Call fn swallowing its (very chatty) stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def reconstruct(seed, num_trials, fast, log, idx):
    """Build a PenSimMCPILCO and load the trial-`idx` GP model from `log` (no training).

    Identical to the `reconstruct` helper in the evaluations notebooks."""
    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast)
    cfg["mc_pilco_init"]["log_path"] = None
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
    ml.dim_state = len(STATE_NAMES)
    ml.init_gp_models()
    params = log[f"parameters_gp_{idx}"]
    for k in range(ml.num_gp):
        ml.gp_list[k].load_state_dict(params[k])
        ml.norm_list[k] = torch.max(torch.abs(ml.gp_output_list[k]))
    with torch.no_grad():
        for k in range(ml.num_gp):
            ml.pretrain_gp(k)
    ml.set_eval_mode()
    return agent


def _init_state_dist(agent):
    """The fixed particle-launch distribution used during policy optimisation (config_single_phase):
    a single Gaussian around a fresh reactor, tiny variance, with the deterministic-time channel."""
    mean = torch.tensor(initial_state_norm(), dtype=agent.dtype, device=agent.device)
    iv = 0.01 * np.ones(STATE_DIM)
    iv[TIME_IDX] = TIME_INIT_VAR
    var = torch.tensor(iv, dtype=agent.dtype, device=agent.device)
    return mean, var


def imagined_metrics(agent, policy_state_dict, control_horizon, num_particles,
                     n_repeats=5, base_seed=0):
    """Roll `policy_state_dict` through the agent's currently-loaded GP and summarise the imagined
    outcome over the particle cloud, pooled across `n_repeats` fixed-seed draws.

    Two costs are returned because they say different things:
      * `cost_mean`   = sum-over-time of MEAN-over-particles cost -- this is EXACTLY the objective
        the policy optimiser minimised. On these runs it is dominated by a runaway tail (the
        log-encoded P channel exponentiates when the GP predicts off-distribution), so it blows up
        to +/-1e8..1e21 and is not directly plottable.
      * `cost_median` = median-over-particles of the per-particle total cost -- the robust statistic
        (runaway particles do not drag a median; this is the same logic as the notebook's C.2b
        particle-median band). This is what we plot.

    Also reports where the imagined batch actually ends up:
      * `final_P_median` (g/L)  -- median imagined final penicillin (reveals reversion toward 0),
      * `hallucination_frac`    -- fraction of particles whose imagined final P sits AT OR ABOVE the
        declared physical ceiling P_PHYS_MAX. Without the rollout clamp this is the impossible-
        explosion tail (final P -> 1e6..1e21); with the clamp those same particles are pinned exactly
        at the ceiling (40 g/L). Either way it counts the particles the policy is driving into the
        penicillin ceiling. Threshold is `>= (1 - 1e-6) * ceiling` so it catches the pinned case
        robustly instead of depending on float rounding right at 40.0.
    """
    agent.control_policy.load_state_dict(policy_state_dict)
    agent.model_learning.set_eval_mode()
    mean, var = _init_state_dist(agent)
    lo, hi = STATE_RANGES["P"]
    per_particle_cost, final_P = [], []
    with torch.no_grad():
        for r in range(n_repeats):
            torch.manual_seed(base_seed + r)
            states_seq, inputs_seq = agent.apply_policy(
                particles_initial_state_mean=mean,
                particles_initial_state_var=var,
                flg_particles_init_uniform=False,
                particles_init_up_bound=None,
                particles_init_low_bound=None,
                flg_particles_init_multi_gauss=False,
                num_particles=num_particles,
                T_control=control_horizon,
                p_dropout=0.0,
            )
            # raw per-[time, particle] cost tensor (before Expected_cost's mean/sum reduction).
            # sum(mean over particles) == mean(sum over particles), so cost_mean below is the
            # optimiser's true objective, and cost_median is its robust counterpart.
            costs = agent.cost_function.cost_function(states_seq, inputs_seq, None)
            per_particle_cost.append(costs.sum(0))                      # [particles]
            Pn = states_seq[-1, :, P_IDX]
            final_P.append(decode_state_value("P", lo + (Pn + 1.0) * (hi - lo) / 2.0))
    per_particle_cost = torch.cat(per_particle_cost)
    final_P = torch.cat(final_P)
    return dict(
        cost_mean=float(per_particle_cost.mean()),
        cost_median=float(per_particle_cost.median()),
        cost_q25=float(torch.quantile(per_particle_cost.double(), 0.25)),
        cost_q75=float(torch.quantile(per_particle_cost.double(), 0.75)),
        final_P_median=float(final_P.median()),
        hallucination_frac=float((final_P >= P_PHYS_MAX * (1.0 - 1e-6)).double().mean()),
    )


def run_arm(wrapper, seed, np_policy):
    """One RL batch on `seed`; returns the monitor dict."""
    wrapper.rollout(None, np_policy, CONTROL_H, T_SAMPLING, 0, seed=seed, pid_baseline=False)
    return wrapper.monitor[-1]


def run_recipe(wrapper, seed):
    wrapper.rollout(None, None, CONTROL_H, T_SAMPLING, 0, seed=seed, pid_baseline=True)
    return wrapper.monitor[-1]


def real_yield(wrapper, np_policy, seeds):
    ys = np.array([yield_kg(run_arm(wrapper, s, np_policy)) for s in seeds])
    return float(ys.mean()), float(ys.std()), ys


def evaluate_run(label, seed, suffix, real_seeds, imag_repeats, num_particles, fast):
    run_dir = Path(_ROOT) / "results" / "single_phase" / f"seed{seed}_{suffix}"
    log = pd.read_pickle(run_dir / "log.pkl")
    trial_policies = log["parameters_trial_list"]
    logged_cost = log.get("cost_trial_list", None)
    n_stages = len(trial_policies)
    num_trials = n_stages          # the run's actual trained trial count, not an assumed constant
    control_horizon = int(CONTROL_H / T_SAMPLING)   # = 45, the training imagined-rollout length

    # Pair policy stage k with the GP it was actually optimised against, read from the log's own
    # saved trial indices -- not an assumed NUM_EXPLORATIONS/FIRST_GP_IDX offset, which silently
    # mis-pairs (or, as here, crashes on an architecture mismatch) the moment a run's exploration
    # count or code version differs from what the offset was written for.
    gp_trials = sorted(int(k.split("_")[-1]) for k in log if k.startswith("parameters_gp_"))
    assert len(gp_trials) == n_stages, (
        f"{n_stages} policy stages but {len(gp_trials)} saved GP trials -- stage/GP pairing "
        "assumes these line up one-to-one")

    eval_wrapper = PenSimWrapper()
    rows = []
    print(f"\n=== {label}  (results/single_phase/seed{seed}_{suffix})  stages={n_stages}  "
          f"imagined H={control_horizon} steps, {num_particles} particles x{imag_repeats} ===")
    print(f"{'stage':>5} {'gp_idx':>6} | {'cost_median':>11} | {'cost_mean(objective)':>21} | "
          f"{'imagP_med':>9} {'hallu%':>6} | {'real_yield_kg':>13} {'+/-':>7}")
    for k in range(1, n_stages + 1):
        gp_idx = gp_trials[k - 1]
        agent = _quiet(reconstruct, seed, num_trials, fast, log, gp_idx)
        m = imagined_metrics(agent, trial_policies[k - 1], control_horizon,
                             num_particles, n_repeats=imag_repeats)
        np_pol = agent.control_policy.get_np_policy()
        ry_mean, ry_std, ys = real_yield(eval_wrapper, np_pol, real_seeds)
        logged = (float(logged_cost[k - 1][-1])
                  if logged_cost is not None and len(logged_cost) >= k else np.nan)
        rows.append(dict(run=label, seed=seed, stage=k, gp_idx=gp_idx,
                         imagined_cost_median=m["cost_median"],
                         imagined_cost_q25=m["cost_q25"], imagined_cost_q75=m["cost_q75"],
                         imagined_cost_mean_objective=m["cost_mean"],
                         imagined_final_P_median=m["final_P_median"],
                         hallucination_frac=m["hallucination_frac"],
                         real_yield_kg=ry_mean, real_yield_std=ry_std,
                         logged_train_cost=logged,
                         **{f"real_yield_seed_{s}": y for s, y in zip(real_seeds, ys)}))
        print(f"{k:>5} {gp_idx:>6} | {m['cost_median']:>11.2f} | {m['cost_mean']:>21.3e} | "
              f"{m['final_P_median']:>9.2f} {100 * m['hallucination_frac']:>5.0f}% | "
              f"{ry_mean:>13.1f} {ry_std:>7.1f}")

    # recipe / PID baseline on the same fixed seeds (policy-independent; same for both runs)
    base_ys = np.array([yield_kg(run_recipe(eval_wrapper, s)) for s in real_seeds])
    return pd.DataFrame(rows), float(base_ys.mean())


def make_figure(df, baseline_mean, real_seeds, out_path):
    runs = list(dict.fromkeys(df["run"]))
    C_COST, C_YIELD = "#1f77b4", "#d62728"
    seed_color = {r: c for r, c in zip(runs, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # row 1: per-run robust imagined cost (left axis) vs real yield (right axis) across trials
    for j, run in enumerate(runs[:2]):
        d = df[df["run"] == run].sort_values("stage")
        ax = axes[0, j]
        ax.plot(d["stage"], d["imagined_cost_median"], "o-", color=C_COST, lw=2,
                label="imagined cost (particle median)")
        ax.fill_between(d["stage"], d["imagined_cost_q25"], d["imagined_cost_q75"],
                        color=C_COST, alpha=.12, label="imagined cost IQR (25-75%)")
        ax.axhline(0.0, color=C_COST, ls=":", lw=1, alpha=.6)
        ax.set_xlabel("trial (policy stage)")
        ax.set_ylabel("imagined cost, particle median\n(axis reversed: up = lower cost = better)",
                     color=C_COST)
        ax.tick_params(axis="y", labelcolor=C_COST)
        ax.set_xticks(d["stage"])
        ax.grid(alpha=.25)
        ax.invert_yaxis()   # so "up" agrees with real yield's "up = better" on the twin axis

        ax2 = ax.twinx()
        ax2.plot(d["stage"], d["real_yield_kg"], "s--", color=C_YIELD, lw=2,
                 label="real yield (sim)")
        ax2.fill_between(d["stage"], d["real_yield_kg"] - d["real_yield_std"],
                         d["real_yield_kg"] + d["real_yield_std"], color=C_YIELD, alpha=.12)
        ax2.axhline(baseline_mean, color="0.35", ls=":", lw=1.6, label="recipe / PID")
        ax2.set_ylabel("real yield (kg)  (higher = better)", color=C_YIELD)
        ax2.tick_params(axis="y", labelcolor=C_YIELD)

        rho, p_rho = stats.spearmanr(d["imagined_cost_median"], d["real_yield_kg"])
        ax.set_title(f"{run}: imagined cost vs real yield across trials\n"
                     f"Spearman(cost_median, yield) = {rho:+.2f}  (p={p_rho:.2f})", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="best")

    # row 2 left: the direct scatter -- does low imagined cost predict high real yield?
    axc = axes[1, 0]
    for run in runs:
        d = df[df["run"] == run].sort_values("stage")
        c = seed_color[run]
        axc.plot(d["imagined_cost_median"], d["real_yield_kg"], "-", color=c, alpha=.35, lw=1)
        axc.scatter(d["imagined_cost_median"], d["real_yield_kg"], color=c, s=55, zorder=3, label=run)
        for _, r in d.iterrows():
            axc.annotate(int(r["stage"]), (r["imagined_cost_median"], r["real_yield_kg"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=8, color=c)
    axc.axhline(baseline_mean, color="0.35", ls=":", lw=1.6, label="recipe / PID")
    rho_all, p_all = stats.spearmanr(df["imagined_cost_median"], df["real_yield_kg"])
    axc.set_xlabel("imagined cost, particle median  (lower = better) ->")
    axc.set_ylabel("real yield (kg)  (higher = better)")
    axc.invert_xaxis()   # so lower imagined cost ("better" per the model) runs left->right
    axc.set_title("Does low imagined cost predict high real yield?\n"
                  f"pooled Spearman = {rho_all:+.2f}  (p={p_all:.2f})   (labels = trial stage)",
                  fontsize=10)
    axc.grid(alpha=.25)
    axc.legend(fontsize=8, loc="best")

    # row 2 right: WHY the objective is untrustworthy -- the imagined cloud is largely impossible.
    axh = axes[1, 1]
    for run in runs:
        d = df[df["run"] == run].sort_values("stage")
        c = seed_color[run]
        axh.plot(d["stage"], 100 * d["hallucination_frac"], "o-", color=c, lw=2,
                 label=f"{run}: particles at/above the {P_PHYS_MAX:g} g/L P ceiling")
    axh.set_xlabel("trial (policy stage)")
    axh.set_ylabel(f"% imagined particles at/above physical P ceiling ({P_PHYS_MAX:g} g/L)")
    axh.set_ylim(0, 100)
    axh.set_xticks(sorted(df["stage"].unique()))
    axh.grid(alpha=.25)
    axh.set_title("Mechanism: a large fraction of the imagined cloud is driven into the P ceiling\n"
                  "(unclamped -> explodes to 1e21 and the MEAN objective blows up; clamped -> pinned "
                  "at 40 g/L)", fontsize=9.5)
    axh.legend(fontsize=7.5, loc="best")

    fig.suptitle("Imagined (GP-planned) cost vs real (simulated) yield, per trial policy   "
                 f"[real yield = mean over fixed seeds {list(real_seeds)}]\n"
                 "particle-median cost is plotted because the mean (optimiser objective) is "
                 "dominated by runaway particles (see mechanism panel)",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-seeds", type=int, default=3,
                    help="number of fixed held-out seeds (from 700000) to average real yield over")
    ap.add_argument("--imag-repeats", type=int, default=5,
                    help="particle-sampling draws to average the imagined cost over")
    ap.add_argument("--num-particles", type=int, default=100,
                    help="particles per imagined rollout (training used 100)")
    ap.add_argument("--fast", action="store_true", help="pass fast=True to get_config")
    args = ap.parse_args()

    real_seeds = [700000 + i for i in range(args.real_seeds)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames, baselines = [], []
    for label, seed, suffix in RUNS:
        df, base = evaluate_run(label, seed, suffix, real_seeds,
                                args.imag_repeats, args.num_particles, args.fast)
        frames.append(df)
        baselines.append(base)
    all_df = pd.concat(frames, ignore_index=True)
    baseline_mean = float(np.mean(baselines))

    csv_path = OUT_DIR / "imagined_vs_real.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nrecipe/PID baseline on {real_seeds}: {baseline_mean:.1f} kg")
    print(f"saved table   -> {csv_path}")

    make_figure(all_df, baseline_mean, real_seeds, OUT_DIR / "imagined_vs_real.png")


if __name__ == "__main__":
    main()

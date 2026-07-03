
import argparse
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCO, CONTROL_H, T_SAMPLING, PAA_BAND
from experiments.eval_utils import yield_kg, constraint_diagnostics


def run_arm(wrapper, seed, policy=None, pid_baseline=False):
    """Run one batch on `seed` and return its monitor dict."""
    wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0, seed=seed, pid_baseline=pid_baseline)
    return wrapper.monitor[-1]


def main(seed=1, num_trials=10, n_seeds=20, eval_base=700000, log_dir=None, out_dir=None):
    cfg = get_config(seed=seed, num_trials=num_trials)
    folder = str(log_dir or cfg["mc_pilco_init"]["log_path"]).rstrip("/") + "/"
    out = Path(out_dir or _os.path.join(_ROOT, "results/rl_vs_pid"))
    out.mkdir(parents=True, exist_ok=True)

    # held-out guard: eval seeds must be disjoint from the policy's training window
    # [seed*1000, seed*1000 + 4 + num_trials]  (5 explorations + num_trials trials).
    train_lo, train_hi = seed * 1000, seed * 1000 + 4 + num_trials
    eval_seeds = [eval_base + i for i in range(n_seeds)]
    assert all(not (train_lo <= h <= train_hi) for h in eval_seeds), \
        f"eval seeds overlap training window [{train_lo},{train_hi}]"

    wrapper = PenSimWrapper()
    agent = PenSimMCPILCO(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    agent.load_policy_from_log(num_trial=num_trials, folder=folder)
    np_policy = agent.control_policy.get_np_policy()

    rows, mons_rl, mons_pid = [], [], []
    for h in eval_seeds:
        m_rl = run_arm(wrapper, h, policy=np_policy, pid_baseline=False)
        m_pid = run_arm(wrapper, h, policy=None, pid_baseline=True)
        mons_rl.append(m_rl)
        mons_pid.append(m_pid)
        y_rl, y_pid = yield_kg(m_rl), yield_kg(m_pid)
        row = {"seed": h, "yield_rl": y_rl, "yield_pid": y_pid, "delta": y_rl - y_pid}
        row.update({f"rl_{k}": v for k, v in constraint_diagnostics(m_rl).items()})
        row.update({f"pid_{k}": v for k, v in constraint_diagnostics(m_pid).items()})
        rows.append(row)
        print(f"seed {h}: yield_rl={y_rl:8.2f}  yield_pid={y_pid:8.2f}  delta={y_rl - y_pid:+8.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "paired_metrics.csv", index=False)

    delta = df["delta"].values
    n = len(delta)
    mean_d = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    ci = 1.96 * se
    t_p = float(stats.ttest_rel(df["yield_rl"], df["yield_pid"]).pvalue) if n > 1 else float("nan")
    try:
        w_p = float(stats.wilcoxon(df["yield_rl"], df["yield_pid"]).pvalue)
    except ValueError:
        w_p = float("nan")  # e.g. all-zero differences

    summary = pd.DataFrame([{
        "n_seeds": n,
        "mean_yield_rl": float(df["yield_rl"].mean()),
        "mean_yield_pid": float(df["yield_pid"].mean()),
        "mean_delta": mean_d,
        "delta_ci95_lo": mean_d - ci,
        "delta_ci95_hi": mean_d + ci,
        "ttest_p": t_p,
        "wilcoxon_p": w_p,
        "winrate_rl_gt_pid": float((delta > 0).mean()),
    }])
    summary.to_csv(out / "summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.T.to_string(header=False))

    _plots(df, mons_rl, mons_pid, out)
    print(f"\nSaved results to {out}")


def _plots(df, mons_rl, mons_pid, out):
    t = np.asarray(mons_rl[0]["t"])
    avg = lambda mons, key: np.mean([m[key] for m in mons], axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    lo = float(min(df["yield_pid"].min(), df["yield_rl"].min()))
    hi = float(max(df["yield_pid"].max(), df["yield_rl"].max()))
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].scatter(df["yield_pid"], df["yield_rl"], c="C0")
    ax[0].set_xlabel("PID yield (kg)")
    ax[0].set_ylabel("RL yield (kg)")
    ax[0].set_title("Paired yield (above y=x => RL wins)")

    ax[1].bar(np.arange(len(df)), df["delta"].values,
              color=["C2" if d > 0 else "C3" for d in df["delta"]])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("held-out seed idx")
    ax[1].set_ylabel("delta yield RL-PID (kg)")
    ax[1].set_title("Per-seed difference")

    ax[2].plot(t, avg(mons_rl, "P"), label="RL", color="C0")
    ax[2].plot(t, avg(mons_pid, "P"), label="PID", color="C1")
    ax[2].set_xlabel("time (h)")
    ax[2].set_ylabel("P (g/L)")
    ax[2].set_title("Seed-averaged penicillin")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "rl_vs_pid_overview.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    for key, axx in [("Fpaa", ax2[0]), ("PAA", ax2[1])]:
        axx.plot(t, avg(mons_rl, key), label="RL", color="C0")
        axx.plot(t, avg(mons_pid, key), label="PID", color="C1")
        axx.set_xlabel("time (h)")
        axx.set_title(f"Seed-averaged {key}")
        axx.legend()
    ax2[1].axhspan(PAA_BAND[0], PAA_BAND[1], color="green", alpha=0.08)  # PAA target band
    fig2.tight_layout()
    fig2.savefig(out / "rl_vs_pid_control.png", dpi=120)
    plt.close(fig2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1, help="which trained policy (seed) to load")
    p.add_argument("--num_trials", type=int, default=10, help="trial count of that run (last trial is loaded)")
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--eval_base", type=int, default=700000, help="held-out seed block start")
    p.add_argument("--log_dir", type=str, default=None, help="folder with log.pkl (default: config log_path)")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()
    main(args.seed, args.num_trials, args.n_seeds, args.eval_base, args.log_dir, args.out_dir)

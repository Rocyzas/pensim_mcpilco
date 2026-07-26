"""Open-loop feed sweep on the REAL simulator -- the "ceiling" test.

Question this answers
---------------------
Is there ANY fixed Fs-residual trajectory in the +/-100% action space that beats the recipe on
the real PenSimPy simulator? For a handful of open-loop feed policies we roll the *real* plant
(not the GP) on a shared set of held-out seeds and measure feasibility-gated yield_kg, paired
against the recipe on the same seeds.

How to read it
--------------
The action is a MULTIPLICATIVE residual on the recipe Fs profile: fs = fs_recipe * (1 + FS_SCALE*a),
with FS_SCALE=1, so a=0 is the recipe, a=+1 is 2x feed, a=-1 is zero substrate feed (starvation).

Open-loop (constant / time-scheduled) policies are a LOWER bound on the achievable ceiling: a
closed-loop state-feedback policy (what MC-PILCO learns) can do strictly better than any open-loop
one in a stochastic plant. So:
  * if the best FEASIBLE open-loop arm already beats the recipe -> the RL is leaving value on the
    table (it should at least match a constant feed);
  * if NO feasible open-loop arm beats the recipe -> the action space / reward is the suspect, not
    the optimiser. (Not conclusive that no closed-loop policy can win -- a proper open-loop
    trajectory optimisation, e.g. CMA-ES over a piecewise-constant Fs schedule, is the stronger
    follow-up. This script is the cheap first cut.)

Note: the RL cost optimises delta-mass-with-penalties, NOT yield_kg. If the yield-optimal feed here
is far from a=0 while the RL policy sits near a=0, that mismatch is itself an explanation for poor
RL yield -- the reward is a weak proxy for yield.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # miniforge OMP dup-libomp workaround

import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pensim_mcpilco
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy optional
    wilcoxon = None

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, WARMUP_H,
                                    FS_SCALE, PAA_BAND, WT_OVERFLOW, VISC_MAX)
from experiments.eval_utils import yield_kg, constraint_diagnostics

# ------------------------------------------------------------------ config ----
EVAL_BASE     = 700_000          # held-out seed block (disjoint from training seeds)
N_EVAL_SEEDS  = 5
CONST_LEVELS  = [-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0]  # a-residual sweep (incl. asks)
# CONST_LEVELS  = [1.0]  # a-residual sweep (incl. asks)
OUT_DIR       = Path(_ROOT) / "results" / "feed_sweep_ceiling"

# shaped, time-scheduled deviations: list of (t_start_h, t_end_h, level); gaps default to a=0
SHAPED = {
    # "feed harder in 40-120h, back off late"
    "early_hard_late_soft": [(40.0, 120.0, 0.6), (120.0, CONTROL_H + WARMUP_H, -0.3)],
    # the reverse: hold back early, push late
    "early_soft_late_hard": [(0.0, 120.0, -0.3), (120.0, CONTROL_H + WARMUP_H, 0.6)],
}


# ---------------------------------------------------------------- policies ----
def const_policy(a):
    a = float(a)
    return lambda state, decision_idx: np.array([a])


def shaped_policy(schedule):
    def pol(state, decision_idx):
        t = WARMUP_H + decision_idx * T_SAMPLING
        for lo, hi, lvl in schedule:
            if lo <= t < hi:
                return np.array([float(lvl)])
        return np.array([0.0])
    return pol


def run_arm(wrapper, seed, policy=None, pid_baseline=False):
    """One real-sim batch on `seed`; returns its monitor dict (same as evaluations.ipynb)."""
    wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0, seed=seed, pid_baseline=pid_baseline)
    return wrapper.monitor[-1]


def _feasible(diag):
    """Hard physical feasibility: no broth overflow and no viscosity blow-up.
    PAA-band fraction and peak viscosity are reported separately as soft context."""
    return (not diag["wt_overflow"]) and (not diag["visc_exceed"])


# --------------------------------------------------------------------- run ----
def build_arms(const_levels, shaped):
    """Ordered list of (name, kind, policy, pid_baseline)."""
    arms = [("recipe", "recipe", None, True)]
    for a in const_levels:
        arms.append((f"const_a={a:+.2f}", "const", const_policy(a), False))
    for name, sched in shaped.items():
        arms.append((name, "shaped", shaped_policy(sched), False))
    return arms


def sweep(n_seeds=N_EVAL_SEEDS, const_levels=CONST_LEVELS, shaped=SHAPED, verbose=True):
    seeds = [EVAL_BASE + i for i in range(n_seeds)]
    arms = build_arms(const_levels, shaped)
    wrapper = PenSimWrapper()

    rows = []
    for name, kind, pol, pid in arms:
        for s in seeds:
            mon = run_arm(wrapper, s, policy=pol, pid_baseline=pid)
            diag = constraint_diagnostics(mon)
            rows.append({
                "arm": name, "kind": kind, "seed": s,
                "yield_kg": yield_kg(mon), "final_P": diag["final_P"],
                "max_Wt": diag["max_Wt"], "max_visc": diag["max_viscosity"],
                "paa_frac_oob": diag["paa_frac_out_of_band"],
                "wt_overflow": diag["wt_overflow"], "visc_exceed": diag["visc_exceed"],
                "feasible": _feasible(diag),
            })
        if verbose:
            sub = pd.DataFrame(rows)
            sub = sub[sub.arm == name]
            print(f"  {name:22s} yield {sub.yield_kg.mean():8.1f} +/- {sub.yield_kg.std():6.1f} kg | "
                  f"feasible {int(sub.feasible.sum())}/{len(sub)}")
    return pd.DataFrame(rows), seeds


def summarise(df, seeds):
    """Per-arm summary with PAIRED delta vs recipe on the shared seeds + Wilcoxon signed-rank."""
    rec = df[df.arm == "recipe"].set_index("seed")["yield_kg"]
    out = []
    for name, g in df.groupby("arm", sort=False):
        g = g.set_index("seed")
        delta = (g["yield_kg"] - rec).reindex(seeds)
        row = {
            "arm": name, "kind": g["kind"].iloc[0],
            "yield_mean": g["yield_kg"].mean(), "yield_std": g["yield_kg"].std(),
            "final_P_mean": g["final_P"].mean(),
            "feasible_frac": g["feasible"].mean(),
            "delta_vs_recipe": delta.mean(),
            "beats_recipe_frac": float((delta > 0).mean()),
            "max_visc_mean": g["max_visc"].mean(), "paa_frac_oob_mean": g["paa_frac_oob"].mean(),
        }
        if wilcoxon is not None and name != "recipe" and np.any(delta.values != 0):
            try:
                row["wilcoxon_p"] = float(wilcoxon(delta.values).pvalue)
            except Exception:
                row["wilcoxon_p"] = np.nan
        else:
            row["wilcoxon_p"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def plot(df, summ, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_mean = summ.loc[summ.arm == "recipe", "yield_mean"].iloc[0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # (1) yield-vs-constant-feed curve, feasibility-shaded markers
    csum = summ[summ.kind == "const"].copy()
    csum["a"] = csum.arm.str.extract(r"a=([+-][\d.]+)").astype(float)
    csum = csum.sort_values("a")
    ax[0].axhline(rec_mean, color="red", ls="--", lw=2, label="recipe (a=0 baseline)")
    for _, r in csum.iterrows():
        feas = r["feasible_frac"] >= 1.0
        ax[0].errorbar(r["a"], r["yield_mean"], yerr=r["yield_std"], fmt="o",
                       color="C0" if feas else "0.6", ms=8,
                       mec="k" if feas else "crimson", mew=1.2, capsize=3)
    ax[0].plot(csum["a"], csum["yield_mean"], "-", color="C0", alpha=.5)
    ax[0].set_xlabel("constant Fs residual  a  (fs = fs_recipe * (1 + a))")
    ax[0].set_ylabel(f"batch yield (kg), mean over {len(seeds)} held-out seeds")
    ax[0].set_title("Constant-feed sweep vs recipe\n(grey/red-edge = infeasible on >=1 seed)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)

    # (2) all arms, paired delta vs recipe (feasible arms only, sorted)
    s = summ[summ.arm != "recipe"].copy().sort_values("delta_vs_recipe")
    colors = ["C0" if f >= 1.0 else "0.6" for f in s["feasible_frac"]]
    ax[1].barh(s["arm"], s["delta_vs_recipe"], color=colors, edgecolor="k")
    ax[1].axvline(0, color="red", ls="--", lw=2)
    ax[1].set_xlabel("paired delta yield vs recipe (kg)  [>0 beats recipe]")
    ax[1].set_title("Every arm vs recipe (paired, same seeds)\n(grey = infeasible on >=1 seed)")
    ax[1].grid(alpha=.3, axis="x")

    fig.suptitle("Ceiling test: open-loop feed sweep on the real simulator", y=1.02)
    fig.tight_layout()
    fp = out_dir / "feed_sweep_ceiling.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fp


def main(n_seeds=N_EVAL_SEEDS, const_levels=CONST_LEVELS, shaped=SHAPED, out_dir=OUT_DIR):
    print(f"[ceiling] {n_seeds} held-out seeds from {EVAL_BASE}; "
          f"{len(const_levels)} constant levels + {len(shaped)} shaped arms")
    df, seeds = sweep(n_seeds, const_levels, shaped)
    summ = summarise(df, seeds)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_arm_seed.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)
    fp = plot(df, summ, seeds, out_dir)

    show = summ.copy().sort_values("delta_vs_recipe", ascending=False)
    cols = ["arm", "yield_mean", "delta_vs_recipe", "beats_recipe_frac",
            "feasible_frac", "final_P_mean", "max_visc_mean", "wilcoxon_p"]
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== summary (sorted by paired delta vs recipe) ===")
    print(show[cols].to_string(index=False,
          float_format=lambda x: f"{x:8.3f}" if abs(x) < 1000 else f"{x:8.1f}"))

    feas = show[(show.feasible_frac >= 1.0) & (show.arm != "recipe")]
    best = feas.iloc[0] if len(feas) else None
    print("\n=== verdict ===")
    if best is not None and best["delta_vs_recipe"] > 0:
        print(f"A FEASIBLE open-loop arm beats recipe: '{best['arm']}' by "
              f"{best['delta_vs_recipe']:+.1f} kg ({100*best['beats_recipe_frac']:.0f}% of seeds). "
              f"=> the RL is leaving value on the table.")
    else:
        print("No FEASIBLE open-loop arm beats the recipe on this simulator "
              "=> suspect the action space / reward, not the optimiser (see docstring for the "
              "stronger open-loop trajectory-optimisation follow-up).")
    print(f"\nsaved: {out_dir/'per_arm_seed.csv'}\n       {out_dir/'summary.csv'}\n       {fp}")
    return df, summ


if __name__ == "__main__":
    main()

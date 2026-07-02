import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES, PAA_BAND, VISC_MAX,
                                    WARMUP_H, FPAA_MIN, FPAA_MAX)
from utils.recipe import Recipe
from utils.constants import STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import DISCHARGE, DISCHARGE_DEFAULT_PROFILE

SEEDS = [1]
RESULTS_DIR = _os.path.join(_ROOT, "results/single_phase/cluster")
OUT = _os.path.join(_ROOT, "results/single_phase/cluster/aggregate")
RECIPE_DIR = _os.path.join(_ROOT, "results/batch_recipe_generation")
NUM_EXPLORATIONS = 5
RECIPE_BATCH = 1

P_IDX = STATE_NAMES.index("P")
_DISCH = Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE)
REF_STYLE = dict(color="red", lw=2.2, ls="--", zorder=6)


def load_recipe_reference(batch=RECIPE_BATCH, recipe_dir=RECIPE_DIR):
    d = Path(recipe_dir)
    m = pd.read_csv(d / "per_batch_metrics.csv").set_index("batch")
    ref = {"batch": batch,
           "final_P": float(m.loc[batch, "final_penicillin_conc"]),
           "yield": float(m.loc[batch, "yield"])}
    col = f"batch_{batch}"
    for key, fname in [("P", "penicillin_concentration_timeseries.csv"),
                       ("PAA", "paa_concentration_timeseries.csv"),
                       ("Viscosity", "viscosity_timeseries.csv"),
                       ("Fpaa", "fpaa_setpoint_timeseries.csv")]:
        fp = d / fname
        if fp.exists():
            ts = pd.read_csv(fp).set_index("time_h")
            ref[key] = (ts.index.values, ts[col].values)
    return ref


def _denorm(x, lo, hi):
    return lo + (x + 1.0) * (hi - lo) / 2.0


def final_P(state_norm):
    P = _denorm(state_norm[:, P_IDX], *STATE_RANGES["P"])
    return float(P[-1]), float(P.mean())


def yield_kg(mon):
    P, V, t = mon["P"], mon["Wt"], mon["t"]
    Fdis = np.array([_DISCH.get_value_at(float(tt)) for tt in t])
    net = (P[-1] * V[-1] - P[0] * V[0]) / 1000.0
    harvest = float((P * Fdis * STEP_IN_HOURS).sum()) / 1000.0
    return net + harvest


def _ep_color(i, n_ep, n_expl):
    if i < n_expl:
        return "0.72"
    span = max(1, n_ep - n_expl - 1)
    return cm.viridis((i - n_expl) / span)


def main(seeds=SEEDS, results_dir=RESULTS_DIR, out=OUT,
         num_explorations=NUM_EXPLORATIONS, recipe_batch=RECIPE_BATCH):
    Path(out).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ref = load_recipe_reference(batch=recipe_batch)
    ref_lbl = f"recipe (seed {ref['batch']})"

    for s in seeds:
        d = Path(results_dir) / f"seed{s}"
        print(d)
        log_p, mon_p = d / "log.pkl", d / "monitor.pkl"
        print('-->', log_p, mon_p)
        if not log_p.exists():
            print("BEDABEDA")
            continue

        hist = pickle.load(open(log_p, "rb"))["state_samples_history"]
        finals = np.array([final_P(b)[0] for b in hist])
        ax[0, 0].plot(np.arange(len(finals)), finals, marker="o", label=f"seed {s}")

        if not mon_p.exists():
            continue
        monitors = pickle.load(open(mon_p, "rb"))
        n_ep = len(monitors)

        yields = []
        for i, m in enumerate(monitors):
            c = _ep_color(i, n_ep, num_explorations)
            lbl = f"ep{i} {'expl' if i < num_explorations else 'trial'}"
            ax[0, 1].plot(m["t"], m["PAA"], color=c, lw=1, alpha=.85, label=lbl)
            ax[0, 2].plot(m["t"], m["Viscosity"], color=c, lw=1, alpha=.85, label=lbl)
            ax[1, 0].plot(m["t"], m["Fpaa"], color=c, lw=1, alpha=.85, label=lbl)
            ax[1, 1].plot(m["t"], m["P"], color=c, lw=1, alpha=.85)
            yields.append(yield_kg(m))
        yields = np.array(yields)
        colors = [_ep_color(i, n_ep, num_explorations) for i in range(n_ep)]
        ax[1, 2].bar(np.arange(n_ep), yields, color=colors)

    ax[0, 0].axhline(ref["final_P"], label=ref_lbl, **REF_STYLE)
    ax[1, 2].axhline(ref["yield"], label=ref_lbl, **REF_STYLE)
    for axis, key in [(ax[0, 1], "PAA"), (ax[0, 2], "Viscosity"),
                      (ax[1, 0], "Fpaa"), (ax[1, 1], "P")]:
        if key in ref:
            t, y = ref[key]
            axis.plot(t, y, label=ref_lbl, **REF_STYLE)

    ax[0, 0].set_title("Final penicillin conc per episode")
    ax[0, 0].set_xlabel("episode (0..=exploration then trials)"); ax[0, 0].set_ylabel("P (g/L)")
    ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=8)

    ax[0, 1].axhspan(*PAA_BAND, color="green", alpha=.12, label="allowed band")
    ax[0, 1].axvline(WARMUP_H, color="gray", ls=":", label="RL on (100 h)")
    ax[0, 1].set_title("PAA conc (all episodes)"); ax[0, 1].set_xlabel("time (h)")
    ax[0, 1].set_ylabel("PAA (mg/L)"); ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize=6, ncol=2)

    ax[0, 2].axhline(VISC_MAX, color="crimson", ls="--", label=f"limit {VISC_MAX:.0f} cP")
    ax[0, 2].axvline(WARMUP_H, color="gray", ls=":", label="RL on (100 h)")
    ax[0, 2].set_title("Viscosity (all episodes)"); ax[0, 2].set_xlabel("time (h)")
    ax[0, 2].set_ylabel("viscosity (cP)"); ax[0, 2].grid(alpha=.3); ax[0, 2].legend(fontsize=6, ncol=2)

    ax[1, 0].axvline(WARMUP_H, color="gray", ls=":", label="RL on (100 h)")
    ax[1, 0].axhspan(FPAA_MIN, FPAA_MAX, color="orange", alpha=.06, label=f"clamp [{FPAA_MIN:.0f},{FPAA_MAX:.0f}]")
    ax[1, 0].set_title("FPAA setpoint = ACTION (all episodes)"); ax[1, 0].set_xlabel("time (h)")
    ax[1, 0].set_ylabel("Fpaa (L/h)"); ax[1, 0].grid(alpha=.3); ax[1, 0].legend(fontsize=6, ncol=2)

    ax[1, 1].axvline(WARMUP_H, color="gray", ls=":", label="RL on (100 h)")
    ax[1, 1].set_title("Penicillin trajectories (all episodes)"); ax[1, 1].set_xlabel("time (h)")
    ax[1, 1].set_ylabel("P (g/L)"); ax[1, 1].grid(alpha=.3)
    ax[1, 1].plot([], [], color="0.72", label="exploration"); ax[1, 1].plot([], [], color=cm.viridis(0.9), label="trials")
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].set_title("Penicillin yield per episode")
    ax[1, 2].set_xlabel("episode"); ax[1, 2].set_ylabel("yield (kg)"); ax[1, 2].grid(alpha=.3, axis="y")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle("Single-phase MC-PILCO baseline: reward (P), action (PAA), constraints, yield")
    fig.tight_layout()
    fig.savefig(Path(out) / "single_phase_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out}/single_phase_summary.png")


if __name__ == "__main__":
    main()

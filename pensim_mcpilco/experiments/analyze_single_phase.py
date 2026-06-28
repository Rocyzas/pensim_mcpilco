"""
Analysis / plots for the single-phase MC-PILCO baseline.

Reads results/single_phase/seed{N}/{log.pkl, monitor.pkl} and produces a 2x3 figure:
  row 1 (existing): final P per trial; PAA conc vs [600,1800] band; viscosity vs 100 cP;
  row 2 (added):    PAA setpoint = the action (all episodes, with range);
                    all-episode penicillin-concentration trajectories;
                    penicillin yield (kg) per episode.
PAA concentration is now an observed state and its band is penalised in the cost, so
the per-episode printout also reports how many steps each episode spends out of band.
Also prints, per episode, the final P, the yield, and the action (Fpaa) range.

Usage:  PYTHONPATH=.. python -m experiments.analyze_single_phase --seeds 1
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
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

P_IDX = STATE_NAMES.index("P")
_DISCH = Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE)

# red bold dashed recipe-baseline reference (see experiments/00_batch_recipe_generation.py)
RECIPE_DIR = "results/batch_recipe_generation"
REF_STYLE = dict(color="red", lw=2.2, ls="--", zorder=6)


def load_recipe_reference(batch=1, recipe_dir=RECIPE_DIR):
    """Recipe-baseline batch (random_seed == batch) used as the red reference line.
    Returns scalar final P / yield plus time-series for P, PAA conc, viscosity and
    Fpaa. Time-series files are optional (re-run 00_batch_recipe_generation to make
    them); a missing file just omits that curve."""
    import pandas as pd
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
    """Penicillin yield (kg) of a batch from its native-resolution monitor:
    net reactor-mass change + harvested (P * discharge * dt). Weight Wt is used as
    the volume proxy (broth ~ water density)."""
    P, V, t = mon["P"], mon["Wt"], mon["t"]
    Fdis = np.array([_DISCH.get_value_at(float(tt)) for tt in t])
    net = (P[-1] * V[-1] - P[0] * V[0]) / 1000.0
    harvest = float((P * Fdis * STEP_IN_HOURS).sum()) / 1000.0
    return net + harvest


def _ep_color(i, n_ep, n_expl):
    if i < n_expl:
        return "0.72"                                  # exploration = grey
    span = max(1, n_ep - n_expl - 1)
    return cm.viridis((i - n_expl) / span)             # trials = dark->bright


def main(seeds, results_dir="results/single_phase",
         out="results/single_phase/aggregate", num_explorations=5,
         recipe_batch=1):
    Path(out).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ref = load_recipe_reference(batch=recipe_batch)
    ref_lbl = f"recipe (seed {ref['batch']})"

    for s in seeds:
        d = Path(results_dir) / f"seed{s}"
        log_p, mon_p = d / "log.pkl", d / "monitor.pkl"
        if not log_p.exists():
            print(f"[skip] {log_p} not found"); continue

        hist = pickle.load(open(log_p, "rb"))["state_samples_history"]
        finals = np.array([final_P(b)[0] for b in hist])     # ep 0.. = exploration then trials
        ax[0, 0].plot(np.arange(len(finals)), finals, marker="o", label=f"seed {s}")
        print(f"seed {s}: final P  expl={finals[:num_explorations].round(2)}  "
              f"trials={finals[num_explorations:].round(2)}")

        if not mon_p.exists():
            continue
        monitors = pickle.load(open(mon_p, "rb"))
        n_ep = len(monitors)

        last = monitors[-1]
        paa_viol = int(((last["PAA"] < PAA_BAND[0]) | (last["PAA"] > PAA_BAND[1])).sum())
        visc_viol = int((last["Viscosity"] > VISC_MAX).sum())
        print(f"        last-episode violations: PAA={paa_viol} steps, viscosity={visc_viol} steps")

        # all panels show every episode (PAA conc, viscosity, action, P, yield)
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

        # per-episode printout: final P, yield, action range, PAA range + band violations
        print(f"        per-episode [type | final P g/L | yield kg | Fpaa range L/h | "
              f"PAA range mg/L | PAA out-of-band steps]:")
        for i, (m, y) in enumerate(zip(monitors, yields)):
            cmask = m["t"] >= WARMUP_H
            fp = m["Fpaa"][cmask]
            paa = m["PAA"][cmask]
            paa_oob = int(((paa < PAA_BAND[0]) | (paa > PAA_BAND[1])).sum())
            typ = "expl " if i < num_explorations else "trial"
            print(f"          ep{i:2d} {typ} | P={final_P(hist[i])[0]:6.2f} | "
                  f"yield={y:7.1f} | Fpaa [{fp.min():.2f},{fp.max():.2f}] | "
                  f"PAA [{paa.min():5.0f},{paa.max():5.0f}] | oob={paa_oob}")

    # ---- recipe-baseline reference (red bold dashed) ----
    ax[0, 0].axhline(ref["final_P"], label=ref_lbl, **REF_STYLE)
    ax[1, 2].axhline(ref["yield"], label=ref_lbl, **REF_STYLE)
    for axis, key in [(ax[0, 1], "PAA"), (ax[0, 2], "Viscosity"),
                      (ax[1, 0], "Fpaa"), (ax[1, 1], "P")]:
        if key in ref:
            t, y = ref[key]
            axis.plot(t, y, label=ref_lbl, **REF_STYLE)
    print(f"recipe reference: batch {ref['batch']}  final P={ref['final_P']:.2f} g/L  "
          f"yield={ref['yield']:.1f} kg")

    # ---- row 1 (existing) ----
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

    # ---- row 2 (added) ----
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
    print(f"\nSaved {out}/single_phase_summary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num_explorations", type=int, default=5,
                   help="how many leading episodes are exploration (for labelling/colour)")
    p.add_argument("--recipe_batch", type=int, default=1,
                   help="recipe-baseline batch (== random_seed) used as the red reference line")
    args = p.parse_args()
    main(args.seeds, num_explorations=args.num_explorations, recipe_batch=args.recipe_batch)

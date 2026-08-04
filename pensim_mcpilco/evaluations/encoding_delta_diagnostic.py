"""Which encoding (linear vs log) should Wt/X/P use for a delta-predicting GP?

The GP fits Delta_z = g(x_{t+1}) - g(x_t), not the trajectory itself. The right encoding g is
the one whose DELTAS are best-behaved for a stationary GP: roughly constant scale across the
batch (no fanning-out over orders of magnitude) and homoscedastic (one sigma_n suffices, no
value-dependent noise). This script rolls out ~20 pure-recipe batches (a=0, no RL policy -- the
GP-encoding question is about the channel's own dynamics, not what a trained policy does to it),
reads Wt/X/P at TRUE, unencoded, native simulator resolution, then re-encodes them under two
candidates and compares the resulting deltas:

    linear  -- z = normalise(x_phys,        lo_phys, hi_phys)
    log     -- z = normalise(log(x_phys),   log(lo_phys), log(hi_phys))   (today's production
                                                                            encoding for Wt/X/P,
                                                                            see pensim_wrapper.py
                                                                            STATE_LOG_CHANNELS)

Both candidates share the SAME physical operating band (lo_phys, hi_phys) -- taken by exponentiating
pensim_wrapper.STATE_RANGES' existing (already-calibrated) log bounds for that channel -- so the
comparison is apples-to-apples: same band, only the encoding function differs.

Why native-resolution reads, not PenSimWrapper.rollout()'s returned `states`: those are already
log-encoded AND clipped to [-1, 1] via STATE_RANGES. For a channel like P, whose real value is
near/at the log floor at batch start, decoding that clipped state back to physical units silently
censors the true value (verified empirically: ~5 of P's early decisions round-trip to the clamped
0.01 g/L floor rather than P's true near-zero value). Reading bx.Wt.y / bx.X.y / bx.P.y directly
from env.get_batches(..., return_batch_data=True) sidesteps this entirely -- that's the same
native array pensim_wrapper.py's own _read() indexes into.

Decision-cadence subsampling: rollout()'s states[d] corresponds to native index
i(d) = (K_WARM - 1) + d * STEPS_PER_DECISION for d = 0..n_decisions (verified by cross-checking
against decode_state_value(..., rollout states) for a matching seed -- X and Wt matched to float
precision, P matched everywhere except the clipped-floor decisions above). n_decisions =
int(CONTROL_H / T_SAMPLING), the same expression rollout() uses internally.

For each channel and encoding this produces, and reports which encoding is better by:
  1. delta vs batch time (all batches overlaid)             -> does delta magnitude fan out?
  2. pooled + early/mid/late-third delta histograms          -> phase-invariant shape?
  3. local delta-std vs the (encoded) value x, binned        -> homoscedastic (flat) or not?

Usage:
    python encoding_delta_diagnostic.py                 # 20 batches, default seed block
    python encoding_delta_diagnostic.py --num_batches 30 --show
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from utils.constants import STEP_IN_HOURS
from utils.peni_env_setup import PenSimEnv
# Importing mcpilco.pensim_wrapper already calls patch_fastodeint() at module load time.
from mcpilco.pensim_wrapper import (
    PenSimWrapper, STATE_RANGES, STATE_LOG_FLOOR, K_WARM, CONTROL_H, T_SAMPLING,
    STEPS_PER_DECISION, _normalise,
)

CHANNELS = ["Wt", "X", "P"]
ENCODINGS = ["linear", "log"]

# Reserved seed block for this offline/read-only diagnostic -- clear of MEASUREMENT_SEED_BASE
# (900_000, pensim_wrapper._measure_init_state_stats) and of every seed*1000 training block
# (seed < 900). These batches never touch GP training data.
DIAG_SEED_BASE = 950_000

# Binning for the delta-std-vs-x plot: fixed over [-1, 1], the GP's actual conditioning domain
# (not the observed sample range), so "flat across x" means flat across the whole space a GP
# would actually be asked to interpolate over. Bins with too few points are dropped from the
# flatness score (a std computed from 3 points is noise, not evidence of heteroscedasticity).
NBINS_X = 20
MIN_BIN_COUNT = 20

# Bins for pooled/early/mid/late delta histograms and their pairwise overlap coefficient.
NBINS_HIST = 30

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding_delta_diagnostic")


def get_decision_trajectory(seed):
    """One pure-recipe batch's TRUE physical Wt/X/P, subsampled at decision cadence (see module
    docstring for the i(d) formula and why this reads bx.<channel>.y directly rather than going
    through the log-encoded/clipped rollout() states."""
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    _df, _yield, bx = env.get_batches(random_seed=seed, include_raman=False, return_batch_data=True)
    n_decisions = int(CONTROL_H / T_SAMPLING)
    idx = (K_WARM - 1) + np.arange(n_decisions + 1) * STEPS_PER_DECISION
    t_h = (idx + 1) * STEP_IN_HOURS
    return t_h, {c: np.asarray(getattr(bx, c).y)[idx] for c in CHANNELS}


def collect_batches(num_batches, seed_base):
    t_h = None
    phys = {c: [] for c in CHANNELS}
    for i in range(num_batches):
        t_h_i, vals = get_decision_trajectory(seed_base + i)
        t_h = t_h_i if t_h is None else t_h  # deterministic clock -- identical across batches
        for c in CHANNELS:
            phys[c].append(vals[c])
    return t_h, {c: np.stack(phys[c]) for c in CHANNELS}  # each (num_batches, n_decisions+1)


def encode_log(x, lo_log, hi_log):
    """Today's production encoding for Wt/X/P (see pensim_wrapper.encode_state_value)."""
    return _normalise(np.log(np.maximum(x, STATE_LOG_FLOOR)), lo_log, hi_log)


def encode_linear(x, lo_log, hi_log):
    """Same physical band as encode_log (exponentiating STATE_RANGES' log bounds), no log."""
    lo_phys, hi_phys = np.exp(lo_log), np.exp(hi_log)
    return _normalise(x, lo_phys, hi_phys)


def overlap_coefficient(a, b, bin_edges):
    """1.0 = identical distributions over these bins, 0.0 = fully disjoint. Used to score
    whether early/mid/late delta histograms overlap (phase-invariant) or separate."""
    ha, _ = np.histogram(a, bins=bin_edges)
    hb, _ = np.histogram(b, bins=bin_edges)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.sum(np.minimum(ha, hb)))


def binned_delta_std(x_pre, delta, nbins=NBINS_X, min_count=MIN_BIN_COUNT):
    """Std (and mean, count) of delta within each of `nbins` equal-width bins over [-1, 1] of
    x_pre (the encoded value BEFORE the step that produced delta). Bins under `min_count` are
    kept in the returned table (count=0 rows excluded) but excluded from the flatness score by
    the caller."""
    edges = np.linspace(-1.0, 1.0, nbins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x_pre >= lo) & (x_pre < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        d = delta[mask]
        rows.append({"bin_lo": lo, "bin_hi": hi, "bin_mid": 0.5 * (lo + hi),
                      "count": n, "delta_mean": float(np.mean(d)), "delta_std": float(np.std(d))})
    return rows


def flatness_cv(bin_rows, min_count=MIN_BIN_COUNT):
    """Coefficient of variation (std/mean) of delta_std ACROSS bins with enough points --
    lower = flatter = more homoscedastic = better for a single GP noise term sigma_n."""
    stds = np.array([r["delta_std"] for r in bin_rows if r["count"] >= min_count])
    if len(stds) < 2 or np.mean(stds) == 0:
        return float("nan")
    return float(np.std(stds) / np.mean(stds))


def analyze_channel(channel, t_h, phys):
    """Builds both encodings' delta series/stats for one channel. Returns a dict with everything
    the plotting/reporting functions need."""
    lo_log, hi_log = STATE_RANGES[channel]
    x_phys = phys[channel]  # (num_batches, n_decisions+1)

    z = {"linear": encode_linear(x_phys, lo_log, hi_log),
         "log": encode_log(x_phys, lo_log, hi_log)}
    delta = {enc: np.diff(z[enc], axis=1) for enc in ENCODINGS}  # (num_batches, n_decisions)
    x_pre = {enc: z[enc][:, :-1] for enc in ENCODINGS}
    t_mid = 0.5 * (t_h[:-1] + t_h[1:])

    n_decisions = delta["linear"].shape[1]
    thirds = np.array_split(np.arange(n_decisions), 3)
    third_names = ["early", "mid", "late"]

    result = {"channel": channel, "t_mid": t_mid, "z": z, "delta": delta, "x_pre": x_pre,
              "thirds": dict(zip(third_names, thirds))}

    for enc in ENCODINGS:
        bin_rows = binned_delta_std(x_pre[enc].ravel(), delta[enc].ravel())
        result.setdefault("bin_rows", {})[enc] = bin_rows
        result.setdefault("flatness_cv", {})[enc] = flatness_cv(bin_rows)

        pooled = delta[enc].ravel()
        third_vals = {name: delta[enc][:, idxs].ravel() for name, idxs in zip(third_names, thirds)}
        result.setdefault("third_vals", {})[enc] = third_vals
        result.setdefault("third_stats", {})[enc] = {
            name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(v.size)}
            for name, v in third_vals.items()}

        edges = np.histogram_bin_edges(pooled, bins=NBINS_HIST)
        result.setdefault("overlap", {})[enc] = {
            "early_late": overlap_coefficient(third_vals["early"], third_vals["late"], edges),
            "early_mid": overlap_coefficient(third_vals["early"], third_vals["mid"], edges),
            "mid_late": overlap_coefficient(third_vals["mid"], third_vals["late"], edges),
        }
        result.setdefault("pooled", {})[enc] = pooled

    return result


def plot_delta_vs_time(result, out_dir, show):
    channel = result["channel"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, enc in zip(axes, ENCODINGS):
        delta = result["delta"][enc]
        for b in range(delta.shape[0]):
            ax.plot(result["t_mid"], delta[b], color="C0", alpha=0.35, lw=0.7)
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{channel} -- {enc}")
        ax.set_xlabel("Batch time (h)")
    axes[0].set_ylabel("Delta z (per decision step)")
    fig.suptitle(f"{channel}: per-step delta vs batch time, all batches overlaid")
    fig.tight_layout()
    _save(fig, out_dir, f"delta_vs_time_{channel}.png", show)


def plot_delta_hist(result, out_dir, show, show_thirds=True):
    channel = result["channel"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, enc in zip(axes, ENCODINGS):
        pooled = result["pooled"][enc]
        edges = np.histogram_bin_edges(pooled, bins=NBINS_HIST)
        if show_thirds:
            # pooled as a faint backdrop, with the early/mid/late thirds overlaid as outlines
            ax.hist(pooled, bins=edges, density=True, color="0.6", alpha=0.5, label="pooled")
            for name, color in zip(("early", "mid", "late"), ("C0", "C1", "C2")):
                ax.hist(result["third_vals"][enc][name], bins=edges, density=True,
                         histtype="step", lw=1.6, color=color, label=name)
            ov = result["overlap"][enc]
            ax.set_title(f"{channel} -- {enc}\noverlap(early,late)={ov['early_late']:.2f}")
        else:
            # just the whole-batch delta distribution -- shape, spread, symmetry about 0
            ax.hist(pooled, bins=edges, density=True, color="C0", alpha=0.75,
                     edgecolor="0.3", label="all steps")
            ax.axvline(0.0, color="k", lw=0.8, ls=":")
            ax.set_title(f"{channel} -- {enc}\nmean={np.mean(pooled):+.3f} std={np.std(pooled):.3f}")
        ax.set_xlabel("Delta z")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    suffix = "pooled vs early/mid/late thirds" if show_thirds else "all steps pooled"
    fig.suptitle(f"{channel}: delta histogram, {suffix}")
    fig.tight_layout()
    name = f"delta_hist_{channel}.png" if show_thirds else f"delta_hist_pooled_{channel}.png"
    _save(fig, out_dir, name, show)


def plot_delta_std_vs_x(result, out_dir, show):
    channel = result["channel"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, enc in zip(axes, ENCODINGS):
        rows = result["bin_rows"][enc]
        mids = np.array([r["bin_mid"] for r in rows])
        stds = np.array([r["delta_std"] for r in rows])
        counts = np.array([r["count"] for r in rows])
        ok = counts >= MIN_BIN_COUNT
        ax.plot(mids[ok], stds[ok], "o-", color="C3", label=f"n>={MIN_BIN_COUNT}")
        ax.plot(mids[~ok], stds[~ok], "x", color="0.7", label=f"n<{MIN_BIN_COUNT} (excluded)")
        cv = result["flatness_cv"][enc]
        ax.set_title(f"{channel} -- {enc}\nflatness CV={cv:.2f}")
        ax.set_xlabel("x (encoded value, pre-step)")
        ax.set_xlim(-1.05, 1.05)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Local std(Delta z)")
    fig.suptitle(f"{channel}: local delta std vs (encoded) value -- flatter = more homoscedastic")
    fig.tight_layout()
    _save(fig, out_dir, f"delta_std_vs_x_{channel}.png", show)


def _save(fig, out_dir, name, show):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def recommend(result):
    """Per-channel verdict: which encoding is flatter (homoscedastic) and which is more
    phase-invariant (early/mid/late overlap). If they disagree, that disagreement IS the result
    (encoding-level non-stationarity -- see module docstring / user's framing)."""
    cv = result["flatness_cv"]
    ov = result["overlap"]
    flat_winner = "log" if cv["log"] < cv["linear"] else "linear"
    overlap_winner = "log" if ov["log"]["early_late"] > ov["linear"]["early_late"] else "linear"
    if flat_winner == overlap_winner:
        recommendation, note = flat_winner, "flatness and phase-invariance agree"
    else:
        recommendation = "no single winner"
        note = (f"flatness favors {flat_winner}, phase-invariance favors {overlap_winner} -- "
                 "encoding-level non-stationarity (a multi-phase hook)")
    return {
        "channel": result["channel"],
        "flatness_cv_linear": cv["linear"], "flatness_cv_log": cv["log"],
        "overlap_early_late_linear": ov["linear"]["early_late"],
        "overlap_early_late_log": ov["log"]["early_late"],
        "flatness_winner": flat_winner, "phase_invariance_winner": overlap_winner,
        "recommendation": recommendation, "note": note,
    }


def save_csvs(results, recommendations, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "delta_std_vs_x.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "encoding", "bin_lo", "bin_hi", "bin_mid",
                                           "count", "delta_mean", "delta_std"])
        w.writeheader()
        for r in results:
            for enc in ENCODINGS:
                for row in r["bin_rows"][enc]:
                    w.writerow({"channel": r["channel"], "encoding": enc, **row})
    print(f"Saved {path}")

    path = os.path.join(out_dir, "phase_thirds_stats.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "encoding", "third", "mean", "std", "n",
                                           "overlap_early_late", "overlap_early_mid", "overlap_mid_late"])
        w.writeheader()
        for r in results:
            for enc in ENCODINGS:
                ov = r["overlap"][enc]
                for third, stats in r["third_stats"][enc].items():
                    w.writerow({"channel": r["channel"], "encoding": enc, "third": third,
                                **stats, "overlap_early_late": ov["early_late"],
                                "overlap_early_mid": ov["early_mid"], "overlap_mid_late": ov["mid_late"]})
    print(f"Saved {path}")

    path = os.path.join(out_dir, "summary_recommendation.csv")
    with open(path, "w", newline="") as f:
        fieldnames = list(recommendations[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in recommendations:
            w.writerow(rec)
    print(f"Saved {path}")


def run(num_batches=20, seed_base=DIAG_SEED_BASE, out_dir=OUT_DIR, show=False, show_thirds=True):
    print(f"Rolling {num_batches} pure-recipe batches (seeds {seed_base}..{seed_base + num_batches - 1})...")
    t_h, phys = collect_batches(num_batches, seed_base)

    results, recommendations = [], []
    for channel in CHANNELS:
        result = analyze_channel(channel, t_h, phys)
        plot_delta_vs_time(result, out_dir, show)
        plot_delta_hist(result, out_dir, show, show_thirds=show_thirds)
        plot_delta_std_vs_x(result, out_dir, show)
        results.append(result)

        rec = recommend(result)
        recommendations.append(rec)
        print(f"[{channel}] flatness CV: linear={rec['flatness_cv_linear']:.3f} "
              f"log={rec['flatness_cv_log']:.3f} (lower=flatter) | "
              f"overlap(early,late): linear={rec['overlap_early_late_linear']:.3f} "
              f"log={rec['overlap_early_late_log']:.3f} (higher=more phase-invariant) "
              f"-> {rec['recommendation']} ({rec['note']})")

    save_csvs(results, recommendations, out_dir)
    print(f"Saved plots to {os.path.normpath(out_dir)}")
    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=DIAG_SEED_BASE)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no_thirds", action="store_true",
                        help="plot just the whole-batch pooled delta histogram, "
                             "without the early/mid/late overlay")
    args = parser.parse_args()
    run(num_batches=args.num_batches, seed_base=args.seed_base, out_dir=args.out_dir,
        show=args.show, show_thirds=not args.no_thirds)

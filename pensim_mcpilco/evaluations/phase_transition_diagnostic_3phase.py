"""Offline growth->production->decline 3-phase diagnostic.

Both IndPenSim papers name only two phases in prose ("rapid-growth" and "productive"), but the
structured-biomass model underneath (which PenSimPy implements line-for-line) tracks four
morphological compartments -- A0 (growing), A1 (non-growing, product-forming), A3
(degenerated), A4 (autolysed) -- and the papers themselves describe a real third regime: total
biomass X peaking (~hour 150 in the 2015 paper's Batch 3) and then declining as A1 degenerates
into A3 and then autolyses into A4. That's not "production, just slower" -- it's governed by
its own kinetics (the vacuole population-balance sub-model, dy[15]-dy[26] in
indpensim_ode_py.py) and is a real, third dynamical regime.

  Boundary 1 (growth -> production): PRIMARY = A0's own peak (detect_pivot_a0, imported from
  phase_transition_diagnostic_updated.py), cross-checked against the mu_X-collapse/S-depletion
  proxy. An earlier version of this script used K_diff(t) reaching its floor (Eq. 9-10) here
  instead -- that's the literal equation-named "differentiation switch" in the papers, and it
  IS computed exactly (no proxy) from Culture_age/X. But running it against this recipe showed
  it fires at ~150-154h, essentially concurrent with A1's OWN peak (~159h) -- i.e. it marks the
  END of the production window, not the start. Penicillin (P) is visibly, steadily
  accumulating here from ~hour 20-30 onward (matching the papers' own "~hour 20" Fs-cut
  anchor), so a boundary1 near hour 150 would misclassify ~130h of actively-producing batch
  time as "growth". K_diff(t) is still plotted (see panel 3) as context -- it's a real signal,
  just not a good boundary1 for THIS recipe/duration -- see phase_transition_diagnostic_updated.py's
  docstring for the full detector, kept there for reuse and comparison.

  Boundary 2 (production -> decline): A1's own peak (dA1/dt sign change, + to -). A1 is the
  compartment that actually makes penicillin (r_p is a function of A1, not A0 --
  indpensim_ode_py.py:268); once A1 peaks, degeneration into A3 (da_1_dt's vacuole term,
  indpensim_ode_py.py:329-332) is outpacing branching/differentiation replenishment. Total
  biomass X = a0+a1+a3+a4 peaking is kept as a cross-check, since that's the paper's own stated
  observable anchor for this transition (Fig. 3A).

IMPORTANT -- batch duration mismatch: this repo's batches run to BATCH_LENGTH_HOURS (currently
230h, utils/constants.py), not the ~300h+ (up to 317h) IndPenSim case-study batches where the
biomass turnover was actually observed. Do not assume decline is reached at all within this
recipe/duration -- boundary2 detectors return NaN (rather than mis-picking the tail of the
search window) if their signal is still rising when the search window ends. run() reports what
fraction of seeds actually reach a decline boundary within the batch.

Strictly offline/post-hoc, touches no existing file.

Usage:
    python phase_transition_diagnostic_3phase.py 1            # single seed (matches --seed 1)
    python phase_transition_diagnostic_3phase.py 1-5           # range, inclusive
    python phase_transition_diagnostic_3phase.py 1,3,5-7 --show
or:
    from evaluations.phase_transition_diagnostic_3phase import run
    results = run(training_seeds=[0, 1, 2], show=True)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)  # so `import phase_transition_diagnostic_updated` resolves
                                     # whether this file is run directly or imported as a package

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

# Importing this already calls patch_fastodeint() at module load time (via its own
# mcpilco.pensim_wrapper import), required before any PenSimEnv.step()/get_batches() call.
from phase_transition_diagnostic_updated import (
    get_trajectory, _smooth, detect_pivot_a0, compute_proxy_signals, detect_pivot_proxy,
    detect_pivot_kdiff, BATCH_LENGTH_HOURS, SEED_MULTIPLIER, _parse_seed_spec, PIVOT_HOURS,
    MEAN_P,
)

PIVOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point_3phase")


def detect_pivot_a1(t, a1, smooth_window_h=3.0, exclude_before_h=5.0, exclude_after_h=10.0):
    """Boundary 2 (primary): time of A1's peak (dA1/dt sign change, + to -). Mirrors
    detect_pivot_a0's global-argmax approach, so it carries the same known limitation flagged
    for A0 in pensim_wrapper.py -- if A1(t) has 2+ comparable local maxima in some batch, the
    global-max pivot is a similarly fragile tie-break. Not fixed here (out of scope for a
    diagnostic), just documented, same as the codebase already does for A0.

    Returns (pivot_hours, a1_smoothed); pivot_hours is NaN if A1 is still rising at the end of
    the search window, i.e. decline hasn't started within this batch's duration -- see module
    docstring on the 230h-vs-~300h mismatch."""
    a1_s = _smooth(a1, t, smooth_window_h)
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    peak_idx = idxs[np.argmax(a1_s[idxs])]
    if peak_idx == idxs[-1]:
        return float("nan"), a1_s
    return float(t[peak_idx]), a1_s


def detect_pivot_X(t, X, smooth_window_h=3.0, exclude_before_h=5.0, exclude_after_h=10.0):
    """Boundary 2 cross-check: peak of total biomass X = a0+a1+a3+a4 -- the paper's own stated
    observable for this transition (Fig. 3A: biomass peaks then declines as A1 degenerates).
    Same 'still rising at window end -> NaN' handling as detect_pivot_a1."""
    X_s = _smooth(X, t, smooth_window_h)
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    peak_idx = idxs[np.argmax(X_s[idxs])]
    if peak_idx == idxs[-1]:
        return float("nan"), X_s
    return float(t[peak_idx]), X_s


def plot_and_report(training_seed, out_dir=None, show=False):
    """training_seed is the value you'd pass as --seed to experiments/02.../03...; internally
    mapped to the same simulator seed those scripts use for episode 0 (see SEED_MULTIPLIER)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj = get_trajectory(sim_seed)
    t = traj["t"]

    # Boundary 1: A0-peak (primary), mu_X/S proxy (cross-check). See module docstring for why
    # K_diff-floor is no longer used here -- still computed below, plotted as context only.
    boundary1, a0_s = detect_pivot_a0(t, traj["a0"])
    proxy = compute_proxy_signals(t, traj["X"], traj["S"], traj["P"])
    boundary1_proxy = detect_pivot_proxy(t, proxy, traj["S"])
    _kdiff_pivot_context, _A_t1_s, kdiff = detect_pivot_kdiff(t, traj["Culture_age"], traj["X"])

    # Boundary 2: A1-peak (primary), X-peak (cross-check).
    boundary2_a1, a1_s = detect_pivot_a1(t, traj["a1"])
    boundary2_X, X_s = detect_pivot_X(t, traj["X"])
    decline_reached = not np.isnan(boundary2_a1)

    degen_frac = (traj["a3"] + traj["a4"]) / np.clip(traj["X"], 1e-6, None)

    b2_a1_str = f"{boundary2_a1:.1f} h" if decline_reached else f"not reached (<{BATCH_LENGTH_HOURS:.0f}h)"
    b2_X_str = f"{boundary2_X:.1f} h" if not np.isnan(boundary2_X) else f"not reached (<{BATCH_LENGTH_HOURS:.0f}h)"
    print(f"[seed {training_seed} (sim_seed={sim_seed})] "
          f"boundary1 (growth->production, A0 peak) = {boundary1:.1f} h | "
          f"proxy cross-check = {boundary1_proxy:.1f} h | "
          f"boundary2 (production->decline, A1 peak) = {b2_a1_str} | "
          f"X-peak cross-check = {b2_X_str} | "
          f"K_diff-floor (context only, see docstring) = {_kdiff_pivot_context:.1f} h | "
          f"hardcoded single-pivot PIVOT_HOURS = {PIVOT_HOURS:g} h")

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    ax = axes[0]
    ax.plot(t, traj["a0"], color="0.7", lw=0.8, label="A0 (raw)")
    ax.plot(t, a0_s, color="C0", lw=1.5, label="A0 (smoothed, growing)")
    ax.plot(t, a1_s, color="C1", lw=1.5, label="A1 (smoothed, non-growing/producing)")
    ax.plot(t, traj["a3"], color="C2", lw=1.2, label="A3 (degenerated)")
    ax.plot(t, traj["a4"], color="C3", lw=1.2, label="A4 (autolysed)")
    ax.axvline(boundary1, color="purple", lw=1.8, label=f"boundary1: growth->production ({boundary1:.1f} h)")
    ax.axvline(boundary1_proxy, color="teal", ls=":", lw=1.2, label=f"proxy cross-check ({boundary1_proxy:.1f} h)")
    if decline_reached:
        ax.axvline(boundary2_a1, color="darkred", lw=1.8,
                   label=f"boundary2: production->decline ({boundary2_a1:.1f} h)")
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0, label=f"hardcoded single PIVOT_HOURS ({PIVOT_HOURS:g} h)")
    ax.set_ylabel("Biomass region (g/L)")
    ax.set_title(f"3-phase split -- seed {training_seed} (sim_seed={sim_seed})")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    ax.plot(t, traj["X"], color="k", lw=1.2, label="X (total biomass)")
    ax.plot(t, traj["S"], label="S (substrate)")
    ax.plot(t, traj["P"], label="P (penicillin)")
    if not np.isnan(boundary2_X):
        ax.axvline(boundary2_X, color="darkred", ls=":", lw=1.5, label=f"X-peak ({boundary2_X:.1f} h)")
    ax.axvline(boundary1, color="purple", lw=1.5)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("Conc. (g/L)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t, kdiff, color="C4", lw=1.5, label="K_diff(t) -- context only, NOT a boundary here")
    axb = ax.twinx()
    axb.plot(t, proxy["mu_X"], color="C0", alpha=0.6, label="mu_X (specific growth rate, 1/h)")
    ax.axvline(boundary1, color="purple", lw=1.5)
    ax.axvline(boundary1_proxy, color="teal", ls=":", lw=1.2)
    if decline_reached:
        ax.axvline(boundary2_a1, color="darkred", lw=1.5)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("K_diff (g/L)")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="center right")

    ax = axes[3]
    ax.plot(t, degen_frac, color="C5", lw=1.5, label="(A3+A4) / X  -- degenerated+autolysed fraction")
    ax.axvline(boundary1, color="purple", lw=1.5)
    if decline_reached:
        ax.axvline(boundary2_a1, color="darkred", lw=1.8)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Degenerated fraction")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"phase_transition_3phase_seed{training_seed}.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"training_seed": training_seed, "sim_seed": sim_seed,
            "boundary1_a0_peak_hours": boundary1, "boundary1_proxy_hours": boundary1_proxy,
            "kdiff_floor_hours_context": _kdiff_pivot_context,
            "boundary2_a1_peak_hours": boundary2_a1, "boundary2_X_peak_hours": boundary2_X,
            "decline_reached": decline_reached, "hardcoded_pivot_hours": PIVOT_HOURS}


def _save_csv(results, out_dir):
    """Per-seed boundaries, plus a nan-aware mean/std/min/max summary."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pivot_points_3phase.csv")
    fieldnames = ["training_seed", "sim_seed", "boundary1_a0_peak_hours", "boundary1_proxy_hours",
                  "kdiff_floor_hours_context", "boundary2_a1_peak_hours", "boundary2_X_peak_hours",
                  "decline_reached", "hardcoded_pivot_hours"]
    cols = {name: np.array([r[name] for r in results]) for name in
            ("boundary1_a0_peak_hours", "boundary1_proxy_hours", "kdiff_floor_hours_context",
             "boundary2_a1_peak_hours", "boundary2_X_peak_hours")}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
        for label, fn in (("mean", np.nanmean), ("std", np.nanstd),
                          ("min", np.nanmin), ("max", np.nanmax)):
            row = {"training_seed": label, "sim_seed": "", "decline_reached": "",
                   "hardcoded_pivot_hours": PIVOT_HOURS}
            row.update({name: fn(vals) for name, vals in cols.items()})
            w.writerow(row)
    print(f"Saved {path}")
    return path


def run(training_seeds=(0,), out_dir=PIVOT_DIR, show=False):
    results = [plot_and_report(s, out_dir=out_dir, show=show) for s in training_seeds]
    n_reached = sum(r["decline_reached"] for r in results)
    b1_vals = [r["boundary1_a0_peak_hours"] for r in results]
    b1p_vals = [r["boundary1_proxy_hours"] for r in results]
    b2_vals = [r["boundary2_a1_peak_hours"] for r in results]
    print(f"\nBoundary1 (growth->production, A0-peak) over {len(results)} batch(es): "
          f"{np.nanmean(b1_vals):.1f} +/- {np.nanstd(b1_vals):.1f} h "
          f"(proxy cross-check: {np.nanmean(b1p_vals):.1f} +/- {np.nanstd(b1p_vals):.1f} h)")
    print(f"Boundary2 (production->decline, A1-peak) reached in {n_reached}/{len(results)} "
          f"batch(es) within {BATCH_LENGTH_HOURS:.0f}h"
          + (f": {np.nanmean(b2_vals):.1f} +/- {np.nanstd(b2_vals):.1f} h" if n_reached else "")
          + f" (hardcoded single-phase PIVOT_HOURS = {PIVOT_HOURS:g} h, for reference)")
    if n_reached < len(results):
        print(f"NOTE: {len(results) - n_reached}/{len(results)} batch(es) never reached a "
              f"production->decline boundary within {BATCH_LENGTH_HOURS:.0f}h. The IndPenSim "
              f"papers observed this turnover around hour ~150 in ~300h+ batches -- this "
              f"recipe/duration may simply end before decline sets in. A 3rd GP phase is only "
              f"worth adding if this fraction is meaningfully non-zero across your seed sweep.")
    if out_dir:
        _save_csv(results, out_dir)
        print(f"Saved plots to {os.path.normpath(out_dir)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=str,
                        help="Seed(s) to evaluate, matching the --seed you'd pass to "
                             "experiments/02_mcpilco_single_phase.py / 03_mcpilco_dual_phase.py. "
                             "Single int ('1'), inclusive range ('1-5'), or comma-separated "
                             "mix ('1,3,5-7').")
    parser.add_argument("--out_dir", type=str, default=PIVOT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(training_seeds=_parse_seed_spec(args.seeds), out_dir=args.out_dir, show=args.show)

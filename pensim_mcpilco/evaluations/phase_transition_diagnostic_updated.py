"""Offline growth->production phase-transition (pivot) diagnostic -- updated primary detector.

Forked from phase_transition_diagnostic.py. That script's primary detector was A0's peak
(dA0/dt sign change), justified as "no proxy needed" since A0 is a real ODE state. But A0
peaking is actually a *downstream* consequence of several mechanisms at once (K_diff falling,
substrate depletion via K_b/K_e, A1 feeding back into branching r_b0, dilution) -- and
pensim_wrapper.py's own PIVOT_HOURS comment documents that its output is NOT unimodal (2+
comparable local maxima per batch), making the global-max pivot a fragile tie-break.

This version's PRIMARY detector is the model's actual, literal growth->production switch: the
point where K_diff(t) saturates at its floor (Eq. 9-10 in Goldrick et al. 2015; K_diff_L=0.09
in indpensim_ode_py.py:24). That equation needs "mean cell culture age" A_t1, which both papers
say is deferred to Tiller et al. (1994) and not reproducible from their text alone -- but
PenSimPy already computes and exposes it every step: A_t1 = Culture_age(t) / X(t), where
Culture_age is ODE state y[10] (dy[10] = a0+a1+a3+a4, i.e. an integral of total biomass over
time -- see indpensim_ode_py.py:169,443) and is a first-class channel, bx.Culture_age
(batch_data.py:53; populated at peni_env_setup.py:271). No proxy, no external paper needed --
this is the exact formula the simulator itself runs, computed post-hoc from data
PenSimEnv.get_batches(..., return_batch_data=True) already returns.

K_diff(t) = max(0.75 - 0.006*A_t1(t), 0.09) (constants confirmed against both
PenSimPy/pensimpy/ode/indpensim_ode_py.py and the original MATLAB IndPenSim_V2.02/
Parameter_list.m -- the two agree exactly). Solving 0.75 - 0.006*At = 0.09 gives At=110h; this
script detects the first time A_t1(t) actually reaches that, i.e. the first time K_diff(t)
touches its floor -- a single, exact, closed-form crossing rather than an integrated proxy.

The old A0-peak and mu_X-collapse/S-proxy detectors are kept as cross-checks (they should land
in the same neighbourhood as the K_diff pivot if it's a good primary), not because they're
wrong, but because independent detectors agreeing is the actual evidence a single detector
picked something real.

Batches here run to BATCH_LENGTH_HOURS (see below, currently 230h -- utils/constants.py), not
the ~300h+ IndPenSim case-study batches referenced in the papers. Don't treat any paper-stated
hour (e.g. "~hour 20", "~hour 150") as applicable here -- always cross-check against what this
simulator actually does at this recipe/duration, which is exactly what this script computes.

Strictly offline/post-hoc, touches no existing file.

Usage:
    python phase_transition_diagnostic_updated.py 1            # single seed (matches --seed 1)
    python phase_transition_diagnostic_updated.py 1-5           # range, inclusive
    python phase_transition_diagnostic_updated.py 1,3,5-7 --show
or:
    from evaluations.phase_transition_diagnostic_updated import run
    results = run(training_seeds=[0, 1, 2], show=True)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)  # so `import phase_transition_diagnostic_updated` works
                                     # from phase_transition_diagnostic_3phase.py either way

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from utils.constants import STEP_IN_HOURS, BATCH_LENGTH_IN_MINUTES, MINUTES_PER_HOUR
from utils.peni_env_setup import PenSimEnv
# Importing mcpilco.pensim_wrapper already calls patch_fastodeint() at module load time
# (required before any PenSimEnv.step()/get_batches() call), so it's not repeated here.
from mcpilco.pensim_wrapper import PenSimWrapper, PIVOT_HOURS

BATCH_LENGTH_HOURS = BATCH_LENGTH_IN_MINUTES / MINUTES_PER_HOUR

# Gaussian center (g/L) of the penicillin-production-rate term r_p -- the substrate
# concentration production is centered around, i.e. the paper's "s_max_P" analogue. Hardcoded
# local constant in PenSimPy/pensimpy/ode/indpensim_ode_py.py:17 (mirrors
# IndPenSim_V2.02/Parameter_list.m:8); not exported by that module, so duplicated here.
MEAN_P = 0.002

# K_diff(t) = max(K_DIFF_BASE - K_DIFF_BETA1 * A_t1(t), K_DIFF_FLOOR) -- exact constants from
# indpensim_ode_py.py:21-24,238-240, cross-checked against IndPenSim_V2.02/Parameter_list.m:12-15
# (both agree exactly). A_t1 is the simulator's own "mean cell culture age" term (see module
# docstring): A_t1 = Culture_age(t) / X(t).
K_DIFF_BASE = 0.75
K_DIFF_BETA1 = 0.006
K_DIFF_FLOOR = 0.09
# Solving K_DIFF_BASE - K_DIFF_BETA1 * At = K_DIFF_FLOOR for At (hours, since A_t1 = integral of
# X dt / X has units of time).
K_DIFF_FLOOR_AT_HOURS = (K_DIFF_BASE - K_DIFF_FLOOR) / K_DIFF_BETA1

# Matches config_single_phase.py:194 / config_dual_phase.py:174's wrapper_par["seed_offset"] =
# seed*1000 -- the mapping from a training run's --seed to its actual (episode-0) simulator seed.
SEED_MULTIPLIER = 1000

PIVOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point_updated")


def _parse_seed_spec(spec):
    """Parses '1', '1-5', or a comma-separated mix like '1,3,5-7' into a sorted list of unique
    training seeds (the same values you'd pass as --seed to experiments/02.../03...)."""
    seeds = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-")
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def get_trajectory(seed):
    """Runs one default-recipe batch and pulls its raw per-timestep channels, including
    Culture_age (ODE state y[10], needed for the K_diff detector) and a3/a4 (degenerated /
    autolysed regions, not used here but kept so phase_transition_diagnostic_3phase.py can
    reuse this function without a second sim run)."""
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    (_df, _df_raman), _yield, bx = env.get_batches(
        random_seed=seed, include_raman=False, return_batch_data=True)
    n = len(bx.X.y)
    t = np.array([(i + 1) * STEP_IN_HOURS for i in range(n)])
    return {
        "t": t,
        "X": np.array(bx.X.y), "S": np.array(bx.S.y), "P": np.array(bx.P.y),
        "a0": np.array(bx.a0.y), "a1": np.array(bx.a1.y),
        "a3": np.array(bx.a3.y), "a4": np.array(bx.a4.y),
        "Culture_age": np.array(bx.Culture_age.y),
        "OUR": np.array(bx.OUR.y), "CER": np.array(bx.CER.y),
        "Fs": np.array(bx.Fs.y),
        "Wt": np.array(bx.Wt.y), "Viscosity": np.array(bx.Viscosity.y),
    }


def _smooth(y, t, window_h):
    dt = float(np.mean(np.diff(t)))
    window = max(5, int(round(window_h / dt)))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    return savgol_filter(y, window, polyorder=min(3, window - 1))


def compute_kdiff(t, culture_age, X, smooth_window_h=3.0):
    """A_t1(t) = Culture_age(t)/X(t) -- the simulator's own mean-cell-age term (see module
    docstring), smoothed before dividing into K_diff's linear-decay formula since Culture_age/X
    is a ratio of two noisy signals. Returns (A_t1_smoothed, kdiff_raw, kdiff_floored) --
    kdiff_raw (unclamped) is what the crossing detector below actually tests."""
    X_safe = np.clip(X, 1e-6, None)
    A_t1 = culture_age / X_safe
    A_t1_s = _smooth(A_t1, t, smooth_window_h)
    kdiff_raw = K_DIFF_BASE - K_DIFF_BETA1 * A_t1_s
    kdiff = np.maximum(kdiff_raw, K_DIFF_FLOOR)
    return A_t1_s, kdiff_raw, kdiff


def detect_pivot_kdiff(t, culture_age, X, smooth_window_h=3.0, exclude_before_h=5.0,
                        exclude_after_h=10.0, persist_h=2.0):
    """PRIMARY detector: first time K_diff(t) saturates at its floor (0.09) -- the model's own,
    literal growth->production differentiation switch (Eq. 9-10), computed exactly rather than
    approximated. Requires the crossing to persist for `persist_h` hours before accepting it,
    since discharge events (DISCHARGE_DEFAULT_PROFILE) perturb every concentration state
    including biomass and can otherwise cause a single-sample transient dip in A_t1 to be
    mistaken for the real, sustained crossing.

    Returns (pivot_hours, A_t1_smoothed, kdiff_floored); pivot_hours is NaN if the floor is
    never reached (sustained) within the batch -- explicit, not a silent mis-pick."""
    A_t1_s, kdiff_raw, kdiff = compute_kdiff(t, culture_age, X, smooth_window_h)
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    dt = float(np.mean(np.diff(t)))
    persist_steps = max(1, int(round(persist_h / dt)))
    below = kdiff_raw <= K_DIFF_FLOOR
    for idx in idxs:
        end = idx + persist_steps
        if end <= len(below) and np.all(below[idx:end]):
            return float(t[idx]), A_t1_s, kdiff
    return float("nan"), A_t1_s, kdiff


def detect_pivot_a0(t, a0, smooth_window_h=3.0, exclude_before_h=5.0, exclude_after_h=10.0):
    """Cross-check #1 (was the primary detector in phase_transition_diagnostic.py): time of
    A0's peak (dA0/dt sign change, + to -). Kept here because it's a real, independent signal
    -- if it disagrees badly with the K_diff pivot, that's worth knowing, not just kept because
    "no weights, nothing arbitrary" makes it clean (see module docstring for why it's no longer
    primary)."""
    a0_s = _smooth(a0, t, smooth_window_h)
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idx = np.flatnonzero(mask)[np.argmax(a0_s[mask])]
    return float(t[idx]), a0_s


def compute_proxy_signals(t, X, S, P, smooth_window_h=3.0):
    """Cross-check #2's inputs: specific growth rate mu_X = Xdot/X, and dP/dt -- the
    measurable proxies a real plant (without direct A0/A1/Culture_age access) would have to
    rely on instead of any of the detectors above."""
    X_s, S_s, P_s = (_smooth(v, t, smooth_window_h) for v in (X, S, P))
    Xdot, Pdot = np.gradient(X_s, t), np.gradient(P_s, t)
    return {"X_s": X_s, "S_s": S_s, "P_s": P_s, "Xdot": Xdot, "Pdot": Pdot,
            "mu_X": Xdot / np.clip(X_s, 1e-6, None)}


def detect_pivot_proxy(t, proxy, S, exclude_before_h=5.0, exclude_after_h=10.0,
                        mu_frac=0.2, s_percentile=10.0):
    """Cross-check #2: first time after mu_X's own peak where growth rate has collapsed to
    `mu_frac` of its peak AND substrate has been driven down near mean_P (approximated as
    below the batch's own `s_percentile`, since mean_P=0.002 g/L is tiny relative to typical S
    scale and a hard absolute threshold would be fragile). Returns NaN if never jointly met."""
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    mu_X = proxy["mu_X"]
    peak_idx = idxs[np.argmax(mu_X[idxs])]
    mu_peak = mu_X[peak_idx]
    s_thresh = np.percentile(S[idxs], s_percentile)
    for idx in idxs[idxs >= peak_idx]:
        if mu_X[idx] <= mu_frac * mu_peak and S[idx] <= s_thresh:
            return float(t[idx])
    return float("nan")


def plot_and_report(training_seed, out_dir=None, show=False):
    """training_seed is the value you'd pass as --seed to experiments/02.../03...; internally
    mapped to the same simulator seed those scripts use for episode 0 (see SEED_MULTIPLIER)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj = get_trajectory(sim_seed)
    t = traj["t"]

    pivot_kdiff, A_t1_s, kdiff = detect_pivot_kdiff(t, traj["Culture_age"], traj["X"])
    pivot_a0, a0_s = detect_pivot_a0(t, traj["a0"])
    proxy = compute_proxy_signals(t, traj["X"], traj["S"], traj["P"])
    pivot_proxy = detect_pivot_proxy(t, proxy, traj["S"])

    print(f"[seed {training_seed} (sim_seed={sim_seed})] "
          f"K_diff-floor pivot = {pivot_kdiff:.1f} h | A0-peak pivot = {pivot_a0:.1f} h | "
          f"proxy pivot = {pivot_proxy:.1f} h | hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h")

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    ax = axes[0]
    ax.plot(t, traj["a0"], color="0.7", lw=0.8, label="A0 (raw)")
    ax.plot(t, a0_s, color="C0", lw=1.5, label="A0 (smoothed)")
    ax.plot(t, traj["a1"], color="C1", lw=1.2, label="A1")
    ax.axvline(pivot_kdiff, color="purple", lw=1.8,
               label=f"K_diff-floor pivot ({pivot_kdiff:.1f} h)")
    ax.axvline(pivot_a0, color="teal", ls=":", lw=1.2, label=f"A0-peak pivot ({pivot_a0:.1f} h)")
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0, label=f"hardcoded PIVOT_HOURS ({PIVOT_HOURS:g} h)")
    ax.set_ylabel("Biomass region (g/L)")
    ax.set_title(f"Growth->production pivot (updated) -- seed {training_seed} (sim_seed={sim_seed})")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    ax.plot(t, traj["X"], label="X (biomass)")
    ax.plot(t, traj["S"], label="S (substrate)")
    ax.plot(t, traj["P"], label="P (penicillin)")
    ax.axvline(pivot_kdiff, color="purple", lw=1.5)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("Conc. (g/L)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t, kdiff, color="C3", lw=1.5, label="K_diff(t)")
    axb = ax.twinx()
    axb.plot(t, A_t1_s, color="C4", alpha=0.6, label="A_t1(t) (mean culture age, h)")
    axb.axhline(K_DIFF_FLOOR_AT_HOURS, color="C4", ls=":", lw=1.0,
                label=f"A_t1 floor threshold ({K_DIFF_FLOOR_AT_HOURS:.0f} h)")
    ax.axhline(K_DIFF_FLOOR, color="C3", ls=":", lw=1.0, label=f"K_diff floor ({K_DIFF_FLOOR})")
    ax.axvline(pivot_kdiff, color="purple", lw=1.5)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_ylabel("K_diff (g/L)")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="center right")

    ax = axes[3]
    ax.plot(t, proxy["mu_X"], color="C0", label="mu_X (specific growth rate, 1/h)")
    axb = ax.twinx()
    axb.plot(t, traj["S"], color="C2", alpha=0.6, label="S")
    axb.axhline(MEAN_P, color="C5", ls=":", lw=1.0, label=f"mean_P ({MEAN_P} g/L)")
    ax.axvline(pivot_proxy, color="teal", lw=1.2, label=f"proxy pivot ({pivot_proxy:.1f} h)")
    ax.axvline(pivot_kdiff, color="purple", lw=1.5)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("mu_X (1/h)")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")

    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"phase_transition_updated_seed{training_seed}.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"training_seed": training_seed, "sim_seed": sim_seed,
            "pivot_kdiff_hours": pivot_kdiff, "pivot_a0_hours": pivot_a0,
            "pivot_proxy_hours": pivot_proxy, "hardcoded_pivot_hours": PIVOT_HOURS}


def _save_csv(results, out_dir):
    """Per-seed pivot points, plus a mean/std/min/max range summary over this run's seeds.
    Uses nan-aware stats since pivot_kdiff_hours/pivot_proxy_hours can be NaN (floor/collapse
    never reached within BATCH_LENGTH_HOURS)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pivot_points_updated.csv")
    fieldnames = ["training_seed", "sim_seed", "pivot_kdiff_hours", "pivot_a0_hours",
                  "pivot_proxy_hours", "hardcoded_pivot_hours"]
    kdiff_vals = np.array([r["pivot_kdiff_hours"] for r in results])
    a0_vals = np.array([r["pivot_a0_hours"] for r in results])
    proxy_vals = np.array([r["pivot_proxy_hours"] for r in results])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
        for label, valk, vala0, valprox in (
                ("mean", np.nanmean(kdiff_vals), np.nanmean(a0_vals), np.nanmean(proxy_vals)),
                ("std", np.nanstd(kdiff_vals), np.nanstd(a0_vals), np.nanstd(proxy_vals)),
                ("min", np.nanmin(kdiff_vals), np.nanmin(a0_vals), np.nanmin(proxy_vals)),
                ("max", np.nanmax(kdiff_vals), np.nanmax(a0_vals), np.nanmax(proxy_vals))):
            w.writerow({"training_seed": label, "sim_seed": "", "pivot_kdiff_hours": valk,
                        "pivot_a0_hours": vala0, "pivot_proxy_hours": valprox,
                        "hardcoded_pivot_hours": PIVOT_HOURS})
    print(f"Saved {path}")
    return path


def run(training_seeds=(0,), out_dir=PIVOT_DIR, show=False):
    results = [plot_and_report(s, out_dir=out_dir, show=show) for s in training_seeds]
    kdiff_vals = [r["pivot_kdiff_hours"] for r in results]
    n_reached = int(np.sum(~np.isnan(kdiff_vals)))
    print(f"\nMean K_diff-floor pivot over {n_reached}/{len(results)} batch(es) that reached it "
          f"within {BATCH_LENGTH_HOURS:.0f}h: {np.nanmean(kdiff_vals):.1f} +/- "
          f"{np.nanstd(kdiff_vals):.1f} h (hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h)")
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

"""Offline changepoint detection via the `ruptures` library, as a cross-check to
phase_transition_diagnostic.py's A0-peak detector.

Defaults to ALL of IndPenSim's OBSERVABLE channels -- everything a real plant could actually
know, whether on-line (continuous sensors/setpoints, sampled every simulator step) or off-line
(periodic lab assay). This is a materially bigger/different set than "what the current
dual-phase GP model observes" (STATE_NAMES = Wt/X/P/Viscosity) -- it's "what IndPenSim's own
measurement model says is knowable at all", on-line + off-line together. Confirmed directly in
the simulator source, not guessed:

  - Off-line: IndPenSim_V2.02/indpensim_run.m:47-49 (`Off_line_m=12` hours sampling interval,
    `Off_line_delay=4` hours analysis delay) and indpensim.m:349-374 / PenSimPy's
    peni_env_setup.py:348-365 build exactly 5 off-line channels (X_offline, P_offline,
    NH3_offline, Viscosity_offline, PAA_offline) by delay-sampling their raw counterparts, NaN
    everywhere else. Critically: the RAW (non-"_offline") X/P/NH3/Viscosity/PAA channels are the
    hidden ODE ground truth, not something a real plant ever observes directly -- only the
    delayed/periodic _offline versions are real "measurements". See OFF_LINE_CHANNELS below;
    _ffill() reproduces "the last lab result you know so far" between samples.
  - On-line: every other channel in PenSimPy/pensimpy/data/batch_data.py:15-79 that corresponds
    to a real sensor, off-gas analyzer, or known setpoint (flow rates, pressure, pH, DO2, T,
    weight, off-gas O2/CO2 % and the OUR/CER derived from them, ...). See ON_LINE_CHANNELS below
    for the exact list and what's excluded (pure ODE-internal/mechanistic states with no sensor
    equivalent -- a0/a1/a3/a4, n0-n9/nm/phi0, mu_X/mu_P -- plus non-physical bookkeeping
    channels and an unpopulated stub, X_CER, all noted inline).

Drops the A0-peak detector's shape assumption too: PELT/Binseg with an RBF cost model finds
where the joint DISTRIBUTION of the chosen channel(s) changes, a more general (and more
standard, in the changepoint-detection literature) way to locate a regime shift.

Self-contained: fetches its own trajectory directly off PenSimEnv.get_batches(...)'s raw
batch_data object (_get_full_trajectory below) rather than reusing
phase_transition_diagnostic.get_trajectory, which only exposes a small fixed subset of channels
-- keeps this file independently editable. Still reuses _parse_seed_spec/SEED_MULTIPLIER/
PIVOT_HOURS from that module (generic seed-mapping utilities/constants, not trajectory data --
see that module's docstring for the --seed <-> sim_seed mapping rationale).

Usage:
    python phase_transition_ruptures.py 1-5 --show
    python phase_transition_ruptures.py 1 --n_bkps 1
    python phase_transition_ruptures.py 1 --channels a0 a1 --pen 40   # privileged ground truth
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
_sys.path.insert(0, _ROOT)                    # for `utils`, `mcpilco`, `evaluations`
_sys.path.insert(0, _os.path.dirname(_ROOT))  # repo root, for `PenSimPy`

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import ruptures as rpt

from utils.constants import STEP_IN_HOURS
from utils.peni_env_setup import PenSimEnv
# Importing mcpilco.pensim_wrapper already calls patch_fastodeint() at module load time
# (required before any PenSimEnv.step()/get_batches() call), so it's not repeated here.
from mcpilco.pensim_wrapper import PenSimWrapper, PIVOT_HOURS
from evaluations.phase_transition_diagnostic import _parse_seed_spec, SEED_MULTIPLIER

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point")

# On-line: known/measured every simulator step -- flow-rate setpoints, sensors (pH, DO2, T,
# pressure, weight/volume), off-gas analyzer outputs (O2/CO2 %) and the OUR/CER derived from
# them. batch_data.py:15-79.
ON_LINE_CHANNELS = (
    "Fg", "RPM", "Fs", "sc", "abc", "Fa", "Fb", "Fc", "Fh", "Fw", "pressure", "discharge",
    "DO2", "V", "Wt", "pH", "T", "Q", "CO2outgas", "Fpaa", "Foil", "OUR", "O2", "CER",
    "NH3_shots",
)
# Off-line: only sampled periodically (Off_line_m=12h, Off_line_delay=4h --
# IndPenSim_V2.02/indpensim_run.m:47-49), NaN elsewhere in the raw channel -- _ffill()'d below to
# "the last known lab result", same as an operator would only know the latest assay.
OFF_LINE_CHANNELS = ("X_offline", "P_offline", "NH3_offline", "Viscosity_offline", "PAA_offline")

# with on+off ~75h
# with on  ~75h
DEFAULT_CHANNELS = ON_LINE_CHANNELS

# NOT observable, so NOT in DEFAULT_CHANNELS, but still fetched so an explicit --channels
# override can reach them for a ground-truth comparison (e.g. --channels a0 a1, matching
# phase_transition_diagnostic.py's A0-peak detector): the raw un-delayed versions of the 5
# off-line quantities (a real plant only ever sees their _offline counterpart) and the
# structured-biomass compartments.
_PRIVILEGED_CHANNELS = ("X", "P", "NH3", "Viscosity", "PAA", "S", "a0", "a1", "a3", "a4")
# Deliberately excluded entirely (not fetched even via --channels): pure ODE-internal/
# mechanistic states with no real sensor equivalent (n0-n9, nm, phi0 -- hyphal-length-
# distribution states; mup, mux, mu_X_calc, mu_P_calc -- inferred/calculated growth rates, not
# raw measurements; Culture_age -- a morphological model quantity, not something instrumented;
# CO2_d -- internal dissolved-gas state, distinct from the measured off-gas CO2outgas %);
# non-physical bookkeeping (Fault_ref, Control_ref, PAT_ref, Batch_ref, PAA_pred,
# PRBS_noise_addition); X_CER ("Biomass concen. from CER", an inferential soft-sensor channel
# the paper describes -- but confirmed unpopulated/all-zero in this PenSimPy port, so it would
# only add a dead channel); and Raman_Spec (2200-wavenumber spectra, a fundamentally different
# data shape than every other scalar-per-timestep channel here).


def _zscore(y):
    mu, sd = float(np.mean(y)), float(np.std(y))
    return (y - mu) / sd if sd > 1e-12 else y - mu


def _ffill(y):
    """Forward-fills NaN gaps with the last valid value (an off-line channel's value between lab
    samples); any leading NaNs before the first-ever sample are back-filled with that first
    value (there's no earlier reading to hold)."""
    y = np.asarray(y, dtype=float).copy()
    valid = ~np.isnan(y)
    if not valid.any():
        return np.zeros_like(y)
    idx = np.where(valid, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    y_filled = y[idx]
    y_filled[:int(np.argmax(valid))] = y[int(np.argmax(valid))]
    return y_filled


def _get_full_trajectory(seed):
    """Runs one default-recipe batch and pulls IndPenSim's observable channels directly off the
    raw batch_data object -- ON_LINE_CHANNELS + OFF_LINE_CHANNELS (forward-filled), plus
    _PRIVILEGED_CHANNELS reachable only via an explicit --channels override. See this module's
    docstring for the on-line/off-line/privileged classification and its sources."""
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    (_df, _df_raman), _yield, bx = env.get_batches(
        random_seed=seed, include_raman=False, return_batch_data=True)
    n = len(bx.X.y)
    traj = {"t": np.array([(i + 1) * STEP_IN_HOURS for i in range(n)])}
    for name in ON_LINE_CHANNELS + _PRIVILEGED_CHANNELS:
        y = np.array(getattr(bx, name).y, dtype=float)
        if name == "pH":  # stored as 10^(-pH); see pensim_wrapper.py's _read()
            y = -np.log10(np.clip(y, 1e-12, None))
        traj[name] = y
    for name in OFF_LINE_CHANNELS:
        traj[name] = _ffill(np.array(getattr(bx, name).y, dtype=float))
    return traj


def detect_changepoints(traj, channels=DEFAULT_CHANNELS, pen=5.0, model="rbf", n_bkps=None):
    """Returns (times_hours, sample_indices) for each detected changepoint. Channels are
    z-scored first so ruptures isn't dominated by whichever channel has the largest raw scale
    (e.g. OUR ~1e7 vs a0 ~1-20).

    If n_bkps is given, switches from PELT+pen (which can return anywhere from 0 to many
    changepoints depending on how well `pen` happens to be tuned -- 9 of them at this script's
    original pen=5 default) to ruptures.Binseg, which searches for exactly n_bkps changepoints
    directly -- pass n_bkps=1 to force a single pivot point instead of tuning pen by hand."""
    X = np.column_stack([_zscore(traj[c]) for c in channels])
    if n_bkps is not None:
        algo = rpt.Binseg(model=model).fit(X)
        breaks = algo.predict(n_bkps=n_bkps)
    else:
        algo = rpt.Pelt(model=model).fit(X)
        breaks = algo.predict(pen=pen)
    idxs = [b for b in breaks if b < len(X)]  # ruptures' last entry is len(X), an end-of-series
    times = [float(traj["t"][i]) for i in idxs]  # marker, not a real changepoint -- drop it
    return times, idxs


def plot_and_report(training_seed, channels=DEFAULT_CHANNELS, pen=5.0, model="rbf",
                     n_bkps=None, out_dir=None, show=False):
    """training_seed is the value you'd pass as --seed to experiments/02.../03...; internally
    mapped to the same simulator seed those scripts use for episode 0 (see SEED_MULTIPLIER)."""
    sim_seed = training_seed * SEED_MULTIPLIER
    traj = _get_full_trajectory(sim_seed)
    t = traj["t"]
    times, idxs = detect_changepoints(traj, channels=channels, pen=pen, model=model, n_bkps=n_bkps)

    method = f"n_bkps={n_bkps}" if n_bkps is not None else f"pen={pen:g}"
    print(f"[seed {training_seed} (sim_seed={sim_seed})] ruptures changepoints "
          f"({len(channels)} channels, {method}): "
          f"{[f'{x:.1f}h' for x in times]} | hardcoded PIVOT_HOURS = {PIVOT_HOURS:g} h")
    print(f"  channels used: {list(channels)}")

    # DEFAULT_CHANNELS is now ~30 channels (all observable on-line+off-line) -- one subplot per
    # channel would be an unusable ~90in figure. Below a small explicit selection, keep the old
    # per-channel-subplot-plus-combined-overlay layout; above it, just the combined overlay
    # (colormapped, legend dropped in favour of the printed channel list above).
    per_channel_panels = len(channels) <= 8
    n_rows = len(channels) + 1 if per_channel_panels else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3 * n_rows if per_channel_panels else 6),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    if per_channel_panels:
        for ax, c in zip(axes, channels):
            ax.plot(t, traj[c], color="C0", label=c)
            for x in times:
                ax.axvline(x, color="crimson", lw=1.3)
            ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.0)
            ax.set_ylabel(c)
            ax.legend(fontsize=7, loc="upper right")
        axes[0].set_title(f"ruptures changepoints -- seed {training_seed} (sim_seed={sim_seed}, "
                           f"{len(channels)} channels, {method})")

    ax = axes[-1]
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(channels)))
    for c, color in zip(channels, colors):
        ax.plot(t, _zscore(traj[c]), color=color, lw=0.9, alpha=0.85,
                label=f"{c} (z-scored)" if per_channel_panels else None)
    for i, x in enumerate(times):
        ax.axvline(x, color="crimson", lw=1.6, label=f"changepoint ({x:.1f} h)" if i == 0 else None)
    ax.axvline(PIVOT_HOURS, color="k", ls="--", lw=1.2, label=f"hardcoded PIVOT_HOURS ({PIVOT_HOURS:g} h)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("z-scored channel value")
    if not per_channel_panels:
        ax.set_title(f"ruptures changepoints -- seed {training_seed} (sim_seed={sim_seed}, "
                     f"{len(channels)} channels, {method}) -- see console for channel list")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"ruptures_seed{training_seed}.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"training_seed": training_seed, "sim_seed": sim_seed,
            "changepoints_hours": times, "hardcoded_pivot_hours": PIVOT_HOURS}


def _save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ruptures_pivot_points.csv")
    max_cp = max((len(r["changepoints_hours"]) for r in results), default=0)
    fieldnames = (["training_seed", "sim_seed"]
                  + [f"changepoint_{i + 1}_hours" for i in range(max_cp)]
                  + ["hardcoded_pivot_hours"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {"training_seed": r["training_seed"], "sim_seed": r["sim_seed"],
                   "hardcoded_pivot_hours": r["hardcoded_pivot_hours"]}
            for i, x in enumerate(r["changepoints_hours"]):
                row[f"changepoint_{i + 1}_hours"] = x
            w.writerow(row)
    print(f"Saved {path}")
    return path


def run(training_seeds=(0,), channels=DEFAULT_CHANNELS, pen=5.0, model="rbf", n_bkps=None,
        out_dir=OUT_DIR, show=False):
    results = [plot_and_report(s, channels=channels, pen=pen, model=model, n_bkps=n_bkps,
                                out_dir=out_dir, show=show) for s in training_seeds]
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
    parser.add_argument("--channels", type=str, nargs="+", default=list(DEFAULT_CHANNELS),
                        help="Trajectory channels to feed ruptures. Default: ALL observable "
                             "on-line+off-line IndPenSim channels (see ON_LINE_CHANNELS/"
                             "OFF_LINE_CHANNELS in this file). Also accepts _PRIVILEGED_CHANNELS "
                             "(e.g. a0, a1, X, P, S, ...) for a ground-truth comparison.")
    parser.add_argument("--pen", type=float, default=5.0,
                        help="PELT penalty (higher = fewer/coarser changepoints). Ignored if "
                             "--n_bkps is given.")
    parser.add_argument("--n_bkps", type=int, default=None,
                        help="Force exactly this many changepoints via ruptures.Binseg instead "
                             "of PELT+pen -- e.g. --n_bkps 1 for a single pivot point.")
    parser.add_argument("--model", type=str, default="rbf",
                        help="ruptures cost model (rbf, l2, l1, ...).")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(training_seeds=_parse_seed_spec(args.seeds), channels=args.channels, pen=args.pen,
        model=args.model, n_bkps=args.n_bkps, out_dir=args.out_dir, show=args.show)

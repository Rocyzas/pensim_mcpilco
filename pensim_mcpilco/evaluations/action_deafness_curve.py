"""
PYTHONPATH=.. python -m evaluations.action_deafness_curve \
    single_phase=seed4_0 dual_phase=seed3_1 dual_phase=seed4_2

Action deafness as a CURVE over batch time, for any mix of single-phase and dual-phase runs
overlaid on one axis.

Why this exists as its own script rather than as an edit to the six action_sensitivity*.py
variants:

1. Their `action_deafness_report`s sweep the action at a FICTITIOUS reference state --
   action_sensitivity_multi-phase.py at each phase's pooled mean training state,
   action_sensitivity.py at early/mid/late time-tercile means. Both average over states spanning
   tens of hours of batch time, producing a "reference state" no real batch ever passes through.
   Here the reference state at decision j is a REAL trajectory point, s_base[j] from an a=0
   simulator rollout.
2. Each of those scripts writes into its own run's directory, so they structurally cannot put a
   single-phase run, a time-pivot dual-phase run and a biomass-pivot dual-phase run on the same
   axis -- which is the entire point of the comparison. This script loads N runs at once.
3. Nothing here can be recovered from their saved CSVs: action_sensitivity_deafness.csv is
   already collapsed to one row per (phase, GP) with no decision-time axis, and
   action_sensitivity_table.csv's `j` column carries true-vs-model deltas, a different quantity.
   The sweep has to be re-run against the trained GPs in log.pkl.

Nothing in evaluations/ or mcpilco/ is modified by this script; it only reads runs.

Two design decisions do the real work:

REFERENCE STATES COME FROM THE SIMULATOR, NOT FROM TRAINING DATA. The physics don't know how the
learned model is structured, so for a fixed seed every run on the plot is probed at byte-identical
states and any difference between curves is attributable to the model alone. Binning each run's
own GP training inputs by time would instead confound "this model is deafer" with "this run
happened to collect samples elsewhere".

PROBES GO THROUGH THE DEPLOYED COMPOSITE, NOT phase1/phase2. action_sensitivity_multi-phase.py's
report deliberately bypasses the phase router (it is asking about one phase's GP in isolation).
Doing that here would make the curve two flat segments joined at the pivot BY CONSTRUCTION, which
is precisely the artefact this plot exists to avoid. Every probe instead does
reset_step_counter(j, bm_max0=...) then ml.get_next_state(...), so the sigmoid blend window is
measured rather than assumed -- see model_learning_dual_phase.DualPhaseModelLearning.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if _os.path.dirname(_ROOT) not in _sys.path:
    _sys.path.insert(0, _os.path.dirname(_ROOT))

import evaluations.eval_single_phase_lib as single_lib
import evaluations.eval_multi_phase_lib as multi_lib
from mcpilco.pensim_wrapper import (PenSimWrapper, STATE_NAMES, ACTION_DIM,
                                    CONTROL_H, T_SAMPLING, TIME_IDX)
from mcpilco.config_single_phase import get_config as _sp_get_config
from mcpilco.config_single_phase_baseline import get_config as _sp_baseline_get_config
from mcpilco.config_single_phase_baseline_time import get_config as _sp_baseline_time_get_config
from mcpilco.config_dual_phase import get_config as _dp_get_config
from mcpilco.config_dual_phase_baseline import get_config as _dp_baseline_get_config
from mcpilco.config_dual_phase_baseline_time import get_config as _dp_baseline_time_get_config
from mcpilco.config_dual_phase_baseline_priors import get_config as _dp_baseline_priors_get_config

CHANNELS = [c for c in STATE_NAMES if c != "time"]
CHANNEL_IDX = [STATE_NAMES.index(c) for c in CHANNELS]

# setup name -> (eval lib module, get_config fn, default results root). Passing the wrong setup
# for a run rebuilds the wrong architecture -- and because RBF_WtMassBalance/RBF_RecipeMean share
# their exact parameter set with plain RBF, a mismatch can load_state_dict without error and just
# silently reattach a prior mean the checkpoint was never fit against (see
# eval_multi_phase_lib.reconstruct_gp_agent). Hence the setup is required on every run spec rather
# than guessed from the path.
SETUPS = {
    "single_phase":               (single_lib, _sp_get_config,                "single_phase"),
    "single_phase_baseline":      (single_lib, _sp_baseline_get_config,       "single_phase_baseline"),
    "single_phase_baseline_time": (single_lib, _sp_baseline_time_get_config,  "single_phase_baseline_time"),
    "dual_phase":                 (multi_lib,  _dp_get_config,                "dual_phase"),
    "dual_phase_baseline":        (multi_lib,  _dp_baseline_get_config,       "dual_phase_baseline"),
    "dual_phase_baseline_time":   (multi_lib,  _dp_baseline_time_get_config,  "dual_phase_baseline_time"),
    "dual_phase_baseline_priors": (multi_lib,  _dp_baseline_priors_get_config, "dual_phase_baseline_priors"),
}

# Same "spread -> implied SD" conversion the existing action_deafness_report uses: for a monotone
# response swept uniformly over [-1, 1], the implied SD of that response is spread / (2*sqrt(3))
# (the SD of a uniform distribution). Keeping it identical is what makes a point on this curve
# comparable to a row of action_sensitivity_deafness.csv.
SPREAD_TO_SD = 1.0 / (2.0 * math.sqrt(3.0))


def _bm_max0_at(ml, s_base, j):
    """Running-max biomass over the REAL trajectory up to decision j, for seeding a jump.

    Copied verbatim from action_sensitivity_multi-phase.py -- that file's name contains a hyphen,
    so it is not a valid module identifier and cannot be imported (see its own docstring). Under
    --onEachRollout the blend weight is a function of biomass ACCUMULATED SINCE THE START OF THE
    BATCH, so probing decision j via reset_step_counter(j) -- which is what every probe here does,
    deliberately, to avoid replaying the whole batch -- starts with an empty running max and would
    under-weight phase 2 at exactly the late-batch decisions this curve is about. _blend_weight
    raises rather than return that silently wrong weight, so the value has to be supplied here.

    Returns None (and reset_step_counter then behaves as before) whenever the flag is off, so the
    time-sigmoid path and every single-phase run are untouched."""
    if not getattr(ml, "on_each_rollout", False):
        return None
    from mcpilco.model_learning_dual_phase import _bm_from_states
    bm = _bm_from_states(np.asarray(s_base)[:j + 1])
    return torch.tensor([float(np.max(bm))], dtype=ml.dtype, device=ml.device)


def parse_run_spec(spec):
    """'setup=run_id' -> (setup, run_id). The setup half is mandatory; see SETUPS."""
    if "=" not in spec:
        raise ValueError(
            f"run spec {spec!r} must be 'setup=run_id', e.g. 'dual_phase=seed3_1'. "
            f"Known setups: {', '.join(SETUPS)}")
    setup, run_id = spec.split("=", 1)
    if setup not in SETUPS:
        raise ValueError(f"unknown setup {setup!r}; known setups: {', '.join(SETUPS)}")
    return setup, run_id


def load_entry(spec, trial=None, results_root=None):
    """Load one run and everything the sweep needs to know about it."""
    setup, run_id = parse_run_spec(spec)
    lib, get_config_fn, default_root = SETUPS[setup]
    root = Path(_ROOT) / "results" / default_root if results_root is None else Path(results_root)
    run = lib.load_run(run_id, get_config_fn=get_config_fn, results_root=root)
    agent, idx = lib.reconstruct_gp_agent(run, idx=trial, get_config_fn=get_config_fn)
    ml = agent.model_learning
    # Detect the architecture from the model itself rather than from the setup name, so the same
    # code path serves both families and a mislabelled setup can't quietly pick the wrong branch.
    is_dual = hasattr(ml, "phase1") and hasattr(ml, "phase2")
    entry = {
        # Provisional; main() re-labels once every run is loaded, since uniqueness is only
        # decidable across the whole set (see assign_labels).
        "label": f"{setup}/{Path(run.dir).name}",
        "setup": setup,
        "run_dir": run.dir,
        "trial": idx,
        "agent": agent,
        "run": run,
        "is_dual": is_dual,
        "pivot_hours": float(run.pivot_hours) if is_dual else float("nan"),
        "pivot_mode": str(run.pivot_mode) if is_dual else "",
        "blend_half_width_hours": float(run.blend_half_width_hours) if is_dual else float("nan"),
        "on_each_rollout": bool(getattr(ml, "on_each_rollout", False)),
    }
    print(f"[load] {entry['label']}  dir={run.dir}  trial={idx}  "
          f"{'dual' if is_dual else 'single'}-phase"
          + (f"  pivot_hours={entry['pivot_hours']:g} mode={entry['pivot_mode']}"
             f" onEachRollout={entry['on_each_rollout']}" if is_dual else ""))
    return entry


def assign_labels(entries, overrides=None):
    """Short, unique labels for the CSV and the legend.

    Run specs are usually full paths (results/full/ConcCost/multi-phase/No_time/seed4_2), which
    make an unreadable legend. "<setup>/<run dir name>" is almost always enough, but seed names
    repeat across the results tree, so a collision widens EVERY label by one parent directory --
    all of them, not just the clashing pair, so labels stay comparable to each other."""
    if overrides:
        if len(overrides) != len(entries):
            raise ValueError(f"--labels has {len(overrides)} entries for {len(entries)} runs")
        for e, lb in zip(entries, overrides):
            e["label"] = lb
        return entries
    depth = 1
    while True:
        for e in entries:
            parts = Path(e["run_dir"]).parts[-depth:]
            e["label"] = f"{e['setup']}/{'/'.join(parts)}"
        labels = [e["label"] for e in entries]
        if len(set(labels)) == len(labels) or depth >= 6:
            return entries
        depth += 1


def sigma_n_table(entry):
    """Per-channel GP noise SD, as {channel: (sigma_n_phase1, sigma_n_phase2)}.

    sigma_n is the denominator of the deafness ratio and is a per-GP CONSTANT, so it does not move
    along the curve -- but for a dual-phase run the composite blends two GPs, so which one is "the"
    noise floor depends on where in the blend window the probe sits. Both are returned and combined
    per-probe by _sigma_n_at. Single-phase runs report the same value twice, which makes that
    combination a no-op for them."""
    ml = entry["agent"].model_learning
    subs = ([ml.phase1, ml.phase2] if entry["is_dual"] else [ml, ml])
    out = {}
    for c, idx in zip(CHANNELS, CHANNEL_IDX):
        out[c] = tuple(float(torch.sqrt(sub.gp_list[idx].get_sigma_n_2()).detach().cpu())
                       for sub in subs)
    return out


def _sigma_n_at(sn_pair, w):
    """Effective noise floor at blend weight w. Blends the two phases' noise VARIANCES with the
    same w the composite blends its predictions with -- i.e. the first two terms of
    get_next_state's law-of-total-variance mixture -- then takes the root. Blending the SDs
    directly instead would not correspond to any variance the model actually reports."""
    sn1, sn2 = sn_pair
    return math.sqrt((1.0 - w) * sn1 ** 2 + w * sn2 ** 2)


_BASE_CACHE = {}


def rollout_base(seed, base_level=0.0):
    """Real-simulator reference trajectory at a constant action. Cached and shared across every
    run on the plot -- that sharing is what makes the curves comparable (see module docstring),
    and it also keeps the total simulator cost at one rollout per seed regardless of how many
    runs are overlaid."""
    key = (seed, base_level)
    if key not in _BASE_CACHE:
        wrapper = PenSimWrapper(seed_offset=0)
        policy = lambda state, decision_idx: np.array([base_level])
        s_base, _, _ = wrapper.rollout(s0=None, policy=policy, T=CONTROL_H, dt=T_SAMPLING,
                                       noise=None, seed=seed)
        _BASE_CACHE[key] = s_base
        print(f"[rollout] base trajectory seed={seed} a={base_level:g} -> {s_base.shape}")
    return _BASE_CACHE[key]


def blend_weight_at(entry, ref_state, j, bm_max0):
    """The composite's own blend weight at decision j, recorded so the plot can be checked against
    the routing rather than trusting it. Returns 0.0 for single-phase runs (phase1 always, by
    definition of having only one model).

    Reset first: under --onEachRollout _blend_weight accumulates into ml._bm_max, so it must start
    from the same seeded running max the probe itself will use. Calling it here and again inside
    get_next_state is harmless -- the accumulation is a running MAX over the identical value, hence
    idempotent -- but the reset before each is not optional."""
    if not entry["is_dual"]:
        return 0.0
    ml = entry["agent"].model_learning
    ml.reset_step_counter(j, bm_max0=bm_max0)
    with torch.no_grad():
        w = (ml._blend_weight(j, ref_state) if ml.on_each_rollout else ml._blend_weight(j))
    if torch.is_tensor(w):
        # Per-particle under --onEachRollout; every row of the sweep carries the identical state,
        # so these are all the same value and the mean is exact, not an approximation.
        return float(w.detach().cpu().mean())
    return float(w)


def sweep_spread(entry, ref_state, j, bm_max0, a_grid):
    """Peak-to-peak of the one-step predicted delta as the action sweeps [-1, 1], per channel.

    One batched forward pass: the sweep's n_sweep action values become n_sweep rows sharing the
    same state, and a single get_next_state returns every channel at once. The existing
    action_deafness_report instead re-runs the whole sweep once per GP over identical inputs,
    which is STATE_DIM-fold redundant -- affordable for 2 reference states, not for 23 decisions x
    5 seeds x N runs."""
    ml = entry["agent"].model_learning
    n = a_grid.shape[0]
    states = ref_state.repeat(n, 1)
    actions = a_grid.reshape(n, ACTION_DIM)
    with torch.no_grad():
        if entry["is_dual"]:
            ml.reset_step_counter(j, bm_max0=bm_max0)
        next_states, _, _ = ml.get_next_state(current_state=states, current_input=actions,
                                              particle_pred=False)
    deltas = (next_states - states).detach().cpu().numpy()
    return deltas.max(axis=0) - deltas.min(axis=0)


def scan(entries, seeds, j_list, n_sweep=21, base_level=0.0):
    """The whole measurement: every (run, seed, decision) probed at the same reference states."""
    rows = []
    for entry in entries:
        ml = entry["agent"].model_learning
        a_grid = torch.linspace(-1.0, 1.0, n_sweep, dtype=ml.dtype, device=ml.device)
        sn = sigma_n_table(entry)
        print(f"\n--- {entry['label']} ---")
        for seed in seeds:
            s_base = rollout_base(seed, base_level)
            crossover_j = -1
            seed_rows = []
            for j in j_list:
                ref_state = torch.tensor(s_base[j], dtype=ml.dtype,
                                         device=ml.device).unsqueeze(0)
                bm0 = _bm_max0_at(ml, s_base, j)
                w = blend_weight_at(entry, ref_state, j, bm0)
                spreads = sweep_spread(entry, ref_state, j, bm0, a_grid)
                # `time` advances by an exact rule independent of the action (see
                # model_learning_det_time.DETERMINISTIC_CHANNELS), so its spread must be exactly
                # zero. A nonzero value means the action moved the clock, i.e. the probe is not
                # going through the deterministic-channel path at all -- a bug in the query, and
                # every other channel's number on this curve would be suspect too.
                if spreads[TIME_IDX] != 0.0:
                    raise AssertionError(
                        f"{entry['label']} seed={seed} j={j}: sweeping the action moved the "
                        f"`time` channel by {spreads[TIME_IDX]:g}; it must be exactly 0")
                if crossover_j < 0 and entry["is_dual"] and w >= 0.5:
                    crossover_j = j
                for c, idx in zip(CHANNELS, CHANNEL_IDX):
                    spread = float(spreads[idx])
                    implied_sd = spread * SPREAD_TO_SD
                    sigma_n = _sigma_n_at(sn[c], w)
                    seed_rows.append({
                        "run_label": entry["label"], "setup": entry["setup"],
                        "run_dir": str(entry["run_dir"]), "trial": entry["trial"],
                        "seed": seed, "j": j, "hours": j * T_SAMPLING, "channel": c,
                        "spread": spread, "sigma_n": sigma_n, "implied_sd": implied_sd,
                        "sd_over_noise_ratio": (implied_sd / sigma_n if sigma_n
                                                else float("inf")),
                        "blend_weight": w,
                    })
            # Known only after the seed's full sweep, so it is stamped onto every row afterwards
            # rather than carried forward -- under the biomass pivot it differs between seeds,
            # which is exactly what makes a naive 5-seed mean blur the transition. Resolution is
            # the probe grid: it is the first PROBED j with w >= 0.5, so it lands within `every`
            # decisions of the true crossing, not on it.
            for r in seed_rows:
                r["crossover_j"] = crossover_j
            rows.extend(seed_rows)
            print(f"  seed {seed}: crossover_j={crossover_j}"
                  + (f" ({crossover_j * T_SAMPLING:g} h)" if crossover_j >= 0 else " (n/a)"))
    return rows


def summarise(rows):
    """Mean / min / max over seeds, keyed by (run_label, channel, j) -- the plot's line and band."""
    agg = {}
    for r in rows:
        agg.setdefault((r["run_label"], r["channel"], r["j"]), []).append(r["sd_over_noise_ratio"])
    out = []
    for (label, channel, j), vals in sorted(agg.items()):
        v = np.array(vals)
        out.append({"run_label": label, "channel": channel, "j": j, "hours": j * T_SAMPLING,
                    "mean_sd_over_noise_ratio": float(v.mean()),
                    "min_sd_over_noise_ratio": float(v.min()),
                    "max_sd_over_noise_ratio": float(v.max()),
                    "n_seeds": int(v.size)})
    return out


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {path}")


def _mark_transition(ax, entry, colour, crossovers):
    """Where this run's phase transition actually sits on a TIME axis.

    Under the time sigmoid that is pivot_hours with its blend window, so eval_multi_phase_lib's
    own _mark_pivot draws it. Under --onEachRollout it is NOT pivot_hours at all: the blend runs
    on the biomass coordinate, and pivot_hours survives in the config only as the training-split
    fallback. Drawing the clock pivot for such a run would put the marker at 100 h when the model
    actually crosses around 60-70 h -- wrong by two thirds of the blend window, and wrong in the
    direction that makes the curve look mis-timed rather than the marker. So plot the MEASURED
    crossover instead (median over seeds, from the same probes that produced the curve), and say
    so in the label."""
    if not entry["is_dual"]:
        return
    if entry["on_each_rollout"]:
        cj = [c for c in crossovers.get(entry["label"], []) if c >= 0]
        if cj:
            ax.axvline(float(np.median(cj)) * T_SAMPLING, color=colour, ls="-.", lw=1.3,
                       label=f"{entry['label']}: measured crossover (median over seeds)")
        return
    multi_lib._mark_pivot(ax, entry["pivot_hours"],
                          blend_half_width_hours=entry["blend_half_width_hours"],
                          pivot_mode=entry["pivot_mode"])


def plot_curves(summary_rows, entries, out_path, crossovers, base_level=0.0):
    labels = [e["label"] for e in entries]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, c in zip(axes.ravel(), CHANNELS):
        for i, label in enumerate(labels):
            srs = sorted([r for r in summary_rows
                          if r["run_label"] == label and r["channel"] == c],
                         key=lambda r: r["j"])
            if not srs:
                continue
            h = np.array([r["hours"] for r in srs])
            mean = np.array([r["mean_sd_over_noise_ratio"] for r in srs])
            lo = np.array([r["min_sd_over_noise_ratio"] for r in srs])
            hi = np.array([r["max_sd_over_noise_ratio"] for r in srs])
            ax.plot(h, mean, "-o", ms=3, lw=1.4, color=f"C{i}", label=label, zorder=3)
            ax.fill_between(h, lo, hi, color=f"C{i}", alpha=0.15, lw=0, zorder=2)
        # The one threshold that matters: below it the entire action effect is smaller than what
        # the GP itself calls noise, i.e. the model is deaf to the feed rate at that point in the
        # batch no matter what the policy does.
        ax.axhline(1.0, color="k", ls="--", lw=1, zorder=1,
                   label="action effect = noise floor")
        for i, e in enumerate(entries):
            _mark_transition(ax, e, f"C{i}", crossovers)
        ax.set_yscale("log")
        ax.set_title(c)
        ax.set_xlabel("batch time (h)")
        ax.set_ylabel("implied SD / sigma_n")
        ax.grid(alpha=0.3, which="both")
    # One legend for the figure: every panel draws the identical set of lines, and the pivot
    # markers add a duplicate entry per panel otherwise.
    handles, lbls = axes.ravel()[0].get_legend_handles_labels()
    seen, uniq = set(), []
    for hd, lb in zip(handles, lbls):
        if lb not in seen:
            seen.add(lb)
            uniq.append((hd, lb))
    fig.legend([h for h, _ in uniq], [l for _, l in uniq], fontsize=8,
               loc="lower center", ncol=min(4, len(uniq)))
    fig.suptitle(f"action deafness vs batch time (reference states from a={base_level:g} "
                 f"simulator rollouts, shared across runs)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main(specs, trial=None, every=2, n_seeds=5, seed_base=424242, n_sweep=21, base_level=0.0,
         out_dir=None, results_root=None, labels=None):
    entries = [load_entry(s, trial=trial, results_root=results_root) for s in specs]
    assign_labels(entries, overrides=labels)
    seeds = [seed_base + i for i in range(n_seeds)]

    n_decisions = int(CONTROL_H / T_SAMPLING)
    j_list = list(range(0, n_decisions, every))
    print(f"\nn_decisions={n_decisions}  probing every {every} decisions "
          f"({every * T_SAMPLING:g} h) -> {len(j_list)} points, j={j_list[0]}..{j_list[-1]}")
    print(f"seeds={seeds}  n_sweep={n_sweep}  base_level={base_level:g}")

    rows = scan(entries, seeds, j_list, n_sweep=n_sweep, base_level=base_level)
    summary_rows = summarise(rows)
    crossovers = {}
    for r in rows:
        crossovers.setdefault(r["run_label"], {})[r["seed"]] = r["crossover_j"]
    crossovers = {k: list(v.values()) for k, v in crossovers.items()}

    out_dir = Path(entries[0]["run_dir"]) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, out_dir / "action_deafness_curve.csv",
             ["run_label", "setup", "run_dir", "trial", "seed", "j", "hours", "channel",
              "spread", "sigma_n", "implied_sd", "sd_over_noise_ratio", "blend_weight",
              "crossover_j"])
    save_csv(summary_rows, out_dir / "action_deafness_curve_summary.csv",
             ["run_label", "channel", "j", "hours", "mean_sd_over_noise_ratio",
              "min_sd_over_noise_ratio", "max_sd_over_noise_ratio", "n_seeds"])
    plot_curves(summary_rows, entries, out_dir / "action_deafness_curve.png",
                crossovers, base_level=base_level)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Action deafness vs batch time, overlaid across runs.",
        epilog="example: single_phase=seed4_0 dual_phase=seed3_1 dual_phase=seed4_2")
    p.add_argument("runs", nargs="+", metavar="SETUP=RUN_ID",
                   help=f"one or more runs to overlay; SETUP is one of: {', '.join(SETUPS)}")
    p.add_argument("--trial", type=int, default=None,
                   help="which trial's GP model to probe (default: last saved), applied to every run")
    p.add_argument("--every", type=int, default=2,
                   help="probe every N decisions (default 2 = every 10 h at T_SAMPLING=5)")
    p.add_argument("--n_seeds", type=int, default=5, help="simulator seeds for reference states")
    p.add_argument("--seed_base", type=int, default=424242,
                   help="first simulator seed (matches action_sensitivity*.py's SIM_SEEDS)")
    p.add_argument("--n_sweep", type=int, default=21, help="action grid points over [-1, 1]")
    p.add_argument("--base_level", type=float, default=0.0,
                   help="constant action driving the reference rollouts; 0.0 is on-manifold, "
                        "0.6 reproduces the off-manifold probe")
    p.add_argument("--out_dir", type=str, default=None,
                   help="where to write the CSV/PNG (default: the first run's own directory)")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root for every run spec")
    p.add_argument("--labels", nargs="+", default=None,
                   help="legend/CSV label per run, in the same order as the run specs "
                        "(default: <setup>/<run dir name>, widened if that collides)")
    args = p.parse_args()
    main(args.runs, trial=args.trial, every=args.every, n_seeds=args.n_seeds,
         seed_base=args.seed_base, n_sweep=args.n_sweep, base_level=args.base_level,
         out_dir=args.out_dir, results_root=args.results_root, labels=args.labels)

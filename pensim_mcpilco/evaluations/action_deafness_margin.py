"""
PYTHONPATH=.. python -m evaluations.action_deafness_margin \
    single_phase_baseline_time=seed4_1 single_phase_baseline=seed4_0 \
    --every_hours 20

PYTHONPATH=.. python -m evaluations.action_deafness_margin \
    --group_time single_phase_baseline_time=seed4_1 single_phase_baseline_time=seed5_1 \
                 single_phase_baseline_time=seed6_1 \
    --group_no_time single_phase_baseline=seed4_0 single_phase_baseline=seed5_0 \
                    single_phase_baseline=seed6_0 \
    --every_hours 20
(diff mode: plots with_time-minus-no_time margin, paired by position, averaged across pairs)

Action deafness as a SIGNAL-MINUS-NOISE MARGIN over batch time, one plot with every state
channel overlaid (differently coloured), for one or more runs (e.g. a with-time vs without-time
pair on the same axis, distinguished by opacity).

Reuses action_deafness_curve.py's sweep machinery UNCHANGED (load_entry, scan, SETUPS,
parse_run_spec) -- that script already does the hard part correctly: real-simulator reference
states shared across every run (see its own module docstring for why that matters), single- and
dual-phase support, seed-averaged sweeps. Nothing here re-derives spread/sigma_n/implied_sd; it
only adds a different quantity on top and a different plot.

Two things this script does differently from action_deafness_curve.py:

1. MARGIN, NOT RATIO. action_deafness_curve.py plots implied_sd / sigma_n on a log axis (the
   threshold "1.0" is a squint-and-compare line on a log scale). Here the plotted quantity is
   implied_sd - sigma_n -- same units (both SDs), so 0 IS the deaf/not-deaf boundary in the most
   literal sense: a channel's line sitting above 0 means the swept action's effect there is
   bigger than the model's own noise floor (not deaf); below 0 means the effect is smaller than
   what the model already calls noise (deaf), by that many normalised-state units.

2. ONE PANEL, CHANNEL = COLOUR, RUN = OPACITY. action_deafness_curve.py facets by channel (one
   subplot each) and colours by run. Here every non-time channel is one line on ONE shared axis,
   in a FIXED colour regardless of which run or how many are plotted, and which run a line
   belongs to is instead read off its opacity: alpha=1.0 for any run whose setup name ends in
   "_time" (time kept as a GP input), a fixed lower alpha otherwise. Built for exactly the
   with-time vs without-time overlay this module's usage line shows, where the question is "does
   adding time change the deafness margin for the SAME channel", not "how do channels compare
   within one run".

3. DIFF MODE (--group_time/--group_no_time). Answers "what does adding time change about the
deafness margin", not "what is the margin". Give N with-time runs and N no-time runs; they are
paired by POSITION (group_time[i] vs group_no_time[i] -- e.g. matching seed4/seed5/seed6 across
the two groups), never by label matching, so list them in corresponding order. For each pair, at
every (channel, hour), diff = mean_signal_minus_noise(with_time) - mean_signal_minus_noise(
no_time) -- valid pointwise because scan()'s reference states are the same real-simulator rollout
for every run regardless of group (shared _BASE_CACHE keyed by (seed, base_level) only), so a
diff isolates the effect of the time input, not a change in what state it was evaluated at. The
per-pair diffs are then averaged across pairs (mean line + min/max band per channel) -- a single
pair confounds "effect of time" with "idiosyncrasy of that one training run"; more pairs makes
the averaged diff trustworthy rather than anecdotal. 0 means time made no difference to the
margin at that point; positive means adding time moved the channel further from deaf (more
signal above noise); negative means it moved further into deaf. Writes
action_deafness_margin_diff_raw.csv (one row per pair/channel/hour) and
action_deafness_margin_diff.csv (mean/min/max across pairs), plots
action_deafness_margin_diff.png. Positional `runs` and --group_time/--group_no_time are mutually
exclusive -- diff mode has only one line per channel (already subtracted), so run-opacity doesn't
apply.

--csv: skip the sweep entirely and just re-plot an already-saved
action_deafness_margin.csv (or, in diff mode, action_deafness_margin_diff.csv) -- looked up the
same way the out_dir would be resolved when computing one -- no GP reconstruction, no simulator
rollouts, so this is the cheap path once the numbers already exist on disk. Only the summary CSV
is read back; the raw per-seed/per-pair CSV is written whenever computing but is not needed to
redraw the plot.
"""
import argparse
import csv as csv_module
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if _os.path.dirname(_ROOT) not in _sys.path:
    _sys.path.insert(0, _os.path.dirname(_ROOT))

import matplotlib.pyplot as plt

import evaluations.action_deafness_curve as adc

# Fixed colour per channel, independent of run order/count -- so the same channel is always the
# same colour whether one run or five are overlaid. CHANNELS already excludes `time` (see
# action_deafness_curve.py: CHANNELS = [c for c in STATE_NAMES if c != "time"]).
CHANNEL_COLORS = {c: f"C{i}" for i, c in enumerate(adc.CHANNELS)}

TIME_VARIANT_ALPHA = 1.0
NO_TIME_ALPHA = 0.45


def has_time(setup):
    """Whether this setup keeps `time` as an active GP input regressor -- the naming convention
    is consistent across every *_baseline/*_baseline_time pair in this codebase (see e.g.
    action_sensitivity_baseline.py's own SETUPS dict), so a suffix check is exact, not a guess."""
    return setup.endswith("_time")


def compute(specs, trial=None, every_hours=20, n_seeds=5, seed_base=424242, n_sweep=21,
           base_level=0.0, out_dir=None, results_root=None, labels=None):
    """Runs action_deafness_curve.py's sweep, then reduces it to the margin summary this script
    plots. Returns (summary_rows, entries, out_dir)."""
    entries = [adc.load_entry(s, trial=trial, results_root=results_root) for s in specs]
    adc.assign_labels(entries, overrides=labels)

    every = max(1, round(every_hours / adc.T_SAMPLING))
    actual_hours = every * adc.T_SAMPLING
    if abs(actual_hours - every_hours) > 1e-9:
        print(f"[note] --every_hours {every_hours:g} is not a multiple of T_SAMPLING="
             f"{adc.T_SAMPLING:g}h; probing every {every} decisions = {actual_hours:g}h instead")
    seeds = [seed_base + i for i in range(n_seeds)]
    n_decisions = int(adc.CONTROL_H / adc.T_SAMPLING)
    j_list = list(range(0, n_decisions, every))
    print(f"\nn_decisions={n_decisions}  probing every {every} decisions ({actual_hours:g} h) "
         f"-> {len(j_list)} points, j={j_list[0]}..{j_list[-1]}")
    print(f"seeds={seeds}  n_sweep={n_sweep}  base_level={base_level:g}")

    raw_rows = adc.scan(entries, seeds, j_list, n_sweep=n_sweep, base_level=base_level)
    for r in raw_rows:
        r["signal_minus_noise"] = r["implied_sd"] - r["sigma_n"]

    summary = {}
    for r in raw_rows:
        key = (r["run_label"], r["setup"], r["channel"], r["j"])
        summary.setdefault(key, []).append(r["signal_minus_noise"])
    summary_rows = []
    for (label, setup, channel, j), vals in sorted(summary.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        summary_rows.append({
            "run_label": label, "setup": setup, "channel": channel, "j": j,
            "hours": j * adc.T_SAMPLING,
            "mean_signal_minus_noise": sum(vals_sorted) / n,
            "min_signal_minus_noise": vals_sorted[0],
            "max_signal_minus_noise": vals_sorted[-1],
            "n_seeds": n,
        })

    out_dir = Path(entries[0]["run_dir"]) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adc.save_csv(raw_rows, out_dir / "action_deafness_margin_raw.csv",
                ["run_label", "setup", "run_dir", "trial", "seed", "j", "hours", "channel",
                 "spread", "sigma_n", "implied_sd", "signal_minus_noise", "sd_over_noise_ratio",
                 "blend_weight", "crossover_j"])
    adc.save_csv(summary_rows, out_dir / "action_deafness_margin.csv",
                ["run_label", "setup", "channel", "j", "hours", "mean_signal_minus_noise",
                 "min_signal_minus_noise", "max_signal_minus_noise", "n_seeds"])
    return summary_rows, entries, out_dir


def resolve_out_dir_for_replot(specs, results_root=None):
    """Where compute() would have written the CSV, WITHOUT loading any run (no GP
    reconstruction, no simulator) -- the whole point of --csv is to skip that cost. Matches
    compute()'s own default (out_dir=None -> the first run's own directory) using only string
    resolution: parse_run_spec + SETUPS' default results root is enough to name the directory a
    run lives in without opening its log.pkl."""
    setup, run_id = adc.parse_run_spec(specs[0])
    _, _, default_root = adc.SETUPS[setup]
    root = Path(_ROOT) / "results" / default_root if results_root is None else Path(results_root)
    run_dir = Path(run_id)
    return run_dir if run_dir.is_dir() else root / run_id


def load_summary_csv(path):
    with open(path, newline="") as f:
        rows = list(csv_module.DictReader(f))
    for r in rows:
        r["j"] = int(r["j"])
        r["hours"] = float(r["hours"])
        r["mean_signal_minus_noise"] = float(r["mean_signal_minus_noise"])
        r["min_signal_minus_noise"] = float(r["min_signal_minus_noise"])
        r["max_signal_minus_noise"] = float(r["max_signal_minus_noise"])
        r["n_seeds"] = int(r["n_seeds"])
    return rows


def plot_margin(summary_rows, out_path, title=None):
    """One panel: every (run, channel) is a line, coloured by channel (CHANNEL_COLORS, fixed
    regardless of run) and shaded by run via alpha (1.0 if the run's setup keeps time, a lower
    fixed alpha otherwise -- see has_time). 0 is drawn explicitly as the deaf/not-deaf boundary:
    implied signal SD above the model's own noise SD is "not deaf", below is "deaf"."""
    labels = sorted({(r["run_label"], r["setup"]) for r in summary_rows})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run_label, setup in labels:
        alpha = TIME_VARIANT_ALPHA if has_time(setup) else NO_TIME_ALPHA
        for channel in adc.CHANNELS:
            srs = sorted([r for r in summary_rows
                          if r["run_label"] == run_label and r["channel"] == channel],
                         key=lambda r: r["j"])
            if not srs:
                continue
            h = [r["hours"] for r in srs]
            m = [r["mean_signal_minus_noise"] for r in srs]
            ax.plot(h, m, "-o", ms=3, lw=1.6, color=CHANNEL_COLORS[channel], alpha=alpha,
                    label=f"{channel} ({run_label})")
    ax.axhline(0.0, color="k", ls="--", lw=1.2, zorder=1, label="signal = noise")
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("implied signal SD - noise SD  (normalised state units)")
    ax.set_title(title or "Action deafness margin vs batch time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def compute_diff(group_time_specs, group_no_time_specs, trial=None, every_hours=20, n_seeds=5,
                 seed_base=424242, n_sweep=21, base_level=0.0, out_dir=None, results_root=None,
                 pair_labels=None):
    """Runs compute() once over both groups combined (so the sweep is only done once per run,
    even though every run also feeds a diff), then subtracts group_time[i] - group_no_time[i]
    pointwise per (channel, hour), pairing by POSITION. Returns (diff_summary_rows, out_dir)."""
    if len(group_time_specs) != len(group_no_time_specs):
        raise ValueError(
            f"--group_time has {len(group_time_specs)} runs but --group_no_time has "
            f"{len(group_no_time_specs)}; they must be paired 1:1 in matching order (e.g. "
            f"seed4_1,seed5_1,seed6_1 vs seed4_0,seed5_0,seed6_0)")
    n_pairs = len(group_time_specs)
    if pair_labels is None:
        pair_labels = [f"pair{i}" for i in range(n_pairs)]
    print("pairing (by position -- verify this matches what you intended):")
    for i, (t, nt) in enumerate(zip(group_time_specs, group_no_time_specs)):
        print(f"  {pair_labels[i]}: {t}  -  {nt}")

    # Internal-only labels so the two groups can't collide in compute()'s summary keying, even
    # if e.g. the same run_id string happens to appear in both groups under different setups.
    time_labels = [f"__time{i}" for i in range(n_pairs)]
    no_time_labels = [f"__notime{i}" for i in range(n_pairs)]
    all_specs = list(group_time_specs) + list(group_no_time_specs)
    all_labels = time_labels + no_time_labels

    summary_rows, entries, resolved_out_dir = compute(
        all_specs, trial=trial, every_hours=every_hours, n_seeds=n_seeds, seed_base=seed_base,
        n_sweep=n_sweep, base_level=base_level, out_dir=out_dir, results_root=results_root,
        labels=all_labels)

    by_label = {}
    for r in summary_rows:
        by_label.setdefault(r["run_label"], {})[(r["channel"], r["j"])] = r

    diff_rows = []
    for i in range(n_pairs):
        t_rows = by_label[time_labels[i]]
        nt_rows = by_label[no_time_labels[i]]
        keys = sorted(set(t_rows) & set(nt_rows))
        missing = set(t_rows) ^ set(nt_rows)
        if missing:
            print(f"[warn] {pair_labels[i]}: {len(missing)} (channel, hour) points present in "
                 f"only one of the two runs, skipped for this pair")
        for channel, j in keys:
            tr, nr = t_rows[(channel, j)], nt_rows[(channel, j)]
            diff_rows.append({
                "pair": pair_labels[i], "time_spec": group_time_specs[i],
                "no_time_spec": group_no_time_specs[i], "channel": channel, "j": j,
                "hours": tr["hours"],
                "diff_signal_minus_noise": (tr["mean_signal_minus_noise"]
                                            - nr["mean_signal_minus_noise"]),
            })

    agg = {}
    for r in diff_rows:
        agg.setdefault((r["channel"], r["j"], r["hours"]), []).append(r["diff_signal_minus_noise"])
    diff_summary_rows = []
    for (channel, j, hours), vals in sorted(agg.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        diff_summary_rows.append({
            "channel": channel, "j": j, "hours": hours,
            "mean_diff": sum(vals_sorted) / n,
            "min_diff": vals_sorted[0],
            "max_diff": vals_sorted[-1],
            "n_pairs": n,
        })

    adc.save_csv(diff_rows, resolved_out_dir / "action_deafness_margin_diff_raw.csv",
                ["pair", "time_spec", "no_time_spec", "channel", "j", "hours",
                 "diff_signal_minus_noise"])
    adc.save_csv(diff_summary_rows, resolved_out_dir / "action_deafness_margin_diff.csv",
                ["channel", "j", "hours", "mean_diff", "min_diff", "max_diff", "n_pairs"])
    return diff_summary_rows, resolved_out_dir


def load_diff_summary_csv(path):
    with open(path, newline="") as f:
        rows = list(csv_module.DictReader(f))
    for r in rows:
        r["j"] = int(r["j"])
        r["hours"] = float(r["hours"])
        r["mean_diff"] = float(r["mean_diff"])
        r["min_diff"] = float(r["min_diff"])
        r["max_diff"] = float(r["max_diff"])
        r["n_pairs"] = int(r["n_pairs"])
    return rows


def plot_margin_diff(diff_summary_rows, out_path, title=None, n_pairs=None):
    """One panel, channel = colour (same CHANNEL_COLORS as plot_margin), mean diff line + a
    min/max-across-pairs shaded band. No opacity encoding needed -- there's exactly one line per
    channel once with_time - no_time has already been subtracted."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for channel in adc.CHANNELS:
        srs = sorted([r for r in diff_summary_rows if r["channel"] == channel],
                     key=lambda r: r["j"])
        if not srs:
            continue
        h = [r["hours"] for r in srs]
        m = [r["mean_diff"] for r in srs]
        lo = [r["min_diff"] for r in srs]
        hi = [r["max_diff"] for r in srs]
        color = CHANNEL_COLORS[channel]
        ax.plot(h, m, "-o", ms=3, lw=1.8, color=color, label=channel)
        ax.fill_between(h, lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(0.0, color="k", ls="--", lw=1.2, zorder=1, label="no effect of time")
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("signal-minus-noise margin: with_time - no_time  (normalised state units)")
    note = f" (n={n_pairs} seed pairs)" if n_pairs else ""
    ax.set_title(title or f"Effect of time on action deafness margin{note}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def resolve_out_dir_for_diff_replot(group_time_specs, out_dir=None, results_root=None):
    if out_dir is not None:
        return Path(out_dir)
    return resolve_out_dir_for_replot(group_time_specs, results_root=results_root)


def main_diff(group_time_specs, group_no_time_specs, trial=None, every_hours=20, n_seeds=5,
             seed_base=424242, n_sweep=21, base_level=0.0, out_dir=None, results_root=None,
             pair_labels=None, title=None, use_csv=False):
    if use_csv:
        resolved_out_dir = resolve_out_dir_for_diff_replot(
            group_time_specs, out_dir=out_dir, results_root=results_root)
        csv_path = resolved_out_dir / "action_deafness_margin_diff.csv"
        print(f"[--csv] reading {csv_path} (no GP reconstruction, no simulator rollouts)")
        diff_summary_rows = load_diff_summary_csv(csv_path)
        n_pairs = max((r["n_pairs"] for r in diff_summary_rows), default=None)
        plot_margin_diff(diff_summary_rows, resolved_out_dir / "action_deafness_margin_diff.png",
                         title=title, n_pairs=n_pairs)
        return

    diff_summary_rows, resolved_out_dir = compute_diff(
        group_time_specs, group_no_time_specs, trial=trial, every_hours=every_hours,
        n_seeds=n_seeds, seed_base=seed_base, n_sweep=n_sweep, base_level=base_level,
        out_dir=out_dir, results_root=results_root, pair_labels=pair_labels)
    plot_margin_diff(diff_summary_rows, resolved_out_dir / "action_deafness_margin_diff.png",
                     title=title, n_pairs=len(group_time_specs))


def main(specs, trial=None, every_hours=20, n_seeds=5, seed_base=424242, n_sweep=21,
        base_level=0.0, out_dir=None, results_root=None, labels=None, title=None,
        use_csv=False):
    if use_csv:
        resolved_out_dir = Path(out_dir) if out_dir is not None else resolve_out_dir_for_replot(
            specs, results_root=results_root)
        csv_path = resolved_out_dir / "action_deafness_margin.csv"
        print(f"[--csv] reading {csv_path} (no GP reconstruction, no simulator rollouts)")
        summary_rows = load_summary_csv(csv_path)
        plot_margin(summary_rows, resolved_out_dir / "action_deafness_margin.png", title=title)
        return

    summary_rows, _entries, resolved_out_dir = compute(
        specs, trial=trial, every_hours=every_hours, n_seeds=n_seeds, seed_base=seed_base,
        n_sweep=n_sweep, base_level=base_level, out_dir=out_dir, results_root=results_root,
        labels=labels)
    plot_margin(summary_rows, resolved_out_dir / "action_deafness_margin.png", title=title)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Action deafness signal-minus-noise margin vs batch time, one panel, "
                    "channel=colour, run=opacity.",
        epilog="example: single_phase_baseline_time=seed4_1 single_phase_baseline=seed4_0 "
              "--every_hours 20")
    p.add_argument("runs", nargs="*", metavar="SETUP=RUN_ID",
                   help=f"one or more runs to overlay; SETUP is one of: {', '.join(adc.SETUPS)}. "
                        f"Omit and use --group_time/--group_no_time instead for diff mode.")
    p.add_argument("--group_time", nargs="+", metavar="SETUP=RUN_ID", default=None,
                   help="diff mode: with-time runs, e.g. single_phase_baseline_time=seed4_1 "
                        "single_phase_baseline_time=seed5_1 single_phase_baseline_time=seed6_1 "
                        "-- paired by POSITION with --group_no_time (same length, matching seed "
                        "order). Plots mean(with_time margin - no_time margin) per channel, "
                        "averaged across pairs, with a min/max band. Requires --group_no_time; "
                        "mutually exclusive with the positional runs.")
    p.add_argument("--group_no_time", nargs="+", metavar="SETUP=RUN_ID", default=None,
                   help="diff mode: no-time runs, paired by position with --group_time (see its "
                        "help) -- must be the same length and same seed order")
    p.add_argument("--pair_labels", nargs="+", default=None,
                   help="diff mode: name for each pair, same order as --group_time/"
                        "--group_no_time (default: pair0, pair1, ...)")
    p.add_argument("--trial", type=int, default=None,
                   help="which trial's GP model to probe (default: last saved), applied to every run")
    p.add_argument("--every_hours", type=float, default=20.0,
                   help="probe every N hours of batch time (default 20; rounded to the nearest "
                        "multiple of T_SAMPLING)")
    p.add_argument("--n_seeds", type=int, default=5, help="simulator seeds for reference states")
    p.add_argument("--seed_base", type=int, default=424242,
                   help="first simulator seed (matches action_sensitivity*.py's SIM_SEEDS)")
    p.add_argument("--n_sweep", type=int, default=21, help="action grid points over [-1, 1]")
    p.add_argument("--base_level", type=float, default=0.0,
                   help="constant action driving the reference rollouts; 0.0 is on-manifold")
    p.add_argument("--out_dir", type=str, default=None,
                   help="where to read/write the CSV/PNG (default: the first run's own directory)")
    p.add_argument("--results_root", type=str, default=None,
                   help="override the results root for every run spec")
    p.add_argument("--labels", nargs="+", default=None,
                   help="legend/CSV label per run, in the same order as the run specs "
                        "(default: <setup>/<run dir name>, widened if that collides)")
    p.add_argument("--title", type=str, default=None, help="plot title")
    p.add_argument("--csv", dest="use_csv", action="store_true",
                   help="skip the sweep entirely and re-plot the already-saved "
                        "action_deafness_margin.csv (no GP reconstruction, no simulator "
                        "rollouts) -- use once the numbers already exist on disk")
    args = p.parse_args()

    diff_mode = args.group_time is not None or args.group_no_time is not None
    if diff_mode:
        if args.group_time is None or args.group_no_time is None:
            p.error("--group_time and --group_no_time must both be given together")
        if len(args.group_time) != len(args.group_no_time):
            p.error(f"--group_time has {len(args.group_time)} runs but --group_no_time has "
                    f"{len(args.group_no_time)}; they must be paired 1:1, same order")
        if args.runs:
            p.error("positional runs are ignored in diff mode; use --group_time/"
                    "--group_no_time only")
        if args.labels is not None:
            p.error("--labels is not used in diff mode; use --pair_labels instead")
        main_diff(args.group_time, args.group_no_time, trial=args.trial,
                 every_hours=args.every_hours, n_seeds=args.n_seeds, seed_base=args.seed_base,
                 n_sweep=args.n_sweep, base_level=args.base_level, out_dir=args.out_dir,
                 results_root=args.results_root, pair_labels=args.pair_labels, title=args.title,
                 use_csv=args.use_csv)
    else:
        if not args.runs:
            p.error("provide runs (SETUP=RUN_ID ...) or use --group_time/--group_no_time")
        main(args.runs, trial=args.trial, every_hours=args.every_hours, n_seeds=args.n_seeds,
            seed_base=args.seed_base, n_sweep=args.n_sweep, base_level=args.base_level,
            out_dir=args.out_dir, results_root=args.results_root, labels=args.labels,
            title=args.title, use_csv=args.use_csv)

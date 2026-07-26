"""What is each cost term actually worth? -- run this after ANY change to the cost function.

Why this exists
---------------
The cost weights (p_weight, soft_penalty, visc_penalty, rate_penalty) are round numbers that were
never calibrated against anything. This script converts every term into ONE readable currency --
kilograms of penicillin -- so the weights stop being arbitrary:

    reward = p_weight * dmass_kg        =>  1 cost unit == 1 / p_weight kg

So a penalty totalling X cost units over a batch is worth X / p_weight kg of product. That turns
"what should visc_penalty be?" into "how many kg is avoiding a thick batch worth?", which is a
process question with a measurable answer.

How to read the output
----------------------
For a GOOD batch (recipe) and a BAD one (a collapsed episode), each term is summed over the batch:

  * a term reading ~0.00 on BOTH batches is INERT -- it never fires and tunes nothing;
  * a term larger than the reward IS the objective -- the optimiser is solving that, not yield;
  * the reward row should match batch_yield_kg (the number every baseline is scored on). If it
    does not, the thing being optimised is not the thing being reported. That check is what
    caught the missing discharge-harvest credit (reward was 23% low).

The terms come from PeniConcentrationCost._terms, i.e. the SAME code the optimiser runs -- not a
re-implementation, which would drift the moment a formula changed.

FOUR SCENARIOS, NOT TWO -- WHY
-------------------------------
A synthetic sustained-overfeed probe only proves the viscosity penalty fires on a cartoonish
disaster (183 cP); it says nothing about the weight-overflow penalty (never triggered by any
synthetic probe here -- the action space tops out before Wt reaches WT_SOFT) or about whether the
penalty correctly handles a BORDERLINE case: a real, otherwise-attractive policy that merely grazes
a limit rather than blowing through it. Unless disabled with --no_scan, this script therefore scans
`--scan_dir` (default results/single_phase, i.e. real logged training episodes) for two additional
rows, selected as the HIGHEST-YIELD real episode matching each pattern (so they represent policies
that otherwise look good, not flukes):
  * "overfill": a real episode whose max Wt exceeds WT_SOFT_HI, isolated from viscosity where
    possible, so the weight penalty is exercised on a genuine violation instead of asserted inert.
  * "borderline": a real episode whose peak viscosity sits near VISC_MAX (not a collapse) -- this is
    the scenario that actually tests over-conservatism: does the penalty discourage a batch that
    merely brushes the limit but otherwise performs well?
If your local results/single_phase has no run logs, the scan degrades gracefully (prints a warning,
falls back to the synthetic-only scenarios) -- the report still runs, just with a weaker defence.

NET_KG AND THE RECOMMENDED constraint_strength
------------------------------------------------
The overfill/borderline rows are real, otherwise-good policies -- they should NOT simply "lose less
yield than the collapse", they can legitimately have HIGHER yield_kg than the recipe. So the
viscosity SCALE check below (penalty vs. yield LOST relative to recipe) doesn't apply to them: a
policy that grazes the limit while still producing more product isn't a loss to compare against.
The right question for those rows is different: after subtracting the penalty, does the policy
still look BETTER than the recipe, or has the penalty erased its advantage? That is `net_kg =
reward_kg - total_penalty_kg`, computed per row per constraint_strength. The recommendation at the
end of the sweep picks the SMALLEST swept constraint_strength where, simultaneously: (a) every
CATASTROPHIC row's net_kg drops below the recipe's (the constraint successfully discourages
collapse) and (b) every OVERFILL/BORDERLINE row's net_kg stays >= the recipe's (the constraint does
NOT punish a policy for merely touching a limit while still winning). This operationalises the
r - lambda*c ratio directly: too low and (a) fails (collapse isn't discouraged); too high and (b)
fails (the agent is punished into needless conservatism, per the ratio's stated failure mode).

CURRENCY: every penalty is now p_weight * lambda_term * (constraint_strength if it's a constraint)
* kg_violation, where kg_violation = severity_ramp(0..1, soft threshold -> hard limit) *
mass-currently-at-risk. A lambda of 1 means "trade 1 kg of yield to avoid a full-severity violation
of that constraint" -- see mcpilco/penicillin_cost.py's module/class docstrings for the derivation.
`--constraint_strength` is the single global knob that scales the weight- and viscosity-constraint
lambdas TOGETHER (not action_rate, a smoothness preference, and not directly exercisable here --
see the risk-term note below). The `--sweep` section runs this whole report across several
constraint_strength values, which is the calibration defence: it shows the reward/penalty trade-off
the knob actually controls, on both the good and bad batch, rather than asserting a single number.

RISK TERM NOT EXERCISED HERE: risk_weight penalises the across-PARTICLE spread of the summed
trajectory cost (see PeniConcentrationCost.forward), computed from many imagined rollouts of the
SAME seed. This script replays one real trajectory per scenario (one particle), so that spread is
undefined here (std of a single sample) -- `--risk_weight` is accepted and passed through for
completeness but changes nothing this script prints. Check it via the training diagnostics
(std_cost_trial_list / R.1b in evaluations/Rollouts.ipynb) instead.

IMPORTANT CAVEAT
----------------
These numbers are what each term is worth on a REAL trajectory. During training the cost is applied
to GP-PREDICTED states, so a term can be small here and dominant in the optimiser if the model
mis-predicts that channel. Viscosity is the live example: its multi-step calibration ratio is ~4.5x
(see evaluations/evaluate_GPs.ipynb G.7), so a weight calibrated here is applied to a channel the
model over-predicts. Calibrate here, then check the channel's k-step accuracy before trusting it.
The lambdas this script recommends are a STARTING POINT that must be re-checked against
predicted-state rollouts, not a finished calibration.

Usage
-----
    python experiments/cost_term_report.py
    python experiments/cost_term_report.py --bad_run results/single_phase/seed11_0 --bad_episode 6
    python experiments/cost_term_report.py --sweep 0 0.25 0.5 1 2 4 8
    python experiments/cost_term_report.py --no_sweep   # single constraint_strength, no defence table
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, STATE_DIM,
                                    VISC_MAX, WT_SOFT, WT_OVERFLOW, batch_yield_kg)
from mcpilco.penicillin_cost import  VISC_SOFT_SCALE, VISC_SOFT_START
from mcpilco.penicillin_cost import PeniMassChangeCost, PeniConcentrationDenseCost

TERM_ORDER = ["reward", "soft", "visc_soft", "action_rate"]
OUT_DIR = Path(_ROOT) / "results" / "cost_calibration"

# `varying` records whether the row's action actually changes over the batch. action_rate is a
# squared first difference, so it is structurally zero on constant-action rows and any verdict
# drawn from them would describe the probe, not the weight.
Row = namedtuple("Row", "label terms yield_kg peak_visc max_wt varying")

# A rolled real trajectory, kept separate from any cost object so the (expensive) simulator rollout
# happens ONCE and the (cheap) cost re-evaluation happens once per constraint_strength in the sweep.
# `kind` drives the recommendation logic in `recommend_constraint_strength`:
#   "good"        -- the reference row (recipe) everything else is compared against.
#   "catastrophic"-- should become WORSE than "good" as constraint_strength rises (collapse).
#   "overfill"    -- real episode that breaches the weight limit; should stay >= "good" (not
#                    over-penalised) until constraint_strength is clearly too high.
#   "borderline"  -- real episode that grazes the viscosity limit; same requirement as "overfill".
#   "chatter"     -- the max-chatter action_rate probe; not part of the net_kg recommendation
#                    (action_rate isn't scaled by constraint_strength at all).
Scenario = namedtuple("Scenario", "label states inputs mon varying kind")


def evaluate(cost, states, inputs, drop_first=False):
    """Sum every cost term over one real batch. states/inputs are the [T, dim] arrays rollout
    returns, so this scores the trajectory the simulator actually produced (one 'particle').

    `drop_first` exists because rollout only writes states[0] inside its `not pid_baseline`
    branch (pensim_wrapper.py:251-255). A recipe row therefore carries states[0] == zeros, which
    is NOT a null state: normalised zero decodes to the midpoint of every range -- Wt ~80.6 kL,
    Viscosity 100 cP, time 115 h. That pollutes the telescoping dmass[1] = mass[1] - mass[0] and
    makes the harvest term credit decision index 23 on step 0. Small (~0.5 kg) but it lands
    squarely on this script's headline reward-vs-batch_yield_kg check, so drop the row.
    """
    st = torch.tensor(np.asarray(states), dtype=torch.float64).unsqueeze(1)
    ins = torch.tensor(np.asarray(inputs), dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        terms = cost._terms(st, ins)
    sl = slice(1, None) if drop_first else slice(None)
    return {k: float(v[sl].sum()) for k, v in terms.items()}


def roll(wrapper, seed, policy=None, pid_baseline=False):
    states, inputs, _ = wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0,
                                        seed=seed, pid_baseline=pid_baseline)
    return states, inputs, wrapper.monitor[-1]


def _candidate_is_trustworthy(run_dir, episode, yield_kg, required_nonzero_term=None, tol=0.2):
    """Before trusting an auto-picked historical log, check it decodes sanely under TODAY'S state
    encoding and pipeline. This codebase's state space AND its normalisation ranges have been
    revised repeatedly (STATE_DIM/STATE_NAMES/STATE_RANGES, see the many CHANGED_THIS markers in
    pensim_wrapper.py) -- an OLDER run's log.pkl can carry a different number of state channels in
    a different order, so indexing it with today's P_IDX/WT_IDX/etc silently reads the wrong
    columns instead of erroring. Caught in practice, two distinct failure modes:
      1. An auto-picked "overfill" row from a 10-channel-state log (this code now uses 6) credited
         2 kg of reward against its own logged 4,053 kg yield -- a -100% MISMATCH.
      2. A DIFFERENT candidate passed that check (same channel count) but its logged max Wt was
         only ~1% over WT_SOFT_HI, and round-tripping through TODAY'S STATE_RANGES (which may have
         shifted since the log was written) decoded it back UNDER the threshold -- the reward
         looked fine, but the very penalty this scenario exists to exercise never fired.
    Three checks accordingly: state width must match STATE_DIM; decoding must reconstruct the
    logged yield_kg (the SAME reward-vs-batch_yield_kg guard as the rest of this script); and if
    `required_nonzero_term` is given (the term this scenario is meant to exercise -- "soft" for an
    overfill pick, "visc_soft" for borderline), it must actually be nonzero under today's decode,
    not just nonzero in the log's OWN monitor.pkl summary.
    """
    try:
        log = pickle.load(open(Path(run_dir) / "log.pkl", "rb"))
        states, inputs = log["state_samples_history"][episode], log["input_samples_history"][episode]
    except Exception:
        return False
    if np.asarray(states).shape[-1] != STATE_DIM:
        return False
    # soft_penalty/visc_penalty=1.0 (not 0): required_nonzero_term needs a nonzero lambda to be
    # able to fire at all. p_weight=1.0 keeps `reward` directly in kg for the yield check below.
    probe = PeniMassChangeCost(p_weight=1.0, soft_penalty=1.0, rate_penalty=0.0, visc_penalty=1.0,
                              risk_weight=0.0, harvest_reward=True, constraint_strength=1.0)
    try:
        terms = evaluate(probe, states, inputs, drop_first=False)
    except Exception:
        return False
    if not (yield_kg > 0 and abs(terms["reward"] / yield_kg - 1) < tol):
        return False
    if required_nonzero_term is not None and terms[required_nonzero_term] <= 1e-6:
        return False
    return True


def scan_real_episodes(scan_dir, min_yield, max_validated=8):
    """Find the strongest REAL evidence for the two scenarios a synthetic overfeed probe can't
    provide (see the module docstring's FOUR SCENARIOS note): a logged episode that actually
    breaches the weight limit, and one that grazes the viscosity limit while still yielding well.
    Among matches, tries the HIGHEST-YIELD ones FIRST (a policy that otherwise looks attractive is
    the case a penalty must get right) and accepts the first that passes
    `_candidate_is_trustworthy` -- so a state-encoding-mismatched old log is skipped rather than
    silently handed to the report as if it were valid evidence.

    Returns (overfill, borderline, stats) where each of the first two is either None or
    (run_dir: Path, episode: int, max_wt, max_visc, yield_kg), and `stats` is a dict with
    n_scanned / max_wt_seen / max_visc_seen for the "nothing matched" warning message.
    """
    mon_files = sorted(Path(scan_dir).glob("*/monitor.pkl"))
    candidates = []
    for mon_fp in mon_files:
        try:
            mons = pickle.load(open(mon_fp, "rb"))
        except Exception:
            continue
        if not isinstance(mons, list):
            continue
        for i, mon in enumerate(mons):
            try:
                max_wt = float(np.max(mon["Wt"]))
                max_visc = float(np.max(mon["Viscosity"]))
                yld = float(np.sum(mon["yield_per_run"]))
            except Exception:
                continue
            candidates.append((mon_fp.parent, i, max_wt, max_visc, yld))

    stats = {"n_scanned": len(candidates),
             "max_wt_seen": max((c[2] for c in candidates), default=float("nan")),
             "max_visc_seen": max((c[3] for c in candidates), default=float("nan"))}
    if not candidates:
        return None, None, stats

    good = [c for c in candidates if c[4] > min_yield]

    def pick_trustworthy(cands, required_nonzero_term):
        for c in sorted(cands, key=lambda c: -c[4])[:max_validated]:
            if _candidate_is_trustworthy(c[0], c[1], c[4], required_nonzero_term=required_nonzero_term):
                return c
        return None

    # Isolated first choice (low viscosity, so the row cleanly tests ONLY the weight penalty);
    # relax the isolation if nothing reachable does that -- a non-isolated real violation is still
    # far better evidence than none. required_nonzero_term="soft": the point of this scenario is
    # that the WEIGHT penalty fires, so demand that, not just that max_wt LOOKED over threshold in
    # the log's own monitor.pkl (see _candidate_is_trustworthy's failure mode 2).
    overfill = pick_trustworthy([c for c in good if c[2] > WT_SOFT[1] and c[3] < VISC_SOFT_START],
                                required_nonzero_term="soft")
    if overfill is None:
        overfill = pick_trustworthy([c for c in good if c[2] > WT_SOFT[1]], required_nonzero_term="soft")

    band_lo, band_hi = VISC_MAX - 15.0, VISC_MAX + 20.0
    borderline = pick_trustworthy([c for c in good if band_lo <= c[3] <= band_hi and c[2] < WT_SOFT[1]],
                                  required_nonzero_term="visc_soft")
    if borderline is None:
        borderline = pick_trustworthy([c for c in good if VISC_SOFT_START <= c[3] <= 1.3 * VISC_MAX],
                                      required_nonzero_term="visc_soft")

    return overfill, borderline, stats


def load_episode_scenario(run_dir, episode, label, kind):
    """Materialise a Scenario from a logged (run_dir, episode) pair -- the SAME log.pkl/monitor.pkl
    format --bad_run already reads, factored out so `scan_real_episodes`'s picks and an explicit
    --bad_run can share one loader."""
    log = pickle.load(open(Path(run_dir) / "log.pkl", "rb"))
    mons = pickle.load(open(Path(run_dir) / "monitor.pkl", "rb"))
    return Scenario(label, log["state_samples_history"][episode], log["input_samples_history"][episode],
                    mons[episode], varying=True, kind=kind)


def build_rows(cost, scenarios):
    """One Row per Scenario, scored under `cost`. Kept separate from rolling (see `Scenario`) so
    the sweep can re-score the SAME real trajectories under many constraint_strength values without
    re-running the simulator."""
    return [Row(s.label, evaluate(cost, s.states, s.inputs, drop_first=not s.varying),
               batch_yield_kg(s.mon), float(np.max(s.mon["Viscosity"])),
               float(np.max(s.mon["Wt"])), s.varying)
            for s in scenarios]


def report(rows, p_weight):
    """rows: list of Row(label, terms, yield_kg, peak_visc, max_Wt, varying_action)."""
    kg = 1.0 / p_weight
    w = 15
    print(f"\n{'':22}" + "".join(f"{r.label:>{w}}" for r in rows))
    print("-" * (22 + w * len(rows)))
    print(f"{'batch_yield_kg':22}" + "".join(f"{r.yield_kg:>{w},.0f}" for r in rows))
    print(f"{'peak viscosity (cP)':22}" + "".join(f"{r.peak_visc:>{w},.1f}" for r in rows))
    print(f"{'max Wt (kg)':22}" + "".join(f"{r.max_wt:>{w},.0f}" for r in rows))
    print(f"{'action varies?':22}" + "".join(f"{('yes' if r.varying else 'NO'):>{w}}" for r in rows))
    print()
    print(f"{'--- cost units ---':22}")
    for t in TERM_ORDER:
        print(f"{t:22}" + "".join(f"{r.terms[t]:>{w},.2f}" for r in rows))
    print()
    print(f"{'--- kg equivalent ---':22}")
    for t in TERM_ORDER:
        print(f"{t:22}" + "".join(f"{r.terms[t] * kg:>{w},.0f}" for r in rows))
    print()
    # Undefined against a non-positive reward -- a collapsed batch can have net-negative mass gain,
    # and dividing by a 1e-9 floor prints ~1e11% rather than admitting the ratio is meaningless.
    cells = []
    for r in rows:
        pen = sum(r.terms[t] for t in TERM_ORDER[1:])
        cells.append(f"{100 * pen / r.terms['reward']:>{w - 1},.1f}%" if r.terms["reward"] > 0
                     else f"{'n/a':>{w}}")
    print(f"{'penalty % of reward':22}" + "".join(cells))


def verdicts(rows, p_weight):
    print("\n=== verdicts ===")
    # 1. Does the reward measure what we report?
    for r in rows:
        credited = r.terms["reward"] / p_weight
        err = 100 * (credited / r.yield_kg - 1) if r.yield_kg else float("nan")
        flag = "OK" if abs(err) < 10 else "MISMATCH"
        print(f"[{flag}] {r.label}: reward credits {credited:,.0f} kg vs batch_yield_kg "
              f"{r.yield_kg:,.0f} ({err:+.1f}%)")
    print("     -> the optimiser maximises the reward; if it disagrees with batch_yield_kg, it is")
    print("        not optimising the number your baselines are scored on.")

    # 2. Which penalties are inert / dominant?
    print()
    rew_max = max(r.terms["reward"] for r in rows)
    for t in TERM_ORDER[1:]:
        # action_rate is a squared FIRST DIFFERENCE of the action, so it is identically zero on any
        # constant-action probe -- the recipe row (pid_baseline never writes `inputs`) and a constant
        # lambda both qualify. Calling that INERT would blame rate_penalty for a property of the
        # probe, so only rows carrying genuinely varying actions can support a verdict here.
        eligible = [r for r in rows if r.varying] if t == "action_rate" else rows
        if not eligible:
            print(f"[N/A]    {t}: no row carries a varying action, so this term cannot fire here. "
                  f"Re-run with --bad_run to score real logged actions.")
            continue
        worst = max(r.terms[t] for r in eligible)
        if worst < 1e-6:
            print(f"[INERT]  {t}: 0.00 on every eligible batch -- fires nowhere, weight tunes nothing.")
            continue
        share = 100 * worst / rew_max if rew_max > 0 else float("inf")
        tag = "DOMINANT" if share > 100 else ("ACTIVE" if share > 5 else "WEAK")
        print(f"[{tag:8}] {t}: up to {worst:.2f} units = {worst / p_weight:,.0f} kg "
              f"({share:.1f}% of the largest reward)")

    # 3. Is the viscosity penalty proportionate to the loss it should deter?
    ys = [r.yield_kg for r in rows]
    if len(ys) > 1 and max(ys) - min(ys) > 0:
        lost_kg = max(ys) - min(ys)
        worst_visc = max(r.terms["visc_soft"] for r in rows) / p_weight
        if worst_visc < 1e-6:
            # Ratio undefined. The old code divided by a 1e-9 floor and printed ~1e12x, which reads
            # like a computed calibration factor rather than "the term never fired".
            print(f"\n[SCALE]  the bad batch loses {lost_kg:,.0f} kg vs the good one, and the "
                  f"viscosity penalty NEVER FIRED (0 kg charged).")
            print(f"         -> no calibration factor is computable. Check that peak viscosity "
                  f"actually exceeds VISC_MAX={VISC_MAX:g} on the bad row, and that")
            print(f"            STATE_RANGES['Viscosity'] tops out ABOVE VISC_MAX -- if they are "
                  f"equal the state clips at the threshold and the term can never fire.")
            return
        print(f"\n[SCALE]  the bad batch loses {lost_kg:,.0f} kg vs the good one, and the viscosity "
              f"penalty charges {worst_visc:,.0f} kg for it.")
        if worst_visc < 0.5 * lost_kg:
            print(f"         -> UNDER-priced by ~{lost_kg / worst_visc:.0f}x on real trajectories. "
                  f"Raising the weight by that factor would make it proportionate,")
            print(f"            BUT the penalty acts on GP-predicted viscosity -- check its k-step "
                  f"accuracy first (evaluate_GPs.ipynb G.7).")
        elif worst_visc > 2 * lost_kg:
            print("         -> OVER-priced: the optimiser will sacrifice more yield than the failure "
                  "actually costs.")


def sweep_table(scenarios, args, cs_values):
    """THE defence artefact: re-score the already-rolled scenarios under every constraint_strength
    in `cs_values`, printing the full report+verdicts at each value (so the per-term INERT/WEAK/
    ACTIVE/DOMINANT verdicts and the viscosity under/over-pricing check are visible at every point
    on the trade-off curve, not just the CLI default) and returning one tidy row per
    (constraint_strength, scenario) for the table/plot.
    """
    records = []
    for cs in cs_values:
        cost = PeniMassChangeCost(p_weight=args.p_weight, soft_penalty=args.soft_penalty,
                                  rate_penalty=args.rate_penalty, visc_penalty=args.visc_penalty,
                                  risk_weight=args.risk_weight, harvest_reward=args.harvest_reward,
                                  constraint_strength=cs)
        rows = build_rows(cost, scenarios)
        print(f"\n\n{'=' * 70}\nconstraint_strength = {cs:g}\n{'=' * 70}")
        report(rows, args.p_weight)
        verdicts(rows, args.p_weight)
        for s, r in zip(scenarios, rows):  # build_rows preserves scenario order -- see its docstring
            pen_kg = sum(r.terms[t] for t in TERM_ORDER[1:]) / args.p_weight
            reward_kg = r.terms["reward"] / args.p_weight
            records.append({
                "constraint_strength": cs, "batch": r.label, "kind": s.kind,
                "reward_kg": reward_kg,
                "soft_kg": r.terms["soft"] / args.p_weight,
                "visc_soft_kg": r.terms["visc_soft"] / args.p_weight,
                "action_rate_kg": r.terms["action_rate"] / args.p_weight,
                "penalty_pct_reward": 100 * pen_kg * args.p_weight / r.terms["reward"]
                                      if r.terms["reward"] > 0 else np.nan,
                "net_kg": reward_kg - pen_kg,  # r - lambda*c, in kg -- see NET_KG note in the module docstring
                "peak_visc_cP": r.peak_visc, "max_wt_kg": r.max_wt,
                "batch_yield_kg": r.yield_kg,
            })
    return pd.DataFrame(records)


def recommend_constraint_strength(df, good_label):
    """Pick the smallest swept constraint_strength where the two-sided net_kg check (see the
    module docstring's NET_KG note) holds for every scenario: catastrophic rows fall below the
    recipe's net, overfill/borderline rows stay at or above it.

    Returns ((cs, fully_satisfied) or None, per_cs_details). `fully_satisfied=False` means every
    swept value fails at least one condition, so the returned cs is only the closest partial match
    -- a real possibility, and a signal to widen --sweep, not a value to trust blindly.
    """
    details = []
    full_match = None
    best_partial = None  # (cs, n_ok, n_total)
    for cs in sorted(df["constraint_strength"].unique()):
        sub = df[df["constraint_strength"] == cs].set_index("batch")
        if good_label not in sub.index:
            continue
        good_net = sub.loc[good_label, "net_kg"]
        checks = []
        for batch, row in sub.iterrows():
            if row["kind"] == "catastrophic":
                checks.append((batch, "catastrophic (should fall below recipe)",
                              row["net_kg"] < good_net))
            elif row["kind"] in ("overfill", "borderline"):
                checks.append((batch, f"{row['kind']} (should stay >= recipe)",
                              row["net_kg"] >= good_net))
        details.append((cs, checks))
        if not checks:
            continue
        n_ok = sum(ok for _, _, ok in checks)
        if n_ok == len(checks) and full_match is None:
            full_match = cs
        if best_partial is None or n_ok > best_partial[1]:
            best_partial = (cs, n_ok, len(checks))

    if full_match is not None:
        return (full_match, True), details
    if best_partial is not None:
        return (best_partial[0], False), details
    return None, details


# One color per scenario kind, fixed order (never reassigned by which kinds happen to appear) --
# validated colorblind-safe as a set via the dataviz skill's palette validator (light mode, all 5
# slots PASS: lightness band, chroma floor, CVD separation, normal-vision floor; the contrast WARN
# on 3 of the 5 is covered by the legend + the full console/CSV table this script always prints).
COLOR_BY_KIND = {"good": "#2a78d6", "catastrophic": "#eb6834", "overfill": "#1baf7a",
                "borderline": "#eda100", "chatter": "#e87ba4"}
SHORT_LABEL_BY_KIND = {"good": "Recipe", "catastrophic": "Collapsed batch",
                      "overfill": "Overfill batch", "borderline": "Borderline batch",
                      "chatter": "Chatter probe"}
# Distinguishes the 3 penalty types WITHIN a color (panel 0 only, where one scenario needs to show
# more than one term at once) via marker shape rather than a second color.
MARKER_BY_TERM = {"soft_kg": "o", "visc_soft_kg": "s", "action_rate_kg": "^"}
SHORT_LABEL_BY_TERM = {"soft_kg": "Weight", "visc_soft_kg": "Viscosity", "action_rate_kg": "Chatter"}


def plot_sweep(df, out_dir, good_label, recommended=None):
    """The sweep AS a plot, over EVERY scenario present (not just good/bad -- see the module
    docstring's FOUR SCENARIOS note): penalty size, penalty as a share of reward, and the net
    result after the penalty, all against the penalty-strength knob. Each scenario gets its own
    color (fixed order, see COLOR_BY_KIND) so batches are easy to tell apart at a glance; panel 1
    also varies marker shape by penalty type, since one scenario can show more than one there.
    `recommended`, if given as (cs, fully_satisfied) from `recommend_constraint_strength`, is
    marked with a vertical line -- solid if it fully works, dashed if it's the closest partial
    match (see that function's docstring).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kind_of = df.drop_duplicates("batch").set_index("batch")["kind"]
    batches = list(kind_of.index)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    for batch in batches:
        color = COLOR_BY_KIND.get(kind_of[batch], "#767671")
        short = SHORT_LABEL_BY_KIND.get(kind_of[batch], batch)
        sub = df[df.batch == batch].sort_values("constraint_strength")
        for term, marker in MARKER_BY_TERM.items():
            y = sub[term].to_numpy()
            if np.allclose(y, 0.0):
                continue
            ax[0].plot(sub["constraint_strength"], y, "-", color=color, marker=marker, ms=5,
                      label=f"{short} – {SHORT_LABEL_BY_TERM[term]}")
        ax[1].plot(sub["constraint_strength"], sub["penalty_pct_reward"], "-o", color=color,
                  ms=5, label=short)
        ax[2].plot(sub["constraint_strength"], sub["net_kg"], "-o", color=color,
                  ms=5, label=short)

    # symlog, not log: several series pass through an EXACT zero (constraint_strength=0, or a term
    # that never fires on a given batch). A log axis can't represent that and clamping to a floor
    # would draw a fake nonzero point -- misleading in a plot meant to defend a calibration choice.
    ax[0].set_yscale("symlog", linthresh=1e-2)
    ax[0].set_xlabel("penalty strength")
    ax[0].set_ylabel("penalty (kg)")
    ax[0].set_title("Penalty size, by type")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7)

    ax[1].axhline(100, color="0.4", ls=":", lw=1.5, label="penalty = 100% of reward")
    ax[1].set_yscale("symlog")
    ax[1].set_xlabel("penalty strength")
    ax[1].set_ylabel("% of reward lost to penalty")
    ax[1].set_title("Penalty as a share of reward")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=7)

    if good_label in kind_of.index:
        good_net = df[df.batch == good_label].sort_values("constraint_strength")
        ax[2].plot(good_net["constraint_strength"], good_net["net_kg"], color="k", lw=2.5,
                  alpha=0.2, zorder=0, label="_recipe (reference)")
    if recommended is not None and recommended[0] is not None:
        cs, satisfied = recommended
        ax[2].axvline(cs, color="0.3", ls="-" if satisfied else "--", lw=1.5,
                      label=f"suggested strength = {cs:g}" + ("" if satisfied else " (closest match)"))
    ax[2].set_xlabel("penalty strength")
    ax[2].set_ylabel("net kg (reward minus penalty)")
    ax[2].set_title("Result after the penalty\n(above the recipe line = still a win)")
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=7)

    fig.suptitle("How the penalty-strength knob changes each batch", y=1.02)
    fig.tight_layout()
    fp = out_dir / "constraint_strength_sweep.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=700000, help="held-out seed for the good/recipe batch")
    p.add_argument("--bad_run", type=str, default=None,
                   help="run dir holding a collapsed episode to score as the BAD batch")
    p.add_argument("--bad_episode", type=int, default=None,
                   help="episode index within that run's log (see monitor.pkl ordering)")
    p.add_argument("--p_weight", type=float, default=0.5)
    p.add_argument("--soft_penalty", type=float, default=0.5, help="lambda_weight (tank-overflow)")
    p.add_argument("--rate_penalty", type=float, default=0.5, help="lambda_rate (action smoothness)")
    p.add_argument("--visc_penalty", type=float, default=0.5, help="lambda_visc (viscosity collapse)")
    p.add_argument("--risk_weight", type=float, default=0.0,
                   help="lambda_risk (batch-outcome spread) -- accepted for completeness but NOT "
                        "exercised by this script; see the RISK TERM note in the module docstring")
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false")
    p.add_argument("--alternating", action="store_true",
                   help="add a +/-1 alternating-action row so action_rate is exercised without "
                        "--bad_run (bounds the most that term can ever charge)")
    p.add_argument("--constraint_strength", type=float, default=1.0,
                   help="global knob for the primary (single-value) report below; the --sweep "
                        "section always also runs across --sweep regardless of this value")
    p.add_argument("--sweep", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                   help="constraint_strength values for the defence sweep (weight/viscosity "
                        "constraint lambdas scaled together; action_rate is untouched by this)")
    p.add_argument("--no_sweep", action="store_true", help="skip the constraint_strength sweep")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                   help="where the sweep CSV/plot are written")
    p.add_argument("--scan_dir", type=str, default=str(Path(_ROOT) / "results" / "single_phase"),
                   help="run-log directory scanned for real overfill/borderline episodes (see the "
                        "module docstring's FOUR SCENARIOS note); degrades gracefully if empty")
    p.add_argument("--no_scan", action="store_true",
                   help="skip the real-episode scan (falls back to recipe + synthetic bad batch only "
                        "-- a WEAKER defence artefact, see the module docstring)")
    p.add_argument("--min_scan_yield", type=float, default=1000.0,
                   help="only scanned episodes above this yield_kg are eligible, so the overfill/"
                        "borderline picks are otherwise-attractive policies, not degenerate flukes")
    args = p.parse_args()

    print(f"weights: p={args.p_weight} soft(lambda_weight)={args.soft_penalty} "
          f"visc(lambda_visc)={args.visc_penalty} rate(lambda_rate)={args.rate_penalty} "
          f"risk(lambda_risk)={args.risk_weight} harvest={args.harvest_reward}")
    print(f"thresholds: VISC_MAX={VISC_MAX:g} (old fixed scale {VISC_SOFT_SCALE:g}, no longer used "
          f"by the ramp) | WT_SOFT_HI={WT_SOFT[1]:g} (overflow {WT_OVERFLOW:g})")
    print(f"1 cost unit == {1 / args.p_weight:.0f} kg penicillin")

    if args.bad_run:
        print("NOTE: weights above are this script's CLI defaults, not the ones --bad_run trained "
              "with.\n      Cross-check against the run's note.txt before reading the verdicts as "
              "that run's costs.")

    # Roll every real trajectory ONCE. Cost re-evaluation (below, and in the sweep) is cheap tensor
    # arithmetic on these same states/inputs -- only the simulator rollout is expensive.
    wrapper = PenSimWrapper()
    scenarios = []

    def add(label, states, inputs, mon, varying, kind):
        scenarios.append(Scenario(label, states, inputs, mon, varying, kind))

    # GOOD: the recipe on a held-out seed. pid_baseline=True forces Fs to the recipe profile, so
    # this is the reference trajectory every baseline is scored against. It carries no action at
    # all (rollout skips the policy branch), hence varying=False and step 0 dropped.
    good_label = "recipe"
    st, ins, mon = roll(wrapper, args.seed, policy=None, pid_baseline=True)
    add(good_label, st, ins, mon, varying=False, kind="good")

    # BAD: a real collapsed episode, replayed from a training log. Scored on its RECORDED states
    # and actions -- no model involved, so these are the true costs of a batch that actually failed,
    # and the actions genuinely vary, so action_rate is meaningful on this row.
    if args.bad_run and args.bad_episode is not None:
        bad_label = f"ep{args.bad_episode} ({os.path.basename(args.bad_run)})"
        s = load_episode_scenario(args.bad_run, args.bad_episode, bad_label, kind="catastrophic")
        scenarios.append(s)
    else:
        # Fallback with no run supplied: sustained overfeed reliably drives the viscosity collapse.
        bad_label = "a=+1 (overfeed)"
        st, ins, mon = roll(wrapper, args.seed, policy=lambda s, i: np.array([1.0]))
        add(bad_label, st, ins, mon, varying=False, kind="catastrophic")

    # A varying-action probe, so action_rate is exercised even without --bad_run. Alternating +/-1
    # every decision is the worst case the action space allows, which brackets what the term can
    # ever charge -- a real policy will always sit below this.
    if args.alternating:
        st, ins, mon = roll(wrapper, args.seed,
                            policy=lambda s, i: np.array([1.0 if i % 2 == 0 else -1.0]))
        add("a=+/-1 (max chatter)", st, ins, mon, varying=True, kind="chatter")

    # REAL overfill / borderline episodes from actual training logs (see the module docstring's
    # FOUR SCENARIOS note) -- these are what make the defence stand up to "does the weight penalty
    # ever fire?" and "does the penalty over-penalise a policy that merely grazes the limit?", which
    # the synthetic probes above cannot answer.
    if not args.no_scan:
        overfill, borderline, scan_stats = scan_real_episodes(args.scan_dir, args.min_scan_yield)
        if scan_stats["n_scanned"] == 0:
            print(f"\n[SCAN] no run logs found under {args.scan_dir} -- skipping real overfill/"
                  f"borderline scenarios. This defence table will NOT show the weight penalty firing "
                  f"or a near-limit good batch; pass --scan_dir to point at real run logs, or accept "
                  f"a weaker artefact.")
        else:
            if overfill is not None:
                run_dir, ep, max_wt, max_visc, yld = overfill
                label = f"overfill: {run_dir.name}/ep{ep} (wt={max_wt:,.0f}kg, yield={yld:,.0f}kg)"
                scenarios.append(load_episode_scenario(run_dir, ep, label, kind="overfill"))
            else:
                print(f"\n[SCAN] scanned {scan_stats['n_scanned']} real episodes under {args.scan_dir} "
                      f"(yield > {args.min_scan_yield:g} kg); NONE exceed WT_SOFT_HI={WT_SOFT[1]:g} "
                      f"(max observed: {scan_stats['max_wt_seen']:,.0f} kg) -- the weight penalty "
                      f"remains UNTESTED by this run of the report.")
            if borderline is not None:
                run_dir, ep, max_wt, max_visc, yld = borderline
                label = f"borderline: {run_dir.name}/ep{ep} (visc={max_visc:,.1f}cP, yield={yld:,.0f}kg)"
                scenarios.append(load_episode_scenario(run_dir, ep, label, kind="borderline"))
            else:
                print(f"\n[SCAN] scanned {scan_stats['n_scanned']} real episodes under {args.scan_dir} "
                      f"(yield > {args.min_scan_yield:g} kg); none land near VISC_MAX={VISC_MAX:g} "
                      f"(max observed: {scan_stats['max_visc_seen']:,.1f} cP) -- no borderline "
                      f"over-conservatism check is available this run.")

    # Primary report at the CLI's single --constraint_strength (default 1.0, i.e. exactly what the
    # per-term lambdas above specify -- unaffected by the sweep below).
    cost = PeniMassChangeCost(p_weight=args.p_weight, soft_penalty=args.soft_penalty,
                              rate_penalty=args.rate_penalty, visc_penalty=args.visc_penalty,
                              risk_weight=args.risk_weight, harvest_reward=args.harvest_reward,
                              constraint_strength=args.constraint_strength)
    print(f"\n{'#' * 70}\nPRIMARY REPORT -- constraint_strength = {args.constraint_strength:g}\n"
          f"{'#' * 70}")
    rows = build_rows(cost, scenarios)
    report(rows, args.p_weight)
    verdicts(rows, args.p_weight)

    if args.no_sweep:
        return

    print(f"\n\n{'#' * 70}\nCONSTRAINT_STRENGTH SWEEP -- the calibration defence\n{'#' * 70}")
    print(f"values: {args.sweep}")
    scenario_kinds = ", ".join(f"{s.label} [{s.kind}]" for s in scenarios)
    print(f"scenarios: {scenario_kinds}")
    df = sweep_table(scenarios, args, args.sweep)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_fp = out_dir / "constraint_strength_sweep.csv"
    df.to_csv(csv_fp, index=False)

    recommended, rec_details = recommend_constraint_strength(df, good_label)
    print("\n=== recommended constraint_strength (net_kg two-sided check, see module docstring) ===")
    if recommended is None:
        print("[N/A] no catastrophic/overfill/borderline scenario is present to check against -- "
              "pass --bad_run, --alternating, or ensure --scan_dir finds real episodes.")
    else:
        cs, satisfied = recommended
        tag = "SATISFIES BOTH SIDES" if satisfied else "BEST PARTIAL MATCH (no swept value fully works)"
        print(f"[{tag}] constraint_strength = {cs:g}")
        checks = dict(rec_details)[cs]
        for batch, rule, ok in checks:
            print(f"    {'OK' if ok else 'FAIL':4} {batch}: {rule}")
        if not satisfied:
            print("    -> widen --sweep (finer steps, or a larger range) to find a value that "
                  "satisfies every condition.")
        print("    NOTE: this is calibrated on REAL trajectories. Viscosity is over-predicted ~4.5x "
              "during training (see IMPORTANT CAVEAT), so the effective in-training penalty at this "
              "constraint_strength will fire harder than shown here -- treat this as a starting "
              "point to re-validate against predicted-state rollouts, not a final answer.")

    plot_fp = plot_sweep(df, out_dir, good_label, recommended)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    cols = ["constraint_strength", "batch", "kind", "reward_kg", "soft_kg", "visc_soft_kg",
           "action_rate_kg", "net_kg", "penalty_pct_reward", "peak_visc_cP", "max_wt_kg"]
    print("\n=== sweep summary table (this is the dissertation-ready artefact) ===")
    print(df[cols].to_string(index=False,
          float_format=lambda x: f"{x:10.2f}" if abs(x) < 1e5 else f"{x:10.3e}"))
    print(f"\nsaved: {csv_fp}\n       {plot_fp}")
    print("\nREMINDER: these lambdas are calibrated on REAL trajectories. Training applies them to "
          "GP-PREDICTED states, where viscosity is over-predicted several-fold (see the module "
          "docstring's IMPORTANT CAVEAT) -- re-check against predicted-state rollouts before "
          "treating a chosen constraint_strength as final.")


if __name__ == "__main__":
    main()

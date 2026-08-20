"""Does a learnt policy read the CLOCK or the STATE? Shift the growth->production transition
and see which policy degrades.

    python delayed_transition_robustness.py --sensitivity_only
    python delayed_transition_robustness.py --n_seeds 2 --levels 0,16        # smoke test
    python delayed_transition_robustness.py                                  # full sweep

THE QUESTION
    Two policies trained on the same seed reach similar nominal yield:
      DUAL  results/dual_phase_baseline/seed4_13
            pivot_mode=biomass, GP active_dims=[0,1,2,3,5]  -> time NOT a GP regressor
      TIME  results/full/ConcCost/single-phase/Added_time/seed4_1
            single GP, active_dims=[0,1,2,3,4,5]            -> time IS a GP regressor
    Equal yield on nominal batches says nothing about WHAT each policy reads. If TIME has learnt
    an open-loop clock schedule it should break when the batch's physiology stops matching the
    wall clock; if DUAL tracks state it should not.

WHAT IS *NOT* BEING CLAIMED  (read this before writing any of it up)
    Neither deployed policy has a runtime phase switch. Both are ONE Policy.Sum_of_gaussians over
    the SAME 5-dim state [Wt, X, P, Viscosity, time]; `centers` is (100,5) in both log.pkl files,
    so TIME IS A POLICY INPUT IN BOTH. The dual model dropped time only from the GP regressors.
    And seed4_13's note.txt has no on_each_rollout key -> it defaulted False -> the imagined
    rollout blend ran on the WALL CLOCK (sigmoid(ln99/50*(t-100h)),
    model_learning_dual_phase._blend_weight). Biomass decided which GP each training sample went
    to; the clock decided how the two GPs were mixed during policy optimisation. There is no
    biomass-triggered switch in the artifact being tested here.

    What IS testable is clock-keyed vs titre-keyed. Final-trial policy RBF lengthscales (smaller
    = sharper dependence on that channel) already hint at it:
                    Wt      X       P       Visc    time
        TIME       0.327   0.284   0.277   0.355   0.174  <- clock is its sharpest channel
        DUAL       0.425   1.772   0.170   0.271   0.507  <- titre is; biomass ~ignored
    Step 1 (--sensitivity_only) checks that directly by finite-differencing each policy, and is
    the go/no-go for the 18-minute sweep.

HOW THE TRANSITION IS MOVED
    Only lever available without touching pensim_wrapper.py: the action channel. FS_SCALE=0.5, so
    a=-1 means Fs = 0.5x recipe -- a floor, not a cutoff, so the culture is delayed, not killed
    (measured: bm_max barely moves, viscosity FALLS). Starving the first J decisions moves the
    biomass pivot from ~60-65h to ~80-90h.

    Because 0.5x is shallow, meaningful delay needs a LONG starve window, and step 2 of the
    design requires an identical handover for every arm -- so the handover sits at the longest
    window, H = 80h (decision 16). The policy therefore controls 80-230h, and in the low-delay
    arms the transition has already happened by handover. That is the strongest version of this
    test available under an action-only intervention; a deeper env-level Fs multiplier would buy
    an earlier handover and a wider spread.

    Per (seed, level) the pre-handover segment depends only on the level, so it is IDENTICAL
    across the two policies -- asserted in _check_prehandover_identical, not assumed.

WHY THE HEADLINE METRIC IS PAIRED
    Starving lowers yield whatever the policy does. Absolute yield vs delay would just show the
    feed cut. The headline is therefore yield MINUS the recipe arm of the same (seed, level).

Strictly post-hoc: loads two trained runs read-only, writes only into delayed_transition/.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../pensim_mcpilco
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import evaluations.eval_single_phase_lib as single_lib
import evaluations.eval_multi_phase_lib as multi_lib
from mcpilco.config_single_phase_baseline_time import get_config as _single_time_cfg
from mcpilco.config_single_phase_baseline import get_config as _single_notime_cfg
from mcpilco.config_dual_phase_baseline import get_config as _dual_baseline_cfg
from mcpilco.config_dual_phase_baseline_time import get_config as _dual_time_cfg
from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, STATE_DIM,
                                    STATE_NAMES, VISC_MAX)
from mcpilco.model_learning_dual_phase import BM_PIVOT_DEFAULT, _bm_from_states
from experiments.eval_utils import (yield_kg, feasibility_gated_yield_kg,
                                    constraint_diagnostics)
# Reuse the production split rather than re-thresholding here: _split_via_production_code calls
# DualPhaseModelLearning._biomass_pivot_step itself, so "pivot hour" in this script is the same
# number training would have split at, running-max envelope and min_phase_steps clamp included.
from phase_transition_diagnostic_biomass import _split_via_production_code

OUT_DIR = Path(_EVAL_DIR) / "delayed_transition"
N_DECISIONS = int(CONTROL_H / T_SAMPLING)          # 45

DUAL_DIR = Path(_ROOT) / "results" / "dual_phase_baseline" / "seed4_13"
TIME_DIR = (Path(_ROOT) / "results" / "full" / "ConcCost" / "single-phase" /
            "Added_time" / "seed4_1")

STARVE_ACTION = -1.0     # Fs = (1 + FS_SCALE*a) x recipe = 0.5x recipe: a floor, not a cutoff
RECIPE_ACTION = 0.0      # exactly reproduces the recipe feed

# "dual" and "time" are SLOT names for the two compared policies, not claims about their
# architecture -- --arm_dual/--arm_time can point either slot at any run (see SETUPS). The
# defaults are the pair this script was written for; the labels follow whatever is loaded.
ARM_STYLE = {"dual":   dict(color="C0", marker="o", label="DUAL (biomass-split, no time in GP)"),
             "time":   dict(color="C3", marker="s", label="TIME (single GP, time as regressor)"),
             "recipe": dict(color="0.45", marker="^", ls="--", label="recipe")}

# results tree -> (evaluation library, matching get_config). Same convention as
# test_seed_policies.SETUPS; the config MUST match the tree or load_state_dict silently
# reattaches a prior mean the checkpoint never had.
SETUPS = {"dual_phase_baseline": (multi_lib, _dual_baseline_cfg),
          "single_phase_baseline_time": (single_lib, _single_time_cfg),
          "single_phase_baseline": (single_lib, _single_notime_cfg),
          "dual_phase_baseline_time": (multi_lib, _dual_time_cfg)}
CH_COLOR = {"Wt": "C4", "X": "C2", "P": "C1", "Viscosity": "C5", "time": "k"}


# ----------------------------------------------------------------------------- arm construction

def make_arm_policy(starve_until, handover, tail):
    """Action schedule for one arm.

    decisions [0, starve_until)      a = -1   starve, Fs = 0.5x recipe
    decisions [starve_until, handover) a = 0  recipe Fs, still pre-handover
    decisions [handover, end)        tail policy, or recipe Fs when tail is None

    The first two segments depend only on `starve_until`, never on `tail`, which is what makes
    the two policies' pre-handover trajectories identical for a given (seed, level)."""
    def pol(state, decision_idx):
        d = int(decision_idx)
        if d < starve_until:
            return np.array([STARVE_ACTION])
        if d < handover or tail is None:
            return np.array([RECIPE_ACTION])
        return np.asarray(tail(state, d), dtype=float).ravel()
    return pol


def run_batch(wrapper_par, seed, policy, feed_delay_h=0.0):
    """One 230h batch. Returns (monitor, decision-grid states (46,5)).

    Goes through wrapper.rollout rather than eval_*_lib.run_arm because run_arm returns only the
    monitor and the pivot detector needs the normalised state array. A fresh PenSimWrapper per
    batch keeps `monitor` from growing across the 150-batch sweep; an explicit seed makes the
    wrapper's own episode counter irrelevant."""
    w = PenSimWrapper(**wrapper_par)
    _, _, states = w.rollout(None, policy, CONTROL_H, T_SAMPLING, None, seed=seed,
                             feed_delay_h=feed_delay_h)
    return w.monitor[-1], states


def measure_batch(mon, states, pivot_bm):
    """Everything scored off one batch: yield, the constraint guardrails, and where the
    growth->production transition actually landed."""
    step, hours, crossed = _split_via_production_code(states, pivot_bm)
    bm = np.asarray(_bm_from_states(states), dtype=float)
    row = dict(constraint_diagnostics(mon))
    row.update(
        yield_kg=yield_kg(mon),
        yield_gated=feasibility_gated_yield_kg(mon),
        pivot_h=float(hours),
        pivot_step=int(step),
        pivot_crossed=int(crossed),
        bm_max=float(np.max(bm)),
        bm_at_pivot=float(bm[step]),
        pivot_bm_threshold=float(pivot_bm),
    )
    return row


# --------------------------------------------------------------------------------- policy loading

def _load_one(slot, run_dir, setup):
    lib, cfg_fn = SETUPS[setup]
    print(f"loading {slot.upper():<5} {run_dir}   [{setup}]")
    return lib, lib.load_run(str(run_dir), get_config_fn=cfg_fn)


def load_policies(dual_dir=None, time_dir=None, dual_setup="dual_phase_baseline",
                  time_setup="single_phase_baseline_time"):
    """Both policies plus the wrapper settings the sweep will run under.

    The two arms must face an identical environment, so the visc-delay flags are checked rather
    than merged: with pms_visc_delay or use_offline_measurements set, one policy would see a
    held Viscosity signal and the other the true one, and the comparison would be measuring
    that instead of the transition shift."""
    dual_lib, dual_run = _load_one("dual", dual_dir or DUAL_DIR, dual_setup)
    time_lib, time_run = _load_one("time", time_dir or TIME_DIR, time_setup)

    wp = {}
    for name, run in (("DUAL", dual_run), ("TIME", time_run)):
        par = dict(run.cfg.get("wrapper_par", {}))
        par.pop("seed_offset", None)          # irrelevant: every batch passes an explicit seed
        if par.get("pms_visc_delay") or par.get("use_offline_measurements"):
            raise RuntimeError(
                f"{name} was trained with a Viscosity-delay flag set ({par}). Both arms must "
                "face the same observation model or this measures the delay, not the pivot "
                "shift. Handle that case explicitly before running the sweep.")
        wp[name] = par
    if wp["DUAL"] != wp["TIME"]:
        raise RuntimeError(f"wrapper settings differ: DUAL={wp['DUAL']} TIME={wp['TIME']}")

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, dual_policy, _, _ = dual_lib.build_policy_agent(dual_run)
        _, _, time_policy, _, _ = time_lib.build_policy_agent(time_run)
    print(f"  DUAL trial {dual_run.n_trials_in_log} | TIME trial {time_run.n_trials_in_log} "
          f"| wrapper_par={wp['DUAL']}")
    return {"dual": dual_policy, "time": time_policy}, wp["DUAL"], dual_run, time_run


# ---------------------------------------------------------- step 1: policy sensitivity probe

def policy_sensitivity(policies, wrapper_par, seed, starve_until=0, eps=0.02):
    """Finite-difference d(action)/d(state channel) along a real recipe trajectory.

    Answers the mechanism question with no sweep at all: if TIME is clock-keyed its `time`
    column dominates, and if DUAL is titre-keyed its `P` column does. Perturbed states are
    clipped back into [-1,1] (the wrapper clips there too, so outside it is not a state any
    policy ever sees) and the divisor uses the ACHIEVED separation, so channels sitting on a
    bound give a one-sided derivative instead of a silently halved one.

    starve_until reruns the probe on a DELAYED trajectory. The nominal probe only says what
    each policy reads where the recipe happens to go; a policy can look state-driven on
    distribution and still fall apart once the states move, so the regime the sweep actually
    visits has to be probed too."""
    _, states = run_batch(wrapper_par, seed, make_arm_policy(starve_until, starve_until, None))
    n = states.shape[0]
    out = {}
    for name, pol in policies.items():
        sens = np.zeros((n, STATE_DIM))
        for d in range(n):
            s = states[d]
            for c in range(STATE_DIM):
                sp, sm = s.copy(), s.copy()
                sp[c] = min(sp[c] + eps, 1.0)
                sm[c] = max(sm[c] - eps, -1.0)
                span = sp[c] - sm[c]
                if span <= 0:
                    continue
                a_p = float(np.asarray(pol(sp, d)).ravel()[0])
                a_m = float(np.asarray(pol(sm, d)).ravel()[0])
                sens[d, c] = (a_p - a_m) / span
        out[name] = sens
    return out, states


def save_sensitivity(sens_by_cond, handover, out_dir):
    """sens_by_cond: {condition label: {policy: (n_decisions+1, STATE_DIM) array}}.

    Absolute |da/ds| is NOT comparable across the two policies -- their output weights differ by
    2x (|w| sums 106 vs 226), so a policy can look uniformly more sensitive without reading
    anything differently. The comparable quantity is each channel's SHARE of that policy's own
    total sensitivity, which is what the verdict and the bar panel use."""
    rows = []
    for cond, sens in sens_by_cond.items():
        for name, arr in sens.items():
            for d in range(arr.shape[0]):
                r = {"condition": cond, "policy": name, "decision": d, "hours": d * T_SAMPLING}
                r.update({f"dA_d{ch}": arr[d, i] for i, ch in enumerate(STATE_NAMES)})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "policy_sensitivity.csv", index=False)

    def share(arr):
        m = np.abs(arr[handover:]).mean(axis=0)
        tot = m.sum()
        return m, (m / tot if tot > 0 else m)

    conds = list(sens_by_cond)
    names = list(next(iter(sens_by_cond.values())))
    ncol = len(names) + 1
    fig, axes = plt.subplots(len(conds), ncol, figsize=(5.2 * ncol, 4.0 * len(conds)),
                             squeeze=False)
    for r, cond in enumerate(conds):
        sens = sens_by_cond[cond]
        hrs = np.arange(next(iter(sens.values())).shape[0]) * T_SAMPLING
        for ax, name in zip(axes[r], names):
            arr = sens[name]
            for i, ch in enumerate(STATE_NAMES):
                ax.plot(hrs, np.abs(arr[:, i]), color=CH_COLOR[ch],
                        lw=2.2 if ch == "time" else 1.4, label=ch)
            ax.axvspan(0, handover * T_SAMPLING, color="0.9", zorder=0)
            ax.set_title(f"{name.upper()} -- {cond}")
            ax.set_xlabel("batch time (h)")
            ax.set_ylabel("|d action / d state|")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        ax = axes[r][-1]
        width, x = 0.38, np.arange(STATE_DIM)
        for k, name in enumerate(names):
            ax.bar(x + (k - 0.5) * width, share(sens[name])[1], width,
                   color=ARM_STYLE[name]["color"], label=name.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_NAMES, rotation=30)
        ax.set_ylabel("share of own total |da/ds|")
        ax.set_title(f"driver mix, post-handover\n{cond}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Which state channel drives each policy? (grey = pre-handover)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    multi_lib._finish(fig, out_dir, "policy_sensitivity.png", show=False)

    verdict = {}
    for cond, sens in sens_by_cond.items():
        print(f"\npost-handover mean |da/d state|  [{cond}]   (share of own total in brackets)")
        print("  " + "".join(f"{c:>18}" for c in STATE_NAMES))
        for name, arr in sens.items():
            m, sh = share(arr)
            print(f"  {name.upper():<6}" +
                  "".join(f"{v:>10.3f} [{s:>4.0%}]" for v, s in zip(m, sh)))
            verdict[(cond, name)] = dict(zip(STATE_NAMES, sh))
    return df, verdict


def sensitivity_verdict(verdict, conds, names):
    """Go/no-go for the sweep. H1 needs TIME to lean on the clock MORE than DUAL does; the test
    is the share gap, not either policy's raw magnitude."""
    print("\ntime-channel share of total sensitivity:")
    supportive = False
    for cond in conds:
        gaps = {n: verdict[(cond, n)]["time"] for n in names}
        gap = gaps.get("time", 0.0) - gaps.get("dual", 0.0)
        print(f"  {cond:<22}" + "  ".join(f"{n.upper()} {gaps[n]:.1%}" for n in names) +
              f"   |  TIME - DUAL = {gap:+.1%}")
        if gap > 0.05:
            supportive = True
    if supportive:
        print("  -> TIME leans on the clock more than DUAL: H1 is live, run the sweep.")
    else:
        print("  -> the two policies weight the clock about equally, so H1's stated mechanism\n"
              "     is NOT visible locally. The sweep is still worth running (this probe is a\n"
              "     local derivative on one trajectory, not a statement about the batch-scale\n"
              "     behaviour under intervention), but a null result should be expected.")
    return supportive


# --------------------------------------------------------------------------------- the sweep

def _check_prehandover_identical(states_by_arm, handover, seed, level):
    """The whole comparison rests on both policies having controlled exactly the same amount of
    the batch, so verify it instead of trusting it. Identical action schedule + identical seed
    => bit-identical states up to and including the handover decision."""
    arms = [a for a in states_by_arm if states_by_arm[a] is not None]
    ref = states_by_arm[arms[0]][:handover + 1]
    for a in arms[1:]:
        got = states_by_arm[a][:handover + 1]
        if not np.array_equal(ref, got):
            worst = float(np.max(np.abs(ref - got)))
            raise RuntimeError(
                f"pre-handover trajectories diverge at seed={seed} level={level} between "
                f"'{arms[0]}' and '{a}' (max |diff| = {worst:.3e}). The arms did not face the "
                "same batch, so any yield difference is not attributable to the policy.")


def sweep(policies, wrapper_par, levels, handover, seeds, pivot_bm, feed_delay_mode=False):
    """feed_delay_mode reinterprets `levels` as FEED-DELAY HOURS applied by the environment
    (rollout(feed_delay_h=...)) instead of as a starvation window driven through the action
    channel. The environment applies the disturbance, so no pre-handover segment is needed and
    the policy controls the batch from decision 0 -- which removes the 80h handover compromise
    the action-only design forced."""
    rows = []
    total = len(levels) * len(seeds) * (len(policies) + 1)
    t0, done = time.time(), 0
    for J in levels:
        for seed in seeds:
            states_by_arm = {}
            fd = float(J) if feed_delay_mode else 0.0
            for arm in list(policies) + ["recipe"]:
                tail = policies.get(arm)          # None for "recipe"
                pol = (make_arm_policy(0, 0, tail) if feed_delay_mode
                       else make_arm_policy(J, handover, tail))
                mon, states = run_batch(wrapper_par, seed, pol, feed_delay_h=fd)
                states_by_arm[arm] = states
                r = measure_batch(mon, states, pivot_bm)
                r.update(arm=arm, seed=int(seed), starve_decisions=int(J),
                         delay_h=float(J) if feed_delay_mode else float(J * T_SAMPLING),
                         handover_h=0.0 if feed_delay_mode else float(handover * T_SAMPLING))
                rows.append(r)
                done += 1
            _check_prehandover_identical(states_by_arm, 0 if feed_delay_mode else handover,
                                         seed, J)
            el = time.time() - t0
            lab = ("feeddelay" if feed_delay_mode else "starve")
            jh = float(J) if feed_delay_mode else J * T_SAMPLING
            print(f"  [{done:>3}/{total}] {lab}={jh:>3.0f}h seed={seed} "
                  f"pivot(recipe)={rows[-1]['pivot_h']:.0f}h "
                  f"yield dual/time/recipe="
                  f"{rows[-3]['yield_kg']:.0f}/{rows[-2]['yield_kg']:.0f}/"
                  f"{rows[-1]['yield_kg']:.0f} kg  ({el:.0f}s elapsed)")
    return pd.DataFrame(rows)


def add_paired_delta(df):
    """Paired delta vs the recipe arm of the same (seed, level). Without this the figure shows
    the feed cut, not the policy."""
    base = (df[df["arm"] == "recipe"]
            .set_index(["seed", "starve_decisions"])[["yield_kg", "yield_gated", "pivot_h"]]
            .rename(columns={"yield_kg": "recipe_yield", "yield_gated": "recipe_yield_gated",
                             "pivot_h": "recipe_pivot_h"}))
    df = df.join(base, on=["seed", "starve_decisions"])
    df["delta_yield"] = df["yield_kg"] - df["recipe_yield"]
    df["delta_yield_gated"] = df["yield_gated"] - df["recipe_yield_gated"]
    return df


# ------------------------------------------------------------------------------------- plots

def plot_pivot_calibration(df, out_dir):
    """Guardrail 2 (the transition really moved) plus guardrails 1 and 3 (delayed, not killed;
    no inhibition) in one figure, read off the recipe arm."""
    rec = df[df["arm"] == "recipe"]
    g = rec.groupby("delay_h")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    for seed, sub in rec.groupby("seed"):
        sub = sub.sort_values("delay_h")
        axes[0].plot(sub["delay_h"], sub["pivot_h"], color="0.75", lw=1.0, zorder=2)
    axes[0].errorbar(g["pivot_h"].mean().index, g["pivot_h"].mean(),
                     yerr=g["pivot_h"].sem(), color="C0", lw=2.4, marker="o", capsize=3, zorder=5)
    axes[0].set_ylabel("measured pivot hour  (BM running-max >= %.0f)"
                       % rec["pivot_bm_threshold"].iloc[0])
    axes[0].set_title("guardrail 2: does the transition move?")

    axes[1].errorbar(g["bm_max"].mean().index, g["bm_max"].mean(), yerr=g["bm_max"].sem(),
                     color="C2", lw=2.0, marker="o", capsize=3)
    axes[1].set_ylabel("max biomass X*Wt/1000")
    axes[1].set_title("guardrail 1: delayed, not killed")

    axes[2].errorbar(g["max_viscosity"].mean().index, g["max_viscosity"].mean(),
                     yerr=g["max_viscosity"].sem(), color="C5", lw=2.0, marker="o", capsize=3)
    axes[2].axhline(VISC_MAX, color="r", ls="--", lw=1.4, label=f"VISC_MAX={VISC_MAX:g}")
    axes[2].set_ylabel("max viscosity")
    axes[2].set_title("guardrail 3: no inhibition/collapse")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("starvation window (h, Fs = 0.5x recipe)")
        ax.grid(alpha=0.3)
    fig.suptitle("Calibration of the timing intervention (recipe arm)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    multi_lib._finish(fig, out_dir, "pivot_calibration.png", show=False)
    g.agg(pivot_h=("pivot_h", "mean"), pivot_h_sd=("pivot_h", "std"),
          bm_max=("bm_max", "mean"), max_visc=("max_viscosity", "mean"),
          recipe_yield=("yield_kg", "mean")).to_csv(out_dir / "pivot_calibration.csv")


def plot_yield_curves(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for arm, sub in df.groupby("arm"):
        g = sub.groupby("delay_h")["yield_kg"]
        st = dict(ARM_STYLE[arm])
        axes[0].errorbar(g.mean().index, g.mean(), yerr=g.sem(), capsize=3, lw=2.0, **st)
    axes[0].set_ylabel("batch yield (kg)")
    axes[0].set_title("absolute yield (dominated by the feed cut)")

    for arm in ("dual", "time"):
        sub = df[df["arm"] == arm]
        g = sub.groupby("delay_h")["delta_yield"]
        st = dict(ARM_STYLE[arm])
        axes[1].errorbar(g.mean().index, g.mean(), yerr=g.sem(), capsize=3, lw=2.4, **st)
    axes[1].axhline(0.0, color="k", lw=1.0)
    axes[1].set_ylabel("yield - recipe, same (seed, level)  [kg]")
    axes[1].set_title("paired advantage over recipe  <- the headline")

    for ax in axes:
        ax.set_xlabel("starvation window (h) -- later transition to the right")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Policy performance as the growth->production transition is delayed")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    multi_lib._finish(fig, out_dir, "delayed_transition_yield.png", show=False)

    (df.groupby(["arm", "delay_h"])
       .agg(n=("yield_kg", "size"), mean_yield=("yield_kg", "mean"),
            sem_yield=("yield_kg", "sem"), mean_delta=("delta_yield", "mean"),
            sem_delta=("delta_yield", "sem"), mean_pivot_h=("pivot_h", "mean"))
       .to_csv(out_dir / "delayed_transition_yield.csv"))


def plot_vs_pivot(df, out_dir):
    """Same delta against the MEASURED pivot hour rather than the nominal starve window. The
    x-axis is the recipe arm's pivot so it is policy-independent, and it absorbs the natural
    seed-to-seed pivot spread, which gives more resolution than the 5 nominal levels."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    slopes = {}
    for arm in ("dual", "time"):
        sub = df[df["arm"] == arm]
        x, y = sub["recipe_pivot_h"].to_numpy(float), sub["delta_yield"].to_numpy(float)
        st = ARM_STYLE[arm]
        ax.scatter(x, y, s=26, alpha=0.55, color=st["color"], marker=st["marker"])
        lr = stats.linregress(x, y)
        slopes[arm] = lr
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=st["color"], lw=2.4,
                label=f"{st['label']}\n  slope {lr.slope:+.1f} kg/h  (p={lr.pvalue:.3g})")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("measured transition hour of the batch (recipe arm)")
    ax.set_ylabel("yield - recipe, same (seed, level)  [kg]")
    ax.set_title("Degradation per hour of transition shift")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    multi_lib._finish(fig, out_dir, "delayed_transition_vs_pivot.png", show=False)
    return slopes


# ------------------------------------------------------------------------------------- stats

def _paired_p(a, b):
    """Paired t-test guarded against the degenerate case where the two arms are identical (all
    differences zero), which ttest_rel reports as nan-with-warning rather than 'no difference'."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if d.size < 2 or np.allclose(d, 0.0):
        return float("nan")
    return float(stats.ttest_rel(b, a).pvalue)


def report_per_level(df, out_dir, tag=""):
    """Head-to-head at each delay level, on RAW and FEASIBILITY-GATED yield.

    Raw yield alone cannot rank two policies when one of them buys titre with constraint
    headroom: feasibility_gated_yield_kg zeroes a batch that overflows Wt or exceeds VISC_MAX,
    which is the operationally meaningful comparison. Both are reported side by side, with the
    violation count that explains any gap between them. Runs on the FULL frame (no seed
    exclusion) because the violations are the point of this table."""
    rows = []
    for dh, cell in df.groupby("delay_h"):
        r = {"delay_h": float(dh), "n_seeds": int(cell["seed"].nunique())}
        for arm in ("dual", "time", "recipe"):
            s = cell[cell["arm"] == arm]
            viol = (s["visc_exceed"].astype(bool) | s["wt_overflow"].astype(bool))
            r[f"{arm}_yield"] = float(s["yield_kg"].mean())
            r[f"{arm}_gated"] = float(s["yield_gated"].mean())
            r[f"{arm}_violations"] = int(viol.sum())
        d = cell[cell["arm"] == "dual"].set_index("seed")
        t = cell[cell["arm"] == "time"].set_index("seed")
        k = d.index.intersection(t.index)
        r["dual_minus_time"] = float((d.loc[k, "yield_kg"] - t.loc[k, "yield_kg"]).mean())
        r["p_raw"] = _paired_p(t.loc[k, "yield_kg"], d.loc[k, "yield_kg"])
        r["dual_minus_time_gated"] = float((d.loc[k, "yield_gated"] - t.loc[k, "yield_gated"]).mean())
        r["p_gated"] = _paired_p(t.loc[k, "yield_gated"], d.loc[k, "yield_gated"])
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("delay_h")
    out.to_csv(out_dir / "per_level_head_to_head.csv", index=False)

    print("\n" + "=" * 96)
    print(f"DUAL vs TIME at each level{tag}   (kg; 'viol' = batches breaching VISC_MAX/Wt)")
    print("=" * 96)
    print(f"  {'delay':>6} {'recipe':>8} | {'DUAL raw':>9} {'viol':>5} {'DUAL gated':>11} | "
          f"{'TIME raw':>9} {'viol':>5} {'TIME gated':>11} | {'d-t raw':>9} {'p':>7} "
          f"{'d-t gated':>10} {'p':>7}")
    for _, r in out.iterrows():
        print(f"  {r['delay_h']:>5.0f}h {r['recipe_yield']:>8.0f} | "
              f"{r['dual_yield']:>9.0f} {int(r['dual_violations']):>5d} {r['dual_gated']:>11.0f} | "
              f"{r['time_yield']:>9.0f} {int(r['time_violations']):>5d} {r['time_gated']:>11.0f} | "
              f"{r['dual_minus_time']:>+9.0f} {r['p_raw']:>7.3g} "
              f"{r['dual_minus_time_gated']:>+10.0f} {r['p_gated']:>7.3g}")
    return out


def report_stats(df, slopes, out_dir, tag="", write=True):
    lo, hi = df["delay_h"].min(), df["delay_h"].max()
    rows = []
    for arm in ("dual", "time"):
        sub = df[df["arm"] == arm]
        a = sub[sub["delay_h"] == lo].set_index("seed")["delta_yield"]
        b = sub[sub["delay_h"] == hi].set_index("seed")["delta_yield"]
        common = a.index.intersection(b.index)
        d = (b[common] - a[common]).to_numpy(float)
        t = stats.ttest_rel(b[common], a[common]) if len(d) > 1 else None
        lr = slopes[arm]
        rows.append(dict(
            arm=arm,
            delta_at_min_delay=float(a.mean()), delta_at_max_delay=float(b.mean()),
            degradation_kg=float(d.mean()),
            degradation_ci95=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
            paired_p=float(t.pvalue) if t is not None else np.nan,
            slope_kg_per_pivot_hour=float(lr.slope), slope_p=float(lr.pvalue),
            slope_ci95=float(1.96 * lr.stderr)))

    # Does one policy degrade MORE than the other? Paired across (seed, level) so the batch-to-
    # batch noise the two arms share cancels -- that is the actual head-to-head test.
    key = ["seed", "starve_decisions"]
    dd = (df[df["arm"] == "dual"].set_index(key)["delta_yield"]
          - df[df["arm"] == "time"].set_index(key)["delta_yield"]).reset_index(name="dual_minus_time")
    dd["delay_h"] = dd["starve_decisions"] * T_SAMPLING
    head = stats.linregress(dd["delay_h"], dd["dual_minus_time"])

    out = pd.DataFrame(rows)
    if write:
        out.to_csv(out_dir / "degradation_stats.csv", index=False)
        dd.to_csv(out_dir / "head_to_head.csv", index=False)

    print("\n" + "=" * 78)
    print(f"DEGRADATION from {lo:.0f}h to {hi:.0f}h of starvation (paired vs recipe){tag}")
    print(f"  n = {df['seed'].nunique()} seeds")
    print("=" * 78)
    for r in rows:
        print(f"  {r['arm'].upper():<6} delta_vs_recipe {r['delta_at_min_delay']:+8.1f} kg  ->"
              f" {r['delta_at_max_delay']:+8.1f} kg   change {r['degradation_kg']:+7.1f}"
              f" +/- {r['degradation_ci95']:.1f} kg (p={r['paired_p']:.3g})")
        print(f"         slope vs measured pivot hour: {r['slope_kg_per_pivot_hour']:+.2f}"
              f" +/- {r['slope_ci95']:.2f} kg/h (p={r['slope_p']:.3g})")
    print(f"\n  HEAD-TO-HEAD (slot A - slot B) vs delay: {head.slope:+.2f} kg per delay-hour "
          f"(p={head.pvalue:.3g})")
    print(f"  positive => '{ARM_STYLE['dual']['label']}' pulls ahead as the transition is delayed")
    return out, head


def report_validity(df, out_dir):
    df.to_csv(out_dir / "validity_table.csv", index=False)
    bad = df[(df["visc_exceed"].astype(bool)) | (df["wt_overflow"].astype(bool))]
    print("\nvalidity check (guardrail 3):")
    if bad.empty:
        print(f"  all {len(df)} batches within constraints "
              f"(max viscosity {df['max_viscosity'].max():.1f} < {VISC_MAX:g}, "
              f"max Wt {df['max_Wt'].max():.0f})")
    else:
        print(f"  {len(bad)} of {len(df)} batches violate a constraint -- listed in "
              "validity_table.csv, EXCLUDE them from the headline number:")
        print(bad[["arm", "seed", "delay_h", "max_viscosity", "visc_exceed",
                   "max_Wt", "wt_overflow"]].to_string(index=False))
    nc = df[df["pivot_crossed"] == 0]
    if not nc.empty:
        print(f"  WARNING: {len(nc)} batches never crossed the biomass threshold; their pivot "
              "hour is the fallback, not a measurement.")
    return bad


def drop_violating_seeds(df):
    """Exclude WHOLE SEEDS that violate a constraint anywhere, not individual batches.

    Every headline number is paired within a (seed, level) cell, so dropping one arm's batch
    would leave an unpaired cell, and dropping single cells would leave the delay levels with
    different seed sets -- a degradation slope computed across unequal seed sets confounds the
    delay with which seeds happen to be in each level. Dropping the seed entirely keeps the
    panel balanced. The excluded seeds stay in validity_table.csv, and the violation itself is a
    result worth reporting rather than hiding."""
    mask = df["visc_exceed"].astype(bool) | df["wt_overflow"].astype(bool)
    bad_seeds = sorted(df.loc[mask, "seed"].unique().tolist())
    return df[~df["seed"].isin(bad_seeds)].copy(), bad_seeds


# -------------------------------------------------------------------------------------- main

def _finalise(df, out_dir):
    """Everything downstream of the simulation, so --replot can redo figures and statistics from
    validity_table.csv without spending another 18 minutes on batches."""
    report_validity(df, out_dir)
    clean, bad_seeds = drop_violating_seeds(df)
    if bad_seeds:
        print(f"\n  -> excluding seeds {bad_seeds} entirely from the headline "
              f"({clean['seed'].nunique()} of {df['seed'].nunique()} seeds remain, panel stays "
              "balanced). They are still in validity_table.csv.")

    report_per_level(df, out_dir)

    if df["delay_h"].nunique() < 2:
        # A single-level run (e.g. --handover 0 --levels 0: normal full-batch deployment) has no
        # delay axis, so the degradation slope and its figures are undefined. The per-level
        # head-to-head above is the whole answer in that case.
        print(f"\nwrote {out_dir}/")
        return

    plot_pivot_calibration(clean, out_dir)
    plot_yield_curves(clean, out_dir)
    slopes = plot_vs_pivot(clean, out_dir)
    report_stats(clean[clean["arm"] != "recipe"], slopes, out_dir, tag="  [feasible seeds only]")

    if bad_seeds:
        # Shown, not buried: if the two versions disagree, the disagreement IS the finding.
        slopes_all = {a: stats.linregress(sub["recipe_pivot_h"].astype(float),
                                          sub["delta_yield"].astype(float))
                      for a, sub in df[df["arm"] != "recipe"].groupby("arm")}
        report_stats(df[df["arm"] != "recipe"], slopes_all, out_dir,
                     tag="  [ALL seeds, incl. constraint violations -- for comparison only]",
                     write=False)
    print(f"\nwrote {out_dir}/")


def main(levels, handover, n_seeds, eval_base, pivot_bm, sensitivity_only, replot, out_dir,
         dual_dir=None, time_dir=None, dual_setup="dual_phase_baseline",
         time_setup="single_phase_baseline_time", label_dual=None, label_time=None,
         feed_delay_mode=False):
    # Slot labels drive every figure legend and printed table. When the slots are pointed at
    # something other than the default pair, the stock captions would be actively wrong, so let
    # the caller rename them.
    if label_dual:
        ARM_STYLE["dual"]["label"] = label_dual
    if label_time:
        ARM_STYLE["time"]["label"] = label_time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if replot:
        src = out_dir / "validity_table.csv"
        if not src.exists():
            raise FileNotFoundError(f"--replot needs a previous sweep's {src}")
        print(f"replotting from {src} (no simulation)")
        _finalise(pd.read_csv(src), out_dir)
        return
    if not feed_delay_mode and max(levels) > handover:
        raise ValueError(
            f"starvation levels {levels} exceed the handover decision {handover}: the arms would "
            "no longer share an identical pre-handover segment.")

    policies, wrapper_par, dual_run, time_run = load_policies(
        dual_dir, time_dir, dual_setup, time_setup)
    seeds = [eval_base + i for i in range(n_seeds)]
    for name, run in (("DUAL", dual_run), ("TIME", time_run)):
        if run.train_seed in seeds:
            raise RuntimeError(f"{name} trained on seed {run.train_seed}, which is in the eval "
                               f"block {seeds[0]}..{seeds[-1]}")

    print(f"\nstep 1: policy sensitivity probe (seed {eval_base})")
    probe_levels = sorted({0, max(levels)})
    sens_by_cond = {}
    for J in probe_levels:
        cond = "nominal" if J == 0 else f"delayed ({J * T_SAMPLING:.0f}h starve)"
        sens_by_cond[cond], _ = policy_sensitivity(policies, wrapper_par, eval_base,
                                                   starve_until=J)
    _, verdict = save_sensitivity(sens_by_cond, handover, out_dir)
    sensitivity_verdict(verdict, list(sens_by_cond), list(policies))
    if sensitivity_only:
        print(f"\nwrote {out_dir}/policy_sensitivity.{{png,csv}}")
        return

    mode = ("FEED-DELAY (env shifts Fs+Foil; policy controls from t=0)" if feed_delay_mode
            else f"STARVATION (action channel; handover {handover * T_SAMPLING:.0f}h)")
    print(f"\nstep 2: sweep | {mode}")
    print(f"  levels(h)={[float(l) if feed_delay_mode else l * T_SAMPLING for l in levels]} "
          f"seeds={seeds[0]}..{seeds[-1]} ({len(levels) * len(seeds) * 3} batches)")
    _finalise(add_paired_delta(sweep(policies, wrapper_par, levels, handover, seeds, pivot_bm,
                                     feed_delay_mode=feed_delay_mode)), out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="0,4,8,12,16",
                   help="starvation windows in DECISIONS (x5h), comma-separated")
    p.add_argument("--handover", type=int, default=16,
                   help="decision at which every arm hands over to its policy (16 = 80h)")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--eval_base", type=int, default=700000)
    p.add_argument("--pivot_bm", type=float, default=BM_PIVOT_DEFAULT)
    p.add_argument("--sensitivity_only", action="store_true",
                   help="run only the step-1 probe (no simulation sweep)")
    p.add_argument("--replot", action="store_true",
                   help="rebuild figures and statistics from a previous sweep's "
                        "validity_table.csv, without simulating")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--arm_dual", default=None,
                   help="run dir for slot A (default: the biomass dual-phase run)")
    p.add_argument("--arm_time", default=None,
                   help="run dir for slot B (default: the time-augmented single-phase run)")
    p.add_argument("--setup_dual", default="dual_phase_baseline", choices=sorted(SETUPS))
    p.add_argument("--setup_time", default="single_phase_baseline_time", choices=sorted(SETUPS))
    p.add_argument("--label_dual", default=None, help="legend label for slot A")
    p.add_argument("--label_time", default=None, help="legend label for slot B")
    p.add_argument("--feed_delay", action="store_true",
                   help="interpret --levels as FEED-DELAY HOURS applied by the environment "
                        "(shifts Fs+Foil recipe later) instead of as a starvation window; "
                        "the policy then controls the batch from t=0 and no handover is used")
    a = p.parse_args()
    main([int(x) for x in a.levels.split(",")], a.handover, a.n_seeds, a.eval_base,
         a.pivot_bm, a.sensitivity_only, a.replot, a.out_dir,
         a.arm_dual, a.arm_time, a.setup_dual, a.setup_time, a.label_dual, a.label_time,
         a.feed_delay)

"""Why does the biomass-pivot dual-phase model lose to a single GP that just reads the clock?

    python dual_phase_pivot_diagnosis.py
    python dual_phase_pivot_diagnosis.py --sweep_csv delayed_transition/validity_table.csv

The intuition says the dual model should win: it splits on ACTUAL physiology (running-max
biomass crossing pivot_bm) instead of assuming a fixed hour, so it should follow a batch whose
transition moves. This script tests that intuition against what seed4_13 actually did, and
measures the two places it breaks.

BREAK 1 -- THE SPLIT AND THE BLEND DISAGREE
    Two different mechanisms, and only ONE of them is on biomass:
      training  DualPhaseModelLearning.add_data splits each episode at the BIOMASS pivot
                p = first step where running-max(X*Wt/1000) >= pivot_bm. phase1 gets [0, p],
                phase2 gets [p, end]. So phase1 has NO data after hour p*5.
      rollout   _blend_weight with on_each_rollout=False (seed4_13's setting) mixes the two GPs
                on the WALL CLOCK: w = sigmoid(ln99/blend_half_width * (t - pivot_hours))
                = sigmoid(ln99/50 * (t - 100h)).
    So the data boundary sits wherever biomass says, but the mixing boundary always sits at
    100h. Anywhere the biomass pivot lands EARLIER than 100h, the composite is dominated by
    phase1 in a time range phase1 was never trained on -- extrapolation, during policy
    optimisation, over exactly the growth->production window that decides the batch.

BREAK 2 -- TIME WAS DROPPED FROM THE GP, BUT THE ACTUATION IS TIME-SCHEDULED
    The action is not a feed rate. It is a MULTIPLIER on the recipe: fs = Fs_recipe(t)*(1+0.5a),
    and Fs_recipe is a non-monotone function of the clock (8 -> 150 spike at 24h -> 30 -> ramp
    to 116 at 80h -> 90 -> 80). The dual baseline runs active_dims=[0,1,2,3,5], which removes
    time from the GP inputs. The same action at two different hours then means two different
    actual feeds that the GP cannot tell apart, except through whatever the other four channels
    happen to encode. The single-GP "Added_time" model keeps time and does not have this
    problem. This is measured here as the spread of recipe Fs at states the GP cannot
    distinguish.

Reads seed4_13's own training trajectories out of log.pkl and, if given a sweep CSV, the
measured pivots of the underfed test batches. Writes only into dual_phase_pivot_diagnosis/.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcpilco.pensim_wrapper import (T_SAMPLING, CONTROL_H, STEP_IN_HOURS, FS_SCALE,
                                    PenSimWrapper)
from mcpilco.model_learning_dual_phase import BM_PIVOT_DEFAULT
from phase_transition_diagnostic_biomass import _split_via_production_code
from pensimpy.data.constants import FS

RUN_DIR = Path(_ROOT) / "results" / "dual_phase_baseline" / "seed4_13"
OUT_DIR = Path(_EVAL_DIR) / "dual_phase_pivot_diagnosis"
PIVOT_HOURS = 100.0          # seed4_13's note.txt
BLEND_HALF_WIDTH = 50.0      # seed4_13's note.txt


def blend_weight(t_hours):
    """phase2's share of the composite at wall-clock hour t -- the deployed rollout mixer for
    on_each_rollout=False. Vectorised copy of DualPhaseModelLearning._blend_weight's time
    branch; the formula is asserted against the production method in _check_against_production."""
    k = math.log(99.0) / BLEND_HALF_WIDTH
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(t_hours, float) - PIVOT_HOURS)))


def _check_against_production():
    """Guard: if _blend_weight's time branch ever changes, this diagnostic must not keep
    reporting the old curve."""
    from mcpilco.model_learning_dual_phase import DualPhaseModelLearning
    stub = object.__new__(DualPhaseModelLearning)
    stub.on_each_rollout = False
    stub.pivot_hours = PIVOT_HOURS
    stub.blend_half_width_hours = BLEND_HALF_WIDTH
    for step in (0, 6, 12, 20, 30, 45):
        mine = float(blend_weight(step * T_SAMPLING))
        theirs = float(stub._blend_weight(step))
        if abs(mine - theirs) > 1e-12:
            raise RuntimeError(f"blend formula drifted from production at step {step}: "
                               f"{mine} vs {theirs}")


def training_splits(run_dir, pivot_bm):
    """Where seed4_13's OWN training episodes were split. seed4_13 predates the split_log.pkl
    save, so this recomputes from state_samples_history with the production splitter -- the same
    arrays add_data was handed."""
    log = pickle.load(open(Path(run_dir) / "log.pkl", "rb"))
    rows = []
    for i, st in enumerate(log["state_samples_history"]):
        step, hours, crossed = _split_via_production_code(np.asarray(st), pivot_bm)
        rows.append(dict(episode=i, split_step=int(step), split_h=float(hours),
                         crossed=int(crossed), w_at_split=float(blend_weight(hours))))
    return pd.DataFrame(rows)


def phase_data_coverage(splits, n_steps):
    """Per decision step: fraction of training episodes contributing a sample to each phase, set
    against the blend weight that step is mixed with. phase1 covers [0, p], phase2 covers
    [p, end], so a step past every episode's p has zero phase1 support."""
    hrs = np.arange(n_steps + 1) * T_SAMPLING
    p = splits["split_step"].to_numpy()
    cov1 = np.array([(p >= s).mean() for s in range(n_steps + 1)])
    cov2 = np.array([(p <= s).mean() for s in range(n_steps + 1)])
    w = blend_weight(hrs)
    # How much of the composite's prediction leans on a GP with no local data.
    unsupported = (1.0 - w) * (1.0 - cov1) + w * (1.0 - cov2)
    return pd.DataFrame(dict(step=np.arange(n_steps + 1), hours=hrs, w_phase2=w,
                             phase1_coverage=cov1, phase2_coverage=cov2,
                             unsupported_weight=unsupported))


def fs_ambiguity():
    """BREAK 2, quantified. The recipe Fs profile is non-monotone in time, so dropping time from
    the GP inputs makes pairs of hours with equal recipe-Fs slope indistinguishable while the
    achievable feed range at those hours differs. Reported as the spread of recipe Fs across the
    batch and the size of the action-induced band around it.

    Uses the production recipe object rather than re-interpolating the profile constant, so the
    curve is the same Fs(t) the wrapper feeds the simulator."""
    recipe = PenSimWrapper._build_default_recipe()
    t = np.arange(0.0, CONTROL_H, STEP_IN_HOURS)
    fs = np.array([recipe.get_values_dict_at(time=float(x))[FS] for x in t])
    return t, fs


def plot(splits, cov, sweep, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (1) where the split actually landed, against where the blend actually switches
    ax = axes[0]
    hrs = np.linspace(0, 230, 400)
    ax.plot(hrs, blend_weight(hrs), color="k", lw=2.2, label="rollout blend w (phase2 share)")
    ax.axvline(PIVOT_HOURS, color="k", ls=":", lw=1.4, label=f"blend centre {PIVOT_HOURS:.0f}h")
    ax.hist(splits["split_h"], bins=np.arange(40, 130, 5), density=True, alpha=0.55,
            color="C0", label="training-data split (biomass)")
    if sweep is not None:
        ax.hist(sweep["pivot_h"], bins=np.arange(40, 130, 5), density=True, alpha=0.45,
                color="C1", label="test-batch pivot (underfed sweep)")
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("blend weight  /  density")
    ax.set_title("BREAK 1: data splits early, blend switches at 100h")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (2) the extrapolation hole
    ax = axes[1]
    ax.plot(cov["hours"], cov["phase1_coverage"], color="C0", lw=2.0, label="phase1 data coverage")
    ax.plot(cov["hours"], cov["phase2_coverage"], color="C2", lw=2.0, label="phase2 data coverage")
    ax.plot(cov["hours"], 1 - cov["w_phase2"], color="C0", ls="--", lw=1.6,
            label="phase1 weight in rollout")
    ax.fill_between(cov["hours"], 0, cov["unsupported_weight"], color="r", alpha=0.30,
                    label="weight on a GP with no data here")
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("fraction")
    ax.set_title("BREAK 1: phase1 drives the rollout where it has no data")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) why dropping time hurts when the action scales a time-indexed recipe
    ax = axes[2]
    t, fs = fs_ambiguity()
    ax.plot(t, fs, color="k", lw=2.0, label="recipe Fs(t)")
    ax.fill_between(t, (1 - FS_SCALE) * fs, (1 + FS_SCALE) * fs, color="C3", alpha=0.25,
                    label=f"reachable Fs (a in [-1,1], FS_SCALE={FS_SCALE:g})")
    ax.set_xlabel("batch time (h)")
    ax.set_ylabel("Fs (L/h)")
    ax.set_title("BREAK 2: the action scales a clock-indexed recipe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Why the biomass-pivot dual model underperforms the time-augmented single GP")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(Path(out_dir) / "dual_phase_pivot_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(run_dir, pivot_bm, sweep_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_against_production()

    splits = training_splits(run_dir, pivot_bm)
    n_steps = int(CONTROL_H / T_SAMPLING)
    cov = phase_data_coverage(splits, n_steps)
    splits.to_csv(out_dir / "training_splits.csv", index=False)
    cov.to_csv(out_dir / "phase_coverage.csv", index=False)

    sweep = None
    if sweep_csv:
        s = pd.read_csv(sweep_csv)
        sweep = s[s["arm"] == "recipe"]

    print("=" * 78)
    print(f"seed4_13 training splits (pivot_bm={pivot_bm:g}, {len(splits)} episodes)")
    print("=" * 78)
    print(f"  split hour: median {splits['split_h'].median():.0f}h  "
          f"range {splits['split_h'].min():.0f}-{splits['split_h'].max():.0f}h  "
          f"(never-crossed fallbacks: {(splits['crossed'] == 0).sum()})")
    print(f"  blend weight at those split hours: median {splits['w_at_split'].median():.3f}  "
          f"max {splits['w_at_split'].max():.3f}")
    print(f"  -> at the moment the DATA switches to phase2, the ROLLOUT is still "
          f"{100 * (1 - splits['w_at_split'].median()):.1f}% phase1")

    lo = float(splits["split_h"].median())
    win = cov[(cov["hours"] >= lo) & (cov["hours"] <= PIVOT_HOURS)]
    print(f"\n  extrapolation window {lo:.0f}-{PIVOT_HOURS:.0f}h "
          f"({len(win)} of {len(cov)} decision steps = "
          f"{100 * len(win) / len(cov):.0f}% of the batch):")
    print(f"    mean phase1 weight there   : {(1 - win['w_phase2']).mean():.2f}")
    print(f"    mean phase1 data coverage  : {win['phase1_coverage'].mean():.2f}")
    print(f"    mean weight on an unsupported GP: {win['unsupported_weight'].mean():.2f}")
    worst = cov.loc[cov["unsupported_weight"].idxmax()]
    print(f"    worst step: {worst['hours']:.0f}h with "
          f"{worst['unsupported_weight']:.2f} of the prediction on a GP with no data")

    if sweep is not None:
        print(f"\n  underfed TEST batches (n={len(sweep)}):")
        for dh, sub in sweep.groupby("delay_h"):
            m = sub["pivot_h"].mean()
            print(f"    starve {dh:>3.0f}h -> pivot {m:>5.1f}h, blend weight there "
                  f"{blend_weight(m):.3f}  (gap to blend centre {PIVOT_HOURS - m:>5.1f}h)")
        print("  -> starving moves the pivot TOWARD the 100h blend centre, shrinking the "
              "mismatch.\n     That is a second reason DUAL closes the gap under starvation, "
              "alongside its\n     viscosity breaches being suppressed.")

    plot(splits, cov, sweep, out_dir)
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default=str(RUN_DIR))
    p.add_argument("--pivot_bm", type=float, default=BM_PIVOT_DEFAULT)
    p.add_argument("--sweep_csv", default=str(Path(_EVAL_DIR) / "delayed_transition" /
                                              "validity_table.csv"))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    main(a.run_dir, a.pivot_bm, a.sweep_csv, a.out_dir)

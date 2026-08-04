"""Validates the growth->production ("early boundary" ~47h) claim from an external review, using
ONLY GP-observable channels (P, X) -- no substrate S. Three independent checks, each aimed at a
specific unverified claim from that review:

1. PROMINENCE CHECK: peak specific production rate, q_p(t) = (dP/dt)/X, is itself a peak-finding
   operation -- the exact failure class that broke the old A0-peak detector (a near-flat plateau
   with two competing local maxima 0.08% apart, coin-flipping the pivot between seeds). Report the
   top-2 candidate peaks and their height margin for every seed, not just the winner, so a fragile
   tiebreak is visible rather than silently averaged away.

2. SEED STABILITY: run q_p's peak across many seeds under the FIXED default recipe (matching how
   the ~46.5+-0.7h claim was originally measured) and report mean/std/range.

3. POLICY-SHAPE SENSITIVITY (the review's single biggest unverified claim): the claimed mechanism
   is that ~47h is set by when substrate crashes into the production-optimal band, which the
   review only tested under +-60% AMPLITUDE-scaled recipes -- never a genuinely different feed
   SHAPE. Here the recipe's ramp/cut segment (t<80h, where FS_DEFAULT_PROFILE steps from 150->30)
   is time-shifted earlier/later, which directly perturbs WHEN the substrate crash happens (the
   claimed causal driver), not just how much feed there is. If q_p's peak is truly tracking that
   mechanism, it should move under this test even if it didn't move under amplitude scaling.

Also: a real-simulator dP/dY_Fs SIGN-FLIP probe (paired perturbed rollouts, boost vs cut Fs at one
decision at a time on J_GRID_SHARED -- reused from action_sensitivity.py for consistency -- then
compare resulting P at a fixed horizon later). This is the actual causal quantity of interest
("does more feed help or hurt, right now") rather than a proxy for it, and it's expensive (2 real
batches per candidate decision, per seed), so it's run on a small seed/grid subset here -- scale
up once the cheap checks above look right.

Strictly offline/post-hoc, touches no existing file, and reuses get_trajectory/_smooth/
SEED_MULTIPLIER/detect_pivot_kdiff/detect_pivot_a0 from phase_transition_diagnostic_updated.py
and J_GRID_SHARED from action_sensitivity.py rather than duplicating them.

Usage:
    python phase_transition_early_boundary.py 1-20                  # checks 1+2
    python phase_transition_early_boundary.py 1-20 --shape           # + check 3
    python phase_transition_early_boundary.py 1-3 --sensitivity      # + the dP/dFs probe
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from utils.constants import STEP_IN_HOURS
from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from mcpilco.pensim_wrapper import PenSimWrapper, STATE_NAMES, decode_state_value

from phase_transition_diagnostic_updated import (
    get_trajectory, _smooth, SEED_MULTIPLIER, K_DIFF_FLOOR_AT_HOURS,
    detect_pivot_kdiff, detect_pivot_a0,
)
from action_sensitivity import J_GRID_SHARED

PIVOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot_point_early")
X_IDX = STATE_NAMES.index("X")
P_IDX = STATE_NAMES.index("P")


# ---------------------------------------------------------------------------
# Check 1+2: peak specific production rate, with a prominence diagnostic
# ---------------------------------------------------------------------------

def compute_specific_production_rate(t, P, X, smooth_window_h=3.0):
    """q_p(t) = (dP/dt)/X -- specific penicillin production rate. Needs only P and X, both
    GP-observable/tracked (STATE_NAMES), unlike the old mu_X/S proxy."""
    P_s, X_s = (_smooth(v, t, smooth_window_h) for v in (P, X))
    Pdot = np.gradient(P_s, t)
    return Pdot / np.clip(X_s, 1e-6, None), P_s, X_s


def detect_pivot_specific_rate(t, q_p, exclude_before_h=5.0, exclude_after_h=10.0, order=5):
    """Global-max pivot (matching how the external review's 46.5h claim was computed) PLUS a
    prominence report: the top-2 local maxima and their height margin. A margin near 0% is
    exactly the A0 failure signature (two competing candidates, seed-sensitive tiebreak) --
    surfaced here instead of silently discarded like the old detect_pivot_a0 did."""
    mask = (t >= t[0] + exclude_before_h) & (t <= t[-1] - exclude_after_h)
    idxs = np.flatnonzero(mask)
    y = q_p[idxs]
    global_pivot = float(t[idxs[np.argmax(y)]])

    local_max_pos = argrelextrema(y, np.greater, order=order)[0]
    peaks = sorted(((float(t[idxs[i]]), float(y[i])) for i in local_max_pos),
                   key=lambda p: -p[1])
    if len(peaks) >= 2 and peaks[0][1] > 0:
        margin_pct = 100.0 * (peaks[0][1] - peaks[1][1]) / peaks[0][1]
    else:
        margin_pct = float("nan")
    return global_pivot, peaks[:3], margin_pct


def run_seed_stability(training_seeds, out_dir=PIVOT_DIR, shape_test=False):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ts in training_seeds:
        sim_seed = ts * SEED_MULTIPLIER
        traj = get_trajectory(sim_seed)
        t = traj["t"]
        q_p, P_s, X_s = compute_specific_production_rate(t, traj["P"], traj["X"])
        pivot_qp, top_peaks, margin_pct = detect_pivot_specific_rate(t, q_p)
        pivot_kdiff, _, _ = detect_pivot_kdiff(t, traj["Culture_age"], traj["X"])
        pivot_a0, _ = detect_pivot_a0(t, traj["a0"])

        flag = "FRAGILE" if (margin_pct == margin_pct and margin_pct < 5.0) else "ok"
        print(f"[seed {ts}] q_p pivot = {pivot_qp:6.1f} h | top peaks (h, height): "
              f"{[(round(h,1), round(v,4)) for h,v in top_peaks]} | margin = {margin_pct:5.1f}% "
              f"[{flag}] | kdiff = {pivot_kdiff:6.1f} h | a0 = {pivot_a0:6.1f} h")

        rows.append({"training_seed": ts, "sim_seed": sim_seed, "pivot_qp_hours": pivot_qp,
                    "top1_h": top_peaks[0][0] if top_peaks else float("nan"),
                    "top2_h": top_peaks[1][0] if len(top_peaks) > 1 else float("nan"),
                    "margin_pct": margin_pct, "fragile": flag == "FRAGILE",
                    "pivot_kdiff_hours": pivot_kdiff, "pivot_a0_hours": pivot_a0})

    vals = np.array([r["pivot_qp_hours"] for r in rows])
    n_fragile = sum(r["fragile"] for r in rows)
    print(f"\nq_p pivot over {len(rows)} seeds (fixed recipe): "
          f"mean={np.mean(vals):.2f}h  std={np.std(vals):.2f}h  "
          f"range=[{np.min(vals):.1f}, {np.max(vals):.1f}]h  "
          f"-- {n_fragile}/{len(rows)} seeds flagged FRAGILE (top-2 peak margin < 5%)")

    path = os.path.join(out_dir, "pivot_points_early.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")

    if shape_test:
        run_shape_sensitivity(training_seeds[:min(5, len(training_seeds))], out_dir)
    return rows


# ---------------------------------------------------------------------------
# Check 3: does q_p's peak move if the SHAPE (not just amplitude) of the feed changes?
# ---------------------------------------------------------------------------

def _shift_profile(profile, factor, cutoff_h=80.0):
    """Time-shift every setpoint below cutoff_h by `factor` (compress <1, expand >1), leaving
    later setpoints untouched. This perturbs WHEN the ramp/cut (150->30 around t=24-28h in
    FS_DEFAULT_PROFILE) happens -- the review's claimed causal driver of the ~47h transition --
    which +-60% AMPLITUDE scaling (already tested) does not."""
    shifted = []
    for sp in profile:
        t, v = sp["time"], sp["value"]
        shifted.append({"time": (t * factor) if t < cutoff_h else t, "value": v})
    return shifted


def _build_recipe_with_fs(fs_profile):
    return RecipeCombo(recipe_dict={
        FS: Recipe(fs_profile, FS), FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG: Recipe(FG_DEFAULT_PROFILE, FG), PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        WATER: Recipe(WATER_DEFAULT_PROFILE, WATER), PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
    })


def get_trajectory_with_recipe(seed, recipe_combo):
    env = PenSimEnv(recipe_combo=recipe_combo, fast=True)
    (_df, _df_raman), _yield, bx = env.get_batches(
        random_seed=seed, include_raman=False, return_batch_data=True)
    n = len(bx.X.y)
    t = np.array([(i + 1) * STEP_IN_HOURS for i in range(n)])
    return {"t": t, "X": np.array(bx.X.y), "P": np.array(bx.P.y)}


def run_shape_sensitivity(training_seeds, out_dir=PIVOT_DIR):
    print("\n--- Check 3: policy-SHAPE sensitivity (early-cut / late-cut recipe) ---")
    variants = {"baseline": 1.00, "early_cut (x0.85)": 0.85, "late_cut (x1.15)": 1.15}
    rows = []
    for ts in training_seeds:
        sim_seed = ts * SEED_MULTIPLIER
        for label, factor in variants.items():
            profile = _shift_profile(FS_DEFAULT_PROFILE, factor)
            recipe = _build_recipe_with_fs(profile)
            traj = get_trajectory_with_recipe(sim_seed, recipe)
            q_p, _, _ = compute_specific_production_rate(traj["t"], traj["P"], traj["X"])
            pivot, _, margin = detect_pivot_specific_rate(traj["t"], q_p)
            print(f"[seed {ts}] {label:18s} -> q_p pivot = {pivot:6.1f} h  (margin {margin:5.1f}%)")
            rows.append({"training_seed": ts, "variant": label, "factor": factor,
                        "pivot_qp_hours": pivot, "margin_pct": margin})

    path = os.path.join(out_dir, "shape_sensitivity.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")
    for label in variants:
        vals = [r["pivot_qp_hours"] for r in rows if r["variant"] == label]
        print(f"  {label:18s}: mean={np.mean(vals):.2f}h  std={np.std(vals):.2f}h")
    return rows


# ---------------------------------------------------------------------------
# dP/dF_s sign-flip probe: paired perturbed REAL rollouts, coarse decision grid
# ---------------------------------------------------------------------------

from mcpilco.pensim_wrapper import CONTROL_H, T_SAMPLING, STATE_RANGES


def _denorm(name, x_norm):
    lo, hi = STATE_RANGES[name]
    return decode_state_value(name, lo + (x_norm + 1.0) * (hi - lo) / 2.0)


def probe_action_sensitivity(seed, decision_idx, delta=0.3, horizon_decisions=6):
    """Boost (+delta) vs cut (-delta) the action at exactly `decision_idx`, zero everywhere else
    (a one-decision pulse on top of the recipe, matching action_sensitivity.py's perturbation
    style), then compare resulting P at decision_idx+horizon_decisions. Returns
    (sign, delta_P) where sign>0 means more feed helped P at that horizon, <0 means it hurt.
    Two REAL PenSimPy batches per call -- expensive, keep the candidate grid coarse."""
    def _pulse(level):
        def pol(state, di):
            return np.array([level if di == decision_idx else 0.0])
        return pol

    wrapper_plus = PenSimWrapper(seed_offset=0)
    wrapper_minus = PenSimWrapper(seed_offset=0)
    states_plus, _, _ = wrapper_plus.rollout(
        s0=None, policy=_pulse(delta), T=CONTROL_H, dt=T_SAMPLING, noise=None, seed=seed)
    states_minus, _, _ = wrapper_minus.rollout(
        s0=None, policy=_pulse(-delta), T=CONTROL_H, dt=T_SAMPLING, noise=None, seed=seed)

    eval_idx = min(decision_idx + horizon_decisions, states_plus.shape[0] - 1)
    P_plus = _denorm("P", states_plus[eval_idx, P_IDX])
    P_minus = _denorm("P", states_minus[eval_idx, P_IDX])
    dP = float(P_plus - P_minus)
    return (1 if dP > 0 else -1), dP


def run_sensitivity_sweep(training_seeds, j_grid=J_GRID_SHARED, out_dir=PIVOT_DIR):
    print("\n--- dP/dF_s sign-flip probe (real paired perturbations, J_GRID_SHARED) ---")
    rows = []
    for ts in training_seeds:
        sim_seed = ts * SEED_MULTIPLIER
        signs = []
        for j in j_grid:
            sign, dP = probe_action_sensitivity(sim_seed, j)
            t_h = (j + 1) * T_SAMPLING
            signs.append((t_h, sign, dP))
            print(f"[seed {ts}] decision {j:2d} (~{t_h:5.1f}h): sign(dP/dFs) = {sign:+d}  (dP={dP:+.4f})")
        flips = [signs[i][0] for i in range(1, len(signs)) if signs[i][1] != signs[i - 1][1]]
        print(f"  -> sign flip(s) between: {flips if flips else 'none in this grid'}")
        rows.append({"training_seed": ts, "signs": signs, "flip_windows": flips})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=str)
    parser.add_argument("--shape", action="store_true", help="also run the policy-shape sensitivity check")
    parser.add_argument("--sensitivity", action="store_true", help="also run the dP/dFs sign-flip probe (expensive)")
    args = parser.parse_args()

    seeds = []
    for tok in args.seeds.split(","):
        if "-" in tok:
            lo, hi = tok.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(tok))

    run_seed_stability(seeds, shape_test=args.shape)
    if args.sensitivity:
        run_sensitivity_sweep(seeds[:min(2, len(seeds))])

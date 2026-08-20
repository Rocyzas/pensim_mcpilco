"""Is penicillin feed-controllable in the production phase, and is that signal identifiable
from the 4-dim observed state?

    python production_controllability.py --seeds 0-19
    python production_controllability.py --seeds 0-4 --t_bump 90,130,170 --mags -100,-50,50,100

WHY
    Late-batch one-step dynamics are near-unpredictable for EVERY architecture tested
    (RMSE/SD ~ 0.95-1.00 on dX and dP for single-phase, dual-phase2, +time and -time alike).
    That is a fact about the problem, not the phase decomposition. This asks WHY, by separating
    two very different explanations that the fit statistic cannot distinguish:

    QUESTION A -- is P feed-controllable in production at all?
        Paired design: the SAME seed is run twice, identical in every respect except a feed bump
        inside the production window. Pairing removes the stochastic realisation entirely, so
        dP = P_perturbed - P_baseline is the feed effect and nothing else. If dP never clears the
        natural batch-to-batch spread of P, feed has no leverage there and no state augmentation
        or model structure can create any.

    QUESTION B -- if the signal exists, is it identifiable from the OBSERVED state?
        A response can be real yet conditional on variables the agent cannot see (S, DO2, a0,
        a1, PAA), in which case it looks like noise at fixed observed state -- exactly the
        symptom the GP shows. Regressing dP on the observed state {Wt, X, P, Viscosity} and then
        on observed + hidden separates "no signal" from "signal gated by hidden state", and
        names which variable recovers it.

DESIGN NOTES
    * Bump magnitude is expressed as a PERCENTAGE of recipe feed. To reach +-100% the module
      constant FS_SCALE is set to 1.0 for this script only (it is read at call time inside
      fs_from_action), so a = +-1.0 spans +-100% and a = +-0.5 spans +-50%. Baseline is a = 0,
      which is the recipe exactly and is unaffected by FS_SCALE. Nothing is written back.
    * Bump timing is swept, not fixed: feed authority over P plausibly decays through production,
      and "controllable early, not late" is a different and more useful finding than one number.
      It also shows whether the fixed 100h pivot sits where controllability actually dies.
    * Mediating channels (S, DO2, Viscosity, X, a0/a1) are recorded alongside P, to distinguish
      "feed has no leverage" from "feed has leverage that a constraint cancels" -- e.g. feed ->
      viscosity -> O2 transfer collapse -> production choked. The second says production IS
      controllable, but through aeration, an actuator the agent was never given.
    * Discharge/aeration/pressure fire on the wall clock and are IDENTICAL in both members of a
      pair, so they are common-mode and cancel in dP.

SCOPE
    A simulator result. An absent signal means absent in IndPenSim's production model
    (Bajpai-Reuss / Paul-Thomas kinetics), which is a meaningful statement about the modelled
    biology but not a claim about a real bioreactor. State that scope when citing it.

Read-only w.r.t. results/: writes only into production_controllability/.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)
_MCPILCO = _os.path.join(_os.path.dirname(_ROOT), "MC-PILCO")
if _MCPILCO not in _sys.path:
    _sys.path.insert(0, _MCPILCO)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mcpilco.pensim_wrapper as pw
from mcpilco.pensim_wrapper import PenSimWrapper, T_SAMPLING, STEPS_PER_DECISION, K_WARM
from utils.peni_env_setup import PenSimEnv
from utils.constants import NUM_STEPS, STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA

# a = +-1 must mean +-100% of recipe feed for the largest bump. Read at call time by
# fs_from_action, so setting it here is enough; a = 0 (baseline) is the recipe regardless.
pw.FS_SCALE = 1.0

OUT_DIR = Path(_EVAL_DIR) / "production_controllability"
OBS = ["Wt", "X", "P", "Viscosity"]                     # what the agent actually sees
HID = ["S", "DO2", "a0", "a1", "PAA", "CER", "mu_P_calc"]   # what it does not


def run_batch(seed, bump_start_h=None, bump_pct=0.0, bump_hours=20.0):
    """One batch. `bump_pct` is a percentage of recipe Fs held for `bump_hours` from
    `bump_start_h`; outside that window the action is 0, i.e. exactly the recipe.

    Returns the per-timestep channel dict, so the mediating path (S/DO2/Viscosity) can be read
    as well as P."""
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    np.random.seed(seed)
    env.random_seed_ref = seed
    _, bx = env.reset()
    recipe = PenSimWrapper._build_default_recipe()
    lo = bump_start_h if bump_start_h is not None else np.inf
    hi = lo + bump_hours
    for k in range(1, NUM_STEPS + 1):
        t = k * STEP_IN_HOURS
        v = recipe.get_values_dict_at(time=t)
        a = (bump_pct / 100.0) if (lo <= t < hi) else 0.0
        fs = v[FS] * (1.0 + pw.FS_SCALE * a)
        env.bypass_paa_pid = False
        _, bx, _, _ = env.step(k, bx, Fs=fs, Foil=v[FOIL], Fg=v[FG], pressure=v[PRES],
                               discharge=v[DISCHARGE], Fw=v[WATER], Fpaa=v[PAA])
    ch = {"t": np.arange(1, NUM_STEPS + 1) * STEP_IN_HOURS}
    for name in ["P", "X", "S", "Wt", "V", "Viscosity", "DO2", "a0", "a1", "PAA", "CER",
                 "OUR", "mu_P_calc", "Fs"]:
        ch[name] = np.asarray(getattr(bx, name).y, dtype=float)
    return ch


def at(ch, name, hour):
    return float(np.interp(hour, ch["t"], ch[name]))


def main(seeds, t_bumps, mags, bump_hours, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        base = run_batch(s)
        for tp in t_bumps:
            for m in mags:
                pert = run_batch(s, bump_start_h=tp, bump_pct=m, bump_hours=bump_hours)
                r = dict(seed=s, t_bump=tp, bump_pct=m, bump_hours=bump_hours)
                # response, measured at several horizons after the bump
                for h in (10, 30, 50):
                    if tp + h <= 230:
                        r[f"dP_+{h}h"] = at(pert, "P", tp + h) - at(base, "P", tp + h)
                r["dP_final"] = pert["P"][-1] - base["P"][-1]
                r["P_final_base"] = base["P"][-1]
                # mediating path: how far did the intervention actually move the plant?
                for nm in ("S", "DO2", "Viscosity", "X", "CER"):
                    r[f"d{nm}_+10h"] = at(pert, nm, tp + 10) - at(base, nm, tp + 10)
                r["dFs_integral"] = float(np.trapezoid(pert["Fs"] - base["Fs"], pert["t"]))
                # conditioning state at the moment of the bump, from the BASELINE run: what the
                # agent could have known when choosing the action
                for nm in OBS + HID:
                    r[f"state_{nm}"] = at(base, nm, tp)
                rows.append(r)
        print(f"  seed {s}: {len(t_bumps) * len(mags)} perturbations done", flush=True)
    df = pd.DataFrame(rows)
    tag = f"{seeds[0]}_{seeds[-1]}"
    df.to_csv(out_dir / f"controllability_seeds{tag}.csv", index=False)
    print(f"wrote {out_dir}/controllability_seeds{tag}.csv  ({len(df)} rows)")


def _parse_range(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 0,1,2")
    p.add_argument("--t_bump", default="90,130,170", help="bump start hours")
    p.add_argument("--mags", default="-100,-50,50,100", help="bump size, %% of recipe Fs")
    p.add_argument("--bump_hours", type=float, default=20.0)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    main(_parse_range(a.seeds), [float(x) for x in a.t_bump.split(",")],
         [float(x) for x in a.mags.split(",")], a.bump_hours, a.out_dir)

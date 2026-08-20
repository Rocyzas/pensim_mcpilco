"""Which inhibition multiplier limits penicillin production, and does it collapse under high feed?

    python inhibition_terms.py --seeds 0-4 --bump_pct 100 --t_bump 130 --bump_hours 20

WHY THIS IS OBSERVATION, NOT INFERENCE
    IndPenSim factorises both rates as products of independent multipliers (verified against
    PenSimPy/pensimpy/ode/indpensim_ode_py.py):

      line 235  mu_e = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
      line 268  r_p  = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P  -  mu_h*y[3]

    so logging each factor per timestep identifies the limiting one directly. NOTE r_p carries a
    factor the usual write-up omits: P_inhib (line 229) is a GAUSSIAN in substrate,
        P_inhib = 2.5*sd * (sd*sqrt(2pi))^-1 * exp(-0.5*((s-mean_P)/sd)^2),  mean_P=0.002, sd=0.0015
    i.e. production is maximised at s = 0.002 g/L and falls off on BOTH sides -- not a
    monotone Monod tail. Where the batch sits on that bell decides the sign AND magnitude of
    d(production)/d(feed), so it is logged explicitly.

HOW THE TERMS ARE OBTAINED
    Not re-derived. The ODE's own source text is read, a single capture line is inserted
    immediately before `return dy`, and the result is exec'd. The body is byte-identical to the
    shipped file, and _verify_instrumentation() asserts the instrumented function returns exactly
    the same dy as the original on random states before any batch is run. If that assertion ever
    fails the script stops rather than reporting numbers from a diverged copy.

PAIRED DESIGN
    Same seed, nominal vs bumped feed, exactly as in production_controllability.py. The simulator
    is bit-deterministic given a seed, so every difference in a multiplier is caused by the feed.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.dirname(_ROOT), _os.path.join(_os.path.dirname(_ROOT), "MC-PILCO")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, math, re, textwrap
from pathlib import Path
import numpy as np
import pandas as pd

ODE_SRC = Path(_os.path.dirname(_ROOT)) / "PenSimPy/pensimpy/ode/indpensim_ode_py.py"
OUT_DIR = Path(_os.path.dirname(_os.path.abspath(__file__))) / "inhibition_terms"

# every multiplier in the two factorisations, plus the quantities that set them
TERMS = ["s", "P_inhib", "DO_2_inhib_P", "PAA_inhib_P", "mu_h", "rho_a0", "v_a1", "mu_p",
         "pH_inhib", "NH3_inhib", "T_inhib", "DO_2_inhib_X", "CO2_inhib", "PAA_inhib_X",
         "mux_max", "r_p", "mu_e", "pH", "total_pressure", "O_2_in", "Henrys_c",
         "P_crit_DO2", "X_crit_DO2", "mean_P", "P_std_dev"]

_CAPTURE = []


def _build_instrumented():
    src = ODE_SRC.read_text()
    lines = src.splitlines()
    idx = max(i for i, l in enumerate(lines) if l.strip() == "return dy")
    cap = ("    _CAPTURE.append({'t': t, 'DO2': y[1], 'P': y[3], "
           + ", ".join(f"'{k}': locals().get('{k}')" for k in TERMS) + "})")
    lines.insert(idx, cap)
    ns = {"math": math, "_CAPTURE": _CAPTURE}
    exec(compile("\n".join(lines), str(ODE_SRC) + "<instrumented>", "exec"), ns)
    return ns["indpensim_ode_py"]


def _verify_instrumentation(inst):
    from PenSimPy.pensimpy.ode.indpensim_ode_py import indpensim_ode_py as orig
    rng = np.random.default_rng(0)
    from utils.peni_env_setup import PenSimEnv
    from mcpilco.pensim_wrapper import PenSimWrapper
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    _, bx = env.reset()
    par = list(env.param_list) + [15.0, 22.0, 60.0, 0.6, 0.0, 0.0, 5.0, 296.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0]
    ok = True
    for _ in range(20):
        y = list(np.abs(rng.normal(1.0, 0.4, 33)) + 1e-3)
        t = float(rng.uniform(1, 230))
        try:
            a = orig(t, y, par); b = inst(t, y, par)
        except Exception:
            continue
        if not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=0):
            ok = False
            break
    _CAPTURE.clear()
    if not ok:
        raise RuntimeError("instrumented ODE diverged from the shipped one -- refusing to run")
    print("  instrumentation verified: instrumented dy == original dy (exact) on random states")


def run(seed, bump_pct=0.0, t_bump=None, bump_hours=20.0):
    """One batch; returns a DataFrame of every ODE evaluation's multipliers."""
    import utils.ode_patch as op
    from utils.peni_env_setup import PenSimEnv
    from mcpilco.pensim_wrapper import PenSimWrapper
    from utils.constants import NUM_STEPS, STEP_IN_HOURS
    from PenSimPy.pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
    _CAPTURE.clear()
    env = PenSimEnv(recipe_combo=PenSimWrapper._build_default_recipe(), fast=True)
    np.random.seed(seed); env.random_seed_ref = seed
    _, bx = env.reset()
    recipe = PenSimWrapper._build_default_recipe()
    lo = t_bump if t_bump is not None else np.inf
    hi = lo + bump_hours
    for k in range(1, NUM_STEPS + 1):
        t = k * STEP_IN_HOURS
        v = recipe.get_values_dict_at(time=t)
        a = (bump_pct / 100.0) if (lo <= t < hi) else 0.0
        env.bypass_paa_pid = False
        _, bx, _, _ = env.step(k, bx, Fs=v[FS] * (1.0 + a), Foil=v[FOIL], Fg=v[FG],
                               pressure=v[PRES], discharge=v[DISCHARGE], Fw=v[WATER],
                               Fpaa=v[PAA])
    df = pd.DataFrame(_CAPTURE).sort_values("t")
    _CAPTURE.clear()
    return df.groupby(df.t.round(1)).last().reset_index(drop=True)


def main(seeds, bump_pct, t_bump, bump_hours, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import utils.ode_patch as op
    inst = _build_instrumented()
    _verify_instrumentation(inst)
    op.indpensim_ode_py = inst           # ode_patch's LSODA wrapper calls this name
    rows = []
    for s in seeds:
        base = run(s)
        pert = run(s, bump_pct=bump_pct, t_bump=t_bump, bump_hours=bump_hours)
        n = min(len(base), len(pert))
        b, p = base.iloc[:n].reset_index(drop=True), pert.iloc[:n].reset_index(drop=True)
        d = pd.DataFrame({"seed": s, "t": b.t})
        for c in TERMS + ["DO2", "P"]:
            if c in b:
                d[f"base_{c}"] = b[c]; d[f"pert_{c}"] = p[c]
        rows.append(d)
        print(f"  seed {s} done ({n} logged timesteps)", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / f"terms_bump{int(bump_pct)}_t{int(t_bump)}.csv", index=False)
    print(f"wrote {out_dir}/terms_bump{int(bump_pct)}_t{int(t_bump)}.csv ({len(df)} rows)")


def _rng(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--bump_pct", type=float, default=100.0)
    p.add_argument("--t_bump", type=float, default=130.0)
    p.add_argument("--bump_hours", type=float, default=20.0)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    main(_rng(a.seeds), a.bump_pct, a.t_bump, a.bump_hours, a.out_dir)

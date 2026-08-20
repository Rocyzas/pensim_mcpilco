"""Does feed authority over final penicillin COLLAPSE across the growth->production transition?

    python growth_vs_production_authority.py --seeds 0-19 --t_inject 30,50,70,90,130,170

The existing perturbation study injected only at 90/130/170h -- all at or after the transition
(~63h in the undelayed recipe). It shows authority is weak IN production; it cannot show whether
authority is weak EVERYWHERE. This adds pre-transition injections and the instrumentation needed
to say why.

SAME HARNESS AS THE EXISTING STUDY
    Injection shape, magnitudes, pairing and solver are unchanged -- run_batch is imported from
    production_controllability rather than reimplemented, so "the only new thing is WHEN and WHAT
    IS LOGGED" holds literally. The step is held for --bump_hours (default 20h), which is what
    the existing study used; continuity matters more here than the word "step".

WHAT IS MEASURED (and why it is measurement, not inference)
  * Headline is penicillin MASS, P*V/1000, not concentration: the injection changes volume, so a
    concentration delta conflates production with dilution.
  * dP_mass(t) is kept over the WHOLE remaining batch. Its shape separates two hypotheses that a
    single end-of-batch number cannot: HOLD (genuine extra product) vs RISE-THEN-DECAY (the same
    substrate budget spent earlier). Reported as final/peak ratio.
  * a0/a1 are the true ODE states (peni_env_setup writes x.a0/x.a1 straight from y_sol; the
    delayed copies carry an _offline suffix), so "did the step create new product-forming A1
    biomass" is read directly.
  * Substrate partition uses the model's own coefficients: bolus = integral(dFs)*c_s with
    c_s=600 g/L, routed against Y_sX=1.85 (biomass) and Y_sP=0.9 (penicillin) from the ODE.
  * s is logged so each injection can be placed against the Gaussian P_inhib peak at
    s = 0.002 g/L (mean_P), which is what sets the SIGN of the response.

COLLAPSES ARE RECORDED, NOT DROPPED -- a batch that violates viscosity/Wt is itself evidence of
authority, and matters most for the growth-phase claim.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.dirname(_ROOT), _os.path.join(_os.path.dirname(_ROOT), "MC-PILCO")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
from pathlib import Path
import numpy as np, pandas as pd

from evaluations.production_controllability import run_batch, at   # SAME harness, unchanged

OUT_DIR = Path(_os.path.dirname(_os.path.abspath(__file__))) / "growth_vs_production"
C_S = 600.0        # g/L substrate in the feed        (indpensim_ode_py.py:41)
Y_SX, Y_SP = 1.85, 0.9   # substrate per biomass / per penicillin   (lines 37-38)
MEAN_P = 0.002     # Gaussian P_inhib peak in s                      (line 17)
VISC_MAX, WT_MAX = 100.0, 1.2e5


def mass(ch, name):
    return np.asarray(ch[name]) * np.asarray(ch["V"]) / 1000.0


def summarise(b, p, seed, tp, pct):
    t = np.asarray(b["t"])
    dPm = mass(p, "P") - mass(b, "P")
    post = t >= tp
    peak_i = int(np.argmax(np.abs(dPm[post]))) + int(np.argmax(post))
    peak = dPm[peak_i]
    fin = dPm[-1]
    win = (t >= tp) & (t <= tp + 20)
    s_base = np.asarray(b["S"])[win]; s_pert = np.asarray(p["S"])[win]
    dFs = np.asarray(p["Fs"]) - np.asarray(b["Fs"])
    bolus_g = float(np.trapezoid(dFs, t)) * C_S
    dXm = (mass(p, "X") - mass(b, "X"))[-1] * 1000.0     # g
    dPm_g = fin * 1000.0                                  # g
    r = dict(seed=seed, t_inject=tp, pct=pct,
             phase=("growth" if tp < 63 else "transition" if tp < 80 else "production"),
             dP_mass_final=fin, dP_mass_peak=peak, peak_hour=float(t[peak_i]),
             hold_ratio=(fin / peak if peak else np.nan),
             dP_conc_final=p["P"][-1] - b["P"][-1],
             base_P_mass_final=float(mass(b, "P")[-1]),
             s_base_med=float(np.median(s_base)), s_pert_med=float(np.median(s_pert)),
             s_pert_max=float(np.max(s_pert)),
             s_crossed_peak=bool(np.median(s_pert) > MEAN_P),
             bolus_g=bolus_g, dX_mass_g=dXm, dP_mass_g=dPm_g,
             sub_to_biomass_g=Y_SX * dXm, sub_to_penicillin_g=Y_SP * dPm_g,
             da1_10h=at(p, "a1", min(tp + 10, 229)) - at(b, "a1", min(tp + 10, 229)),
             da1_30h=at(p, "a1", min(tp + 30, 229)) - at(b, "a1", min(tp + 30, 229)),
             da1_final=p["a1"][-1] - b["a1"][-1],
             da0_final=p["a0"][-1] - b["a0"][-1],
             base_a1_final=float(b["a1"][-1]),
             max_visc=float(np.max(p["Viscosity"])), max_Wt=float(np.max(p["Wt"])),
             collapsed=bool(np.max(p["Viscosity"]) > VISC_MAX or np.max(p["Wt"]) > WT_MAX))
    return r, dPm


def main(seeds, t_inject, pcts, bump_hours, traj_pct, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, trajs = [], []
    for s in seeds:
        base = run_batch(s)                       # regenerated under the current config
        for tp in t_inject:
            for m in pcts:
                pert = run_batch(s, bump_start_h=tp, bump_pct=m, bump_hours=bump_hours)
                r, dPm = summarise(base, pert, s, tp, m)
                rows.append(r)
                if m == traj_pct:
                    t = np.asarray(base["t"]); k = slice(None, None, 5)   # 1h resolution
                    trajs.append(pd.DataFrame({"seed": s, "t_inject": tp, "pct": m,
                                               "t": t[k], "dP_mass": dPm[k]}))
        print(f"  seed {s}: {len(t_inject)*len(pcts)} injections done", flush=True)
    tag = f"{seeds[0]}_{seeds[-1]}"
    pd.DataFrame(rows).to_csv(out_dir / f"authority_seeds{tag}.csv", index=False)
    pd.concat(trajs, ignore_index=True).to_csv(out_dir / f"traj_seeds{tag}.csv", index=False)
    print(f"wrote authority_seeds{tag}.csv ({len(rows)} rows) and traj_seeds{tag}.csv")


def _rng(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--t_inject", default="30,50,70,90,130,170")
    p.add_argument("--pcts", default="-50,-25,25,50")
    p.add_argument("--bump_hours", type=float, default=20.0)
    p.add_argument("--traj_pct", type=float, default=50.0)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    main(_rng(a.seeds), [float(x) for x in a.t_inject.split(",")],
         [float(x) for x in a.pcts.split(",")], a.bump_hours, a.traj_pct, a.out_dir)

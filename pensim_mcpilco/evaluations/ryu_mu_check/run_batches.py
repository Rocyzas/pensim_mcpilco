"""Run IndPenSim batches with per-step logging of the ODE's internal kinetic terms.

Writes one parquet/csv per (seed, fs_scale) into ./out/.
"""
import sys, os, copy, time
from pathlib import Path

ROOT = Path("/Users/rokaspranevicius/Documents/Aca/UniversityOfEdinburgh/MSc/pensimpy_mcpilco")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pensim_mcpilco"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from utils.ode_patch import patch_fastodeint
patch_fastodeint("lsoda")
import fastodeint

from utils.peni_env_setup import PenSimEnv
from utils.recipe import Recipe, RecipeCombo
from utils.constants import STEP_IN_HOURS, NUM_STEPS
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from ode_diag import indpensim_ode_diag

_records = []
_inner = fastodeint.integrate


def _integrate_logging(y0, par, t_start, t_end, h):
    y1 = _inner(y0, par, t_start, t_end, h)
    yc = [v if (np.isfinite(v) and v > 0) else 0.001 for v in y1[0:31]] + list(y1[31:])
    try:
        d = indpensim_ode_diag(t_end - h, yc, par)
    except Exception as e:                                   # noqa: BLE001
        d = {"t": t_end - h, "err": repr(e)}
    _records.append(d)
    return y1


fastodeint.integrate = _integrate_logging


def build_recipe(fs_scale=1.0):
    fs = [{"time": p["time"], "value": p["value"] * fs_scale} for p in FS_DEFAULT_PROFILE]
    return RecipeCombo(recipe_dict={
        FS: Recipe(fs, FS), FOIL: Recipe(copy.deepcopy(FOIL_DEFAULT_PROFILE), FOIL),
        FG: Recipe(copy.deepcopy(FG_DEFAULT_PROFILE), FG),
        PRES: Recipe(copy.deepcopy(PRESS_DEFAULT_PROFILE), PRES),
        DISCHARGE: Recipe(copy.deepcopy(DISCHARGE_DEFAULT_PROFILE), DISCHARGE),
        WATER: Recipe(copy.deepcopy(WATER_DEFAULT_PROFILE), WATER),
        PAA: Recipe(copy.deepcopy(PAA_DEFAULT_PROFILE), PAA),
    })


def run_one(seed, fs_scale=1.0):
    global _records
    _records = []
    env = PenSimEnv(recipe_combo=build_recipe(fs_scale), fast=True)
    np.random.seed(seed)
    t0 = time.time()
    df, batch_yield, bx = env.get_batches(random_seed=seed, return_batch_data=True)
    diag = pd.DataFrame(_records)
    # channels PenSimPy logs itself (post-processed pH/Q), aligned on the same k index
    n = len(diag)
    logged = pd.DataFrame({
        "k": np.arange(1, n + 1),
        "time_h": np.arange(1, n + 1) * STEP_IN_HOURS,
        "X_log": np.asarray(bx.X.y[:n], dtype=float),
        "P_log": np.asarray(bx.P.y[:n], dtype=float),
        "S_log": np.asarray(bx.S.y[:n], dtype=float),
        "V_log": np.asarray(bx.V.y[:n], dtype=float),
        "Wt_log": np.asarray(bx.Wt.y[:n], dtype=float),
        "Visc_log": np.asarray(bx.Viscosity.y[:n], dtype=float),
        "DO2_log": np.asarray(bx.DO2.y[:n], dtype=float),
        "CER_log": np.asarray(bx.CER.y[:n], dtype=float),
        "OUR_log": np.asarray(bx.OUR.y[:n], dtype=float),
        "mu_X_calc": np.asarray(bx.mu_X_calc.y[:n], dtype=float),
        "mu_P_calc": np.asarray(bx.mu_P_calc.y[:n], dtype=float),
        "Fs": np.asarray(bx.Fs.y[:n], dtype=float),
        "Fpaa": np.asarray(bx.Fpaa.y[:n], dtype=float),
        "Fg": np.asarray(bx.Fg.y[:n], dtype=float),
        "Fw": np.asarray(bx.Fw.y[:n], dtype=float),
        "Foil": np.asarray(bx.Foil.y[:n], dtype=float),
        "discharge": np.asarray(bx.discharge.y[:n], dtype=float),
        "X_offline": np.asarray(bx.X_offline.y[:n], dtype=float),
        "P_offline": np.asarray(bx.P_offline.y[:n], dtype=float),
    })
    out = pd.concat([logged, diag.drop(columns=["t"], errors="ignore")], axis=1)
    out["seed"] = seed
    out["fs_scale"] = fs_scale
    out["batch_yield"] = batch_yield
    print(f"seed={seed} fs_scale={fs_scale}: yield={batch_yield:.1f} kg  "
          f"Pmax={out.P_log.max():.2f} Xmax={out.X_log.max():.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    outdir = HERE / "out"
    outdir.mkdir(exist_ok=True)
    jobs = [(s, 1.0) for s in range(1, 9)] + \
           [(3, f) for f in (0.7, 0.85, 1.15, 1.3)] + \
           [(5, f) for f in (0.7, 0.85, 1.15, 1.3)]
    frames = []
    for seed, fs in jobs:
        try:
            frames.append(run_one(seed, fs))
        except Exception as e:                                # noqa: BLE001
            print(f"FAILED seed={seed} fs={fs}: {e!r}", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_pickle(outdir / "batches.pkl")
    print("wrote", outdir / "batches.pkl", all_df.shape)

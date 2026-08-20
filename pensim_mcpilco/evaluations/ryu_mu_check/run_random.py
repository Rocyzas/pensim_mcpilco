"""Batches with genuinely different feed trajectories, so 'time' has a real chance to fail.

Piecewise-random Fs: the nominal profile multiplied by a random walk of per-segment
gains in [0.4, 1.8], plus a random shift of the whole ramp-up schedule in time.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_batches as RB
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from utils.recipe import Recipe, RecipeCombo
from utils.peni_env_setup import PenSimEnv


def random_fs_profile(rng):
    """nominal Fs schedule, time-warped and gain-perturbed segment by segment."""
    warp = rng.uniform(0.7, 1.45)          # stretch/compress the whole ramp-up
    g = 1.0
    pts = []
    for p in FS_DEFAULT_PROFILE:
        g = float(np.clip(g * np.exp(rng.normal(0, 0.22)), 0.4, 1.8))
        pts.append({"time": min(230.0, round(p["time"] * warp, 3)), "value": p["value"] * g})
    seen, out = set(), []
    for p in sorted(pts, key=lambda d: d["time"]):
        if p["time"] in seen:
            continue
        seen.add(p["time"]); out.append(p)
    return out


def run_random(seed):
    rng = np.random.default_rng(1000 + seed)
    combo = RecipeCombo(recipe_dict={
        FS: Recipe(random_fs_profile(rng), FS),
        FOIL: Recipe(copy.deepcopy(FOIL_DEFAULT_PROFILE), FOIL),
        FG: Recipe(copy.deepcopy(FG_DEFAULT_PROFILE), FG),
        PRES: Recipe(copy.deepcopy(PRESS_DEFAULT_PROFILE), PRES),
        DISCHARGE: Recipe(copy.deepcopy(DISCHARGE_DEFAULT_PROFILE), DISCHARGE),
        WATER: Recipe(copy.deepcopy(WATER_DEFAULT_PROFILE), WATER),
        PAA: Recipe(copy.deepcopy(PAA_DEFAULT_PROFILE), PAA),
    })
    RB._records = []
    env = PenSimEnv(recipe_combo=combo, fast=True)
    np.random.seed(seed)
    df, by, bx = env.get_batches(random_seed=seed, return_batch_data=True)
    diag = pd.DataFrame(RB._records)
    n = len(diag)
    import utils.constants as C
    logged = pd.DataFrame({
        "k": np.arange(1, n + 1), "time_h": np.arange(1, n + 1) * C.STEP_IN_HOURS,
        "X_log": np.asarray(bx.X.y[:n], float), "P_log": np.asarray(bx.P.y[:n], float),
        "S_log": np.asarray(bx.S.y[:n], float), "V_log": np.asarray(bx.V.y[:n], float),
        "Wt_log": np.asarray(bx.Wt.y[:n], float), "Visc_log": np.asarray(bx.Viscosity.y[:n], float),
        "DO2_log": np.asarray(bx.DO2.y[:n], float), "CER_log": np.asarray(bx.CER.y[:n], float),
        "OUR_log": np.asarray(bx.OUR.y[:n], float),
        "mu_X_calc": np.asarray(bx.mu_X_calc.y[:n], float),
        "mu_P_calc": np.asarray(bx.mu_P_calc.y[:n], float),
        "Fs": np.asarray(bx.Fs.y[:n], float), "Fpaa": np.asarray(bx.Fpaa.y[:n], float),
        "Fg": np.asarray(bx.Fg.y[:n], float), "Fw": np.asarray(bx.Fw.y[:n], float),
        "Foil": np.asarray(bx.Foil.y[:n], float),
        "discharge": np.asarray(bx.discharge.y[:n], float),
        "X_offline": np.asarray(bx.X_offline.y[:n], float),
        "P_offline": np.asarray(bx.P_offline.y[:n], float),
    })
    out = pd.concat([logged, diag.drop(columns=["t"], errors="ignore")], axis=1)
    out["seed"] = seed; out["fs_scale"] = np.nan; out["batch_yield"] = by
    out["batch"] = f"rand{seed}"
    print(f"rand{seed}: yield={by:.0f} kg Pmax={out.P_log.max():.1f} Xmax={out.X_log.max():.1f} "
          f"Fs(80h)={out.Fs.iloc[400]:.0f}", flush=True)
    return out


if __name__ == "__main__":
    frames = []
    for s in range(1, 25):
        try:
            frames.append(run_random(s))
        except Exception as e:                            # noqa: BLE001
            print("FAILED", s, repr(e), flush=True)
    pd.concat(frames, ignore_index=True).to_pickle(HERE / "out" / "random.pkl")
    print("done")

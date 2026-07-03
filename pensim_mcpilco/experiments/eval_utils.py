"""Shared evaluation metrics for the single-phase experiments.

Factored out of analyze_single_phase.py so the RL-vs-PID comparison harness and the
analysis script use one implementation of the yield metric.
"""
import os as _os, sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np

from utils.recipe import Recipe
from utils.constants import STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import DISCHARGE, DISCHARGE_DEFAULT_PROFILE
from mcpilco.pensim_wrapper import PAA_BAND, VISC_MAX, WT_OVERFLOW

_DISCH = Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE)


def yield_kg(mon):
    """Total penicillin yield (kg) for a batch.

    Preferred: sum the per-step `yield_per_run` captured from PenSimEnv.step -- this is
    exactly PenSimPy's `batch_yield`, identical to what the recipe (00) and BO (01)
    baselines report via env.get_batches, so all comparisons are commensurable.

    Legacy fallback (pre-fix monitors without `yield_per_run`): approximate from P/Wt.
    """
    if "yield_per_run" in mon:
        return float(np.sum(mon["yield_per_run"]))
    P, V, t = mon["P"], mon["Wt"], mon["t"]
    Fdis = np.array([_DISCH.get_value_at(float(tt)) for tt in t])
    net = (P[-1] * V[-1] - P[0] * V[0]) / 1000.0
    harvest = float((P * Fdis * STEP_IN_HOURS).sum()) / 1000.0
    return net + harvest


def constraint_diagnostics(mon):
    """Per-batch constraint summary from a monitor dict (keys t/PAA/Viscosity/Wt/P/Fpaa)."""
    PAA = np.asarray(mon["PAA"])
    Wt = np.asarray(mon["Wt"])
    P = np.asarray(mon["P"])
    visc = np.asarray(mon["Viscosity"])
    lo, hi = PAA_BAND
    return {
        "final_P": float(P[-1]),
        "max_Wt": float(Wt.max()),
        "wt_overflow": bool(Wt.max() > WT_OVERFLOW),
        "paa_frac_out_of_band": float(((PAA < lo) | (PAA > hi)).mean()),
        "max_viscosity": float(visc.max()),
        "visc_exceed": bool(visc.max() > VISC_MAX),
    }

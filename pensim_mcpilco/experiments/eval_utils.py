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


def yield_kg(mon, strict=False):
    """Total penicillin yield (kg) for a batch.

    Preferred: sum the per-step `yield_per_run` captured from PenSimEnv.step -- this is
    exactly PenSimPy's `batch_yield`, identical to what the recipe (00) and BO (01)
    baselines report via env.get_batches, so all comparisons are commensurable.

    Legacy fallback (pre-fix monitors without `yield_per_run`): approximate from P/Wt. This is a
    DIFFERENT estimator, so mixing it with the preferred one across arms of the same comparison
    would be silently incomparable -- set `strict=True` (or PENSIM_STRICT_YIELD=1) to make the
    fallback raise instead of quietly changing metric under you.
    """
    if "yield_per_run" in mon:
        return float(np.sum(mon["yield_per_run"]))
    if strict or _os.environ.get("PENSIM_STRICT_YIELD") == "1":
        raise KeyError(
            "monitor has no 'yield_per_run': would fall back to the legacy P/Wt estimator, which is "
            "NOT comparable with the primary metric. Re-collect the monitor, or pass strict=False."
        )
    P, V, t = mon["P"], mon["Wt"], mon["t"]
    Fdis = np.array([_DISCH.get_value_at(float(tt)) for tt in t])
    net = (P[-1] * V[-1] - P[0] * V[0]) / 1000.0
    harvest = float((P * Fdis * STEP_IN_HOURS).sum()) / 1000.0
    return net + harvest


def feasibility_gated_yield_kg(mon):
    """yield_kg(mon), zeroed if the batch breached the operating envelope (Wt overflow or
    viscosity collapse). An envelope-breaching batch is operationally rejected -- it contributes
    no usable product -- so this, not raw yield_kg, is the metric training progress should be
    judged on: raw yield can look better while quietly trading away feasibility (see
    evaluations/cost_reward_hacking_bo_results.md, whose own envelope-valid-yield check uses the
    same VISC_MAX/WT_OVERFLOW breach definition inline; this is that same check factored out so
    the training-progression plots use it too, not just the offline BO probe). Reuses
    constraint_diagnostics's breach flags rather than re-deriving them, so there is exactly one
    definition of "breached" for both metrics to agree on."""
    d = constraint_diagnostics(mon)
    return 0.0 if (d["wt_overflow"] or d["visc_exceed"]) else yield_kg(mon)


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

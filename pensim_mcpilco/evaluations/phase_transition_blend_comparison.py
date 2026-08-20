"""One figure: the FIXED (analytical) phase transition against the DYNAMICAL (biomass) one.

    python phase_transition_blend_comparison.py
    python phase_transition_blend_comparison.py --n_batches 30 --feed_delay 0,40,80
    python phase_transition_blend_comparison.py --policy_run results/biomass/rollout/seed4_1

WHAT IT SHOWS
    Both variants answer "how much is phase 2 in charge right now?" with a 1%/99% logistic, but
    on different coordinates:

      FIXED      w(t) = sigmoid( ln99/blend_half_width_hours * (t - pivot_hours) )
                 A single deterministic curve. With the defaults (100 +- 50 h) it rises from 1%
                 at 50 h to 99% at 150 h and is IDENTICAL in every batch -- zero spread by
                 construction, which is exactly the property --onEachRollout exists to remove.

      BIOMASS    w = sigmoid( ln99/blend_half_width_bm * (runmax(X*Wt/1000) - pivot_bm) )
                 No t anywhere. Every batch crosses when ITS OWN biomass crosses pivot_bm, so
                 there is one curve per batch and the spread between them is the quantity of
                 interest. Drawn as faint lines (one per batch) plus the median.

    The right panel turns each curve into the single number it implies -- the hour at which
    w first reaches 0.5 -- so the two mechanisms can be compared as distributions.

EXACTNESS
    Neither sigmoid is re-implemented. A DualPhaseModelLearning is built with object.__new__ and
    given only the attributes _blend_weight touches, then the REAL method is called -- once with
    on_each_rollout=False (fixed branch) and once with True (biomass branch). The N batches are
    passed as N "particles" in a single pass, which is precisely how the per-particle branch is
    driven during policy optimisation, so the running-max accumulation is production behaviour
    rather than a copy of it. _assert_matches_production checks the fixed branch against a
    hand-written logistic at import time; if the production formula ever changes, this fails
    loudly instead of quietly plotting the old curve.

    --feed_delay overlays additional biomass groups from delayed batches (see
    PenSimWrapper.rollout(feed_delay_h=...)), which is the direct visual of the fixed boundary
    standing still while the dynamical one tracks the culture.

Read-only: runs recipe (or a trained policy's) batches and writes only into
phase_transition_blend/.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_EVAL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _EVAL_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_DIR)
# model_learning_det_time does a bare `import model_learning.Model_learning`, which only resolves
# once MC-PILCO/ is on the path. Elsewhere that happens as a SIDE EFFECT of importing a config
# module (mcpilco/config_single_phase.py:10). This script imports the model layer directly and
# would otherwise depend on import order, so add it explicitly.
_MCPILCO = _os.path.join(_os.path.dirname(_ROOT), "MC-PILCO")
if _MCPILCO not in _sys.path:
    _sys.path.insert(0, _MCPILCO)

import argparse
import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcpilco.model_learning_dual_phase import (DualPhaseModelLearning, BM_PIVOT_DEFAULT,
                                               BLEND_HALF_WIDTH_BM_DEFAULT, _bm_from_states)
from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, BLEND_HALF_WIDTH_HOURS,
                                    PIVOT_HOURS)

OUT_DIR = Path(_EVAL_DIR) / "phase_transition_blend"

# pensim_wrapper's module constants are PIVOT_HOURS=90 / BLEND_HALF_WIDTH_HOURS=40, but NO run in
# this study used them: 03_mcpilco_dual_phase_baseline.py's argparse defaults (--pivot_hours 100.0,
# --blend_half_width_hours 50.0) override the constants, and every stored note.txt records 100/50
# (transition window 50-150 h). Defaulting to the module constants would draw a fixed curve no
# trained model ever had, so the study values are the defaults here and --from_run reads the exact
# numbers back out of a run's note.txt.
STUDY_PIVOT_HOURS, STUDY_HALF_HOURS = 100.0, 50.0
FIXED_C, BIO_C = "0.15", "C0"
DELAY_CMAP = ("C1", "C2", "C4", "C5")


def _stub(on_each_rollout, pivot_hours, half_hours, pivot_bm, half_bm):
    """DualPhaseModelLearning carrying only what _blend_weight reads. Built with object.__new__
    because __init__ would construct two full GP phase models we have no use for; if the method
    ever starts reading more state this raises AttributeError rather than silently diverging."""
    s = object.__new__(DualPhaseModelLearning)
    s.on_each_rollout = bool(on_each_rollout)
    s.pivot_hours, s.blend_half_width_hours = pivot_hours, half_hours
    s.pivot_bm, s.blend_half_width_bm = pivot_bm, half_bm
    s._bm_max, s._t = None, 0
    return s


def _assert_matches_production(pivot_hours, half_hours):
    s = _stub(False, pivot_hours, half_hours, BM_PIVOT_DEFAULT, BLEND_HALF_WIDTH_BM_DEFAULT)
    k = math.log(99.0) / half_hours
    for step in (0, 6, 12, 20, 30, 45):
        mine = 1.0 / (1.0 + math.exp(-k * (step * T_SAMPLING - pivot_hours)))
        if abs(mine - float(s._blend_weight(step))) > 1e-12:
            raise RuntimeError("production _blend_weight no longer matches the documented "
                               "logistic; update this script before trusting its figure")


def fixed_curve(n_steps, pivot_hours, half_hours):
    s = _stub(False, pivot_hours, half_hours, BM_PIVOT_DEFAULT, BLEND_HALF_WIDTH_BM_DEFAULT)
    return np.array([float(s._blend_weight(t)) for t in range(n_steps)])


def biomass_curves(states, pivot_bm, half_bm):
    """states (n_batches, n_steps, state_dim) -> w (n_steps, n_batches).

    All batches go through as parallel particles in one pass, which is how the per-particle
    branch is driven in MC_PILCO.compute_particles_trj -- so the causal running max accumulates
    per batch exactly as it does in training."""
    s = _stub(True, PIVOT_HOURS, BLEND_HALF_WIDTH_HOURS, pivot_bm, half_bm)
    s.reset_step_counter(0, bm_max0=None)
    out = []
    for t in range(states.shape[1]):
        w = s._blend_weight(t, torch.tensor(states[:, t, :], dtype=torch.float64))
        out.append(np.asarray(w).ravel())
    return np.stack(out)


def crossing_hours(W, hours):
    """First hour at which each column of W reaches 0.5; NaN if it never does."""
    out = []
    for j in range(W.shape[1]):
        hit = np.flatnonzero(W[:, j] >= 0.5)
        out.append(hours[hit[0]] if hit.size else np.nan)
    return np.array(out, dtype=float)


def run_batches(seeds, feed_delay_h, policy=None):
    st = []
    for sd in seeds:
        w = PenSimWrapper()
        _, _, s = w.rollout(None, policy, CONTROL_H, T_SAMPLING, None, seed=sd,
                            pid_baseline=policy is None, feed_delay_h=feed_delay_h)
        st.append(s)
    return np.stack(st)


def main(n_batches, eval_base, delays, pivot_bm, half_bm, pivot_hours, half_hours,
         policy_run, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _assert_matches_production(pivot_hours, half_hours)
    seeds = [eval_base + i for i in range(n_batches)]

    policy, lbl = None, "recipe"
    if policy_run:
        import evaluations.eval_multi_phase_lib as mlib
        from mcpilco.config_dual_phase_baseline import get_config
        run = mlib.load_run(_os.path.abspath(policy_run), get_config_fn=get_config)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, policy, _, _ = mlib.build_policy_agent(run)
        lbl = f"policy {Path(policy_run).name}"

    groups, rows = {}, []
    for D in delays:
        S = run_batches(seeds, D, policy)
        hours = np.arange(S.shape[1]) * T_SAMPLING
        W = biomass_curves(S, pivot_bm, half_bm)
        groups[D] = (hours, W, crossing_hours(W, hours))
        for j, sd in enumerate(seeds):
            rows.append(dict(feed_delay_h=D, eval_seed=sd, crossing_h=groups[D][2][j],
                             bm_max=float(np.max(_bm_from_states(S[j])))))
        print(f"  feed_delay {D:>3.0f}h: biomass crossing median {np.nanmedian(groups[D][2]):.0f}h "
              f"[{np.nanmin(groups[D][2]):.0f}-{np.nanmax(groups[D][2]):.0f}]  (n={n_batches})")
    pd.DataFrame(rows).to_csv(out_dir / "blend_crossings.csv", index=False)

    hours = groups[delays[0]][0]
    wf = fixed_curve(len(hours), pivot_hours, half_hours)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2),
                           gridspec_kw={"width_ratios": [2.1, 1]})

    a = ax[0]
    lo, hi = pivot_hours - half_hours, pivot_hours + half_hours
    a.axvspan(lo, hi, color=FIXED_C, alpha=.35, zorder=0,
              label=f"fixed transition window {lo:.0f}-{hi:.0f} h")
    a.plot(hours, wf, color="k", lw=3.0, zorder=5,
           label=f"FIXED (analytical), centre {pivot_hours:.0f} h — identical every batch")
    for i, D in enumerate(delays):
        hrs, W, _ = groups[D]
        c = BIO_C if D == 0 else DELAY_CMAP[(i - 1) % len(DELAY_CMAP)]
        a.plot(hrs, W, color=c, alpha=0.18, lw=1.0, zorder=3)          # one line per batch
        a.plot(hrs, np.median(W, axis=1), color=c, lw=2.6, zorder=6,
               label=f"BIOMASS (dynamical), median of {W.shape[1]}"
                     + (f" — feed delay {D:.0f} h" if D else ""))
    a.axhline(0.5, color="0.4", ls=":", lw=1.0)
    a.set_xlabel("batch time (h)"); a.set_ylabel("blend weight w  (phase-2 share)")
    a.set_title(f"Phase transition: fixed clock vs biomass  [{lbl}]")
    a.set_ylim(-0.03, 1.03); a.grid(alpha=.3); a.legend(fontsize=8, loc="center right")

    b = ax[1]
    data = [groups[D][2][~np.isnan(groups[D][2])] for D in delays]
    parts = b.violinplot(data, positions=range(len(delays)), widths=.8, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(BIO_C if delays[i] == 0 else DELAY_CMAP[(i - 1) % len(DELAY_CMAP)])
        pc.set_alpha(.55)
    b.axhline(pivot_hours, color="k", lw=2.5, label=f"FIXED, always {pivot_hours:.0f} h (spread 0)")
    b.axhspan(lo, hi, color=FIXED_C, alpha=.35, zorder=0)
    for i, d in enumerate(data):
        b.scatter(np.full(len(d), i) + np.random.uniform(-.06, .06, len(d)), d,
                  s=12, color="k", alpha=.45, zorder=4)
    b.set_xticks(range(len(delays)))
    b.set_xticklabels([f"{D:.0f}h" for D in delays])
    b.set_xlabel("feed delay applied" if len(delays) > 1 else "")
    b.set_ylabel("hour at which w reaches 0.5")
    b.set_title("Where the transition actually lands")
    b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)

    fig.suptitle("Fixed (analytical) vs dynamical (biomass) phase blending")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "phase_transition_blend_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_dir}/phase_transition_blend_comparison.png and blend_crossings.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--eval_base", type=int, default=700000)
    p.add_argument("--feed_delay", default="0",
                   help="comma-separated feed-delay hours; >1 value overlays delayed groups")
    p.add_argument("--pivot_bm", type=float, default=BM_PIVOT_DEFAULT)
    p.add_argument("--blend_half_width_bm", type=float, default=BLEND_HALF_WIDTH_BM_DEFAULT)
    p.add_argument("--pivot_hours", type=float, default=STUDY_PIVOT_HOURS)
    p.add_argument("--blend_half_width_hours", type=float, default=STUDY_HALF_HOURS)
    p.add_argument("--from_run", default=None,
                   help="read pivot_hours/blend_half_width_hours/pivot_bm/"
                        "blend_half_width_bm from this run dir's note.txt")
    p.add_argument("--policy_run", default=None,
                   help="run dir whose trained policy drives the batches (default: recipe)")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    if a.from_run:
        import re as _re
        _t = Path(a.from_run, "note.txt").read_text(errors="ignore")
        for _k, _d in (("pivot_hours", "pivot_hours"), ("blend_half_width_hours", "blend_half_width_hours"),
                       ("pivot_bm", "pivot_bm"), ("blend_half_width_bm", "blend_half_width_bm")):
            _m = _re.search(rf"^{_k} = (.+)$", _t, _re.M)
            if _m and _m.group(1).strip() not in ("None", ""):
                setattr(a, _d, float(_m.group(1)))
        print(f"from_run {a.from_run}: pivot_hours={a.pivot_hours} "
              f"half_hours={a.blend_half_width_hours} pivot_bm={a.pivot_bm} "
              f"half_bm={a.blend_half_width_bm}")
    main(a.n_batches, a.eval_base, [float(x) for x in a.feed_delay.split(",")],
         a.pivot_bm, a.blend_half_width_bm, a.pivot_hours, a.blend_half_width_hours,
         a.policy_run, a.out_dir)

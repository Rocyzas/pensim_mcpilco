"""Shared analysis library for dual-phase MC-PILCO evaluation.

Both evaluations_multi_phase.py (CLI) and evaluations_multi_phase.ipynb (notebook) import
from here, so there is exactly one implementation of every plot/table -- the two entry points
cannot silently drift apart. Every plot/table function always saves its output into `out_dir`
(PNG for plots, CSV for tables); pass show=True (the notebook does) to also display inline.

Ported from evaluations/evaluations.ipynb (the single-phase notebook), adapted for the
dual-phase composite model (mcpilco.model_learning_dual_phase.DualPhaseModelLearning). Two
things are genuinely different here, not just renamed -- see reconstruct_gp_agent() and the
`reset_step_counter` calls sprinkled through the C.* helpers below:

1. reconstruct_gp_agent() assigns saved GP weights into `model_learning.phase1`/`.phase2`
   directly (each a plain Model_learning_RBF_det_time with the usual mutable gp_inputs/
   gp_output_list/gp_list/init_gp_models()/pretrain_gp()) rather than into `model_learning`
   itself, which only exposes those as READ-ONLY concatenations of the two sub-models.
2. Any helper that calls `model_learning.get_next_state(...)` directly at an arbitrary batch
   origin t0 (bypassing agent.apply_policy/agent.rollout, which already reset the phase
   counter correctly) must call `model_learning.reset_step_counter(t0)` immediately before,
   or the internal phase-routing counter desyncs from the absolute decision time being probed
   and silently scores the wrong phase's GP.

C.7 (short-horizon/anchor diagnostic) from the single-phase notebook is intentionally NOT
ported: anchors/optim-horizon are an unsupported combination with dual-phase (see
PenSimMCPILCOMultiPhase's docstring), so there is nothing meaningful for it to show here.

DualPhaseModelLearning blends phase1/phase2 predictions via a sigmoid centered on
run.pivot_hours (not a hard switch) -- `_mark_pivot` shades the ~1%-99% transition window
(pivot +- run.blend_half_width_hours) on every full-batch time-axis plot, not just a single
cutoff line, since predictions genuinely mix both phases within that window.
"""
import ast
import contextlib
import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if _os.path.dirname(_ROOT) not in _sys.path:
    _sys.path.insert(0, _os.path.dirname(_ROOT))

from utils.recipe import Recipe
from utils.constants import STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import FS, FS_DEFAULT_PROFILE

from mcpilco.config_dual_phase import get_config
from mcpilco.pensim_wrapper import (
    PenSimWrapper, PenSimMCPILCOMultiPhase, STATE_NAMES, STATE_DIM, ACTION_DIM,
    STATE_RANGES, decode_state_value, T_SAMPLING, CONTROL_H, K_WARM, VISC_MAX, WARMUP_H,
    PAA_BAND, FS_SCALE, FPAA_MIN, FPAA_MAX,
)
from experiments.eval_utils import yield_kg, feasibility_gated_yield_kg, constraint_diagnostics

X_IDX = STATE_NAMES.index("X")
P_IDX = STATE_NAMES.index("P")
ACTION_IDX = STATE_DIM  # gp_input columns are [states..., action]

DUAL_STATE_NAMES = [f"p1:{n}" for n in STATE_NAMES] + [f"p2:{n}" for n in STATE_NAMES]
X_IDX_P1, X_IDX_P2 = X_IDX, STATE_DIM + X_IDX
P_IDX_P1, P_IDX_P2 = P_IDX, STATE_DIM + P_IDX

RESULTS_ROOT = Path(_ROOT) / "results" / "dual_phase"

REF_STYLE = dict(color="red", lw=2.2, ls="--", zorder=6)
MODEL_STYLE = dict(color="C0", lw=2.0, zorder=5)

# get_config kwargs recoverable from note.txt's "== run parameters ==" block (see
# _write_note() in experiments/03_mcpilco_dual_phase.py). Anything else written there
# (out_dir) isn't a get_config kwarg and is dropped when building cfg.
_GET_CONFIG_KEYS = ("seed", "num_trials", "fast", "pivot_hours", "blend_half_width_hours",
                    "risk_weight", "visc_penalty", "constraint_strength", "harvest_reward",
                    "pms_visc_delay", "use_offline_measurements")


def _build_cfg_kwargs(params):
    """Filter note.txt's params down to get_config kwargs. Mirrors
    eval_single_phase_lib._build_cfg_kwargs -- see its docstring for why the two
    Viscosity-delay flags need an explicit setdefault rather than falling through to
    get_config's own default: both predate these keys existing in note.txt at all, so a
    run written before they were added has no such line, and absent MUST mean "trained
    before this feature existed" (plain online Viscosity), not whatever get_config's
    default happens to be at reconstruction time."""
    out = {k: v for k, v in params.items() if k in _GET_CONFIG_KEYS}
    out.setdefault("pms_visc_delay", False)
    out.setdefault("use_offline_measurements", False)
    return out


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def resolve_run_dir(run_id_or_path, results_root=None):
    """Accepts a bare run name ("seed3_1") resolved under RESULTS_ROOT (or `results_root`, for
    a sibling results tree such as dual_phase_baseline's), or an existing absolute/relative
    path directly."""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    p = Path(run_id_or_path)
    if p.exists():
        return p
    candidate = root / run_id_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no run at {run_id_or_path!r} or {candidate}")


def parse_run_params(note_path):
    """Parse the '== run parameters ==' block _write_note() wrote back into a dict. Every
    value there is a plain repr'd Python literal (int/float/bool/None/str), written one
    "key = value" per line, so ast.literal_eval round-trips it exactly."""
    lines = Path(note_path).read_text().splitlines()
    try:
        start = lines.index("== run parameters ==") + 1
    except ValueError:
        raise ValueError(f"{note_path}: no '== run parameters ==' section found")
    params = {}
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("=="):
            break
        key, _, value = line.partition(" = ")
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


@dataclass
class Run:
    dir: Path
    params: dict
    cfg: dict
    log: dict
    monitors: list | None
    n_ep: int
    n_trials_in_log: int
    num_explorations: int
    train_seed: int

    @property
    def pivot_hours(self):
        return float(self.params.get("pivot_hours", 90.0))

    @property
    def blend_half_width_hours(self):
        return float(self.params.get("blend_half_width_hours", 40.0))


def log_state_dims(log):
    """(policy_state_dim, gp_input_dim) recorded in a run log; None where not present."""
    pol = log.get("parameters_trial_list") or []
    p_dim = int(np.asarray(pol[-1]["centers"]).shape[1]) if pol else None
    gp_keys = [k for k in log if k.startswith("gp_inputs_")]
    g_dim = None
    if gp_keys:
        newest = max(gp_keys, key=lambda k: int(k.split("_")[-1]))
        g_dim = int(np.asarray(log[newest]).shape[1])
    return p_dim, g_dim


def find_compatible_runs(want_state_dim, run_dir, limit=10):
    """Sibling runs whose saved policy matches the CURRENT state dimension."""
    out = []
    for d in sorted(Path(run_dir).parent.glob("*/")):
        f = d / "log.pkl"
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                pol = pickle.load(fh).get("parameters_trial_list") or []
        except Exception:
            continue
        if pol and int(np.asarray(pol[-1]["centers"]).shape[1]) == want_state_dim:
            out.append(d.name)
    return out[:limit]


def assert_log_matches_code(log, run_dir):
    """Fail EARLY and legibly when a run predates the current STATE_NAMES."""
    p_dim, g_dim = log_state_dims(log)
    want_p, want_g = STATE_DIM, STATE_DIM + ACTION_DIM
    if (p_dim not in (None, want_p)) or (g_dim not in (None, want_g)):
        alts = find_compatible_runs(want_p, run_dir)
        raise RuntimeError(
            f"run '{Path(run_dir).name}' was trained with a DIFFERENT state definition.\n"
            f"  saved:   policy state_dim={p_dim}, gp_input_dim={g_dim}\n"
            f"  current: policy state_dim={want_p}, gp_input_dim={want_g}\n"
            f"  current STATE_NAMES = {STATE_NAMES}\n"
            f"  compatible runs: {alts or 'NONE FOUND - retrain needed'}"
        )
    print(f"log/code state check OK: state_dim={want_p}, channels={STATE_NAMES}")


def _resolve_trial(log, trial):
    """Default to the last trial that has a saved GP model."""
    avail = sorted(int(k.split("_")[-1]) for k in log if k.startswith("parameters_gp_"))
    if not avail:
        raise RuntimeError("log.pkl has no parameters_gp_<i> (no trained GP to load)")
    if trial is None:
        return avail[-1]
    if trial not in avail:
        raise RuntimeError(f"trial {trial} not in saved GP trials {avail}")
    return trial


def load_run(run_id_or_path, get_config_fn=None, results_root=None):
    """The one function both entry points call first. Resolves the dir, parses note.txt,
    rebuilds cfg via get_config(**params), loads log.pkl/monitor.pkl, checks state-dim
    compatibility.

    get_config_fn/results_root default to this module's own config_dual_phase.get_config /
    RESULTS_ROOT (regular runs); pass config_dual_phase_baseline's get_config and its
    results/dual_phase_baseline root to load a plain-RBF ablation run instead -- see
    action_sensitivity_multi_phase_baseline.py / evaluations_multi_phase_baseline.py."""
    get_config_fn = get_config if get_config_fn is None else get_config_fn
    run_dir = resolve_run_dir(run_id_or_path, results_root=results_root)
    note_path = run_dir / "note.txt"
    if not note_path.exists():
        raise FileNotFoundError(f"{run_dir}: no note.txt (needed to recover run params)")
    all_params = parse_run_params(note_path)
    cfg_kwargs = _build_cfg_kwargs(all_params)
    cfg = get_config_fn(**cfg_kwargs)

    log_path = run_dir / "log.pkl"
    if not log_path.exists():
        raise FileNotFoundError(f"{run_dir}: no log.pkl")
    with open(log_path, "rb") as f:
        log = pickle.load(f)
    mon_path = run_dir / "monitor.pkl"
    monitors = None
    if mon_path.exists():
        with open(mon_path, "rb") as f:
            monitors = pickle.load(f)

    n_ep = len(log["state_samples_history"])
    n_trials_in_log = len(log.get("parameters_trial_list", [])) or int(all_params.get("num_trials", 0))
    assert_log_matches_code(log, run_dir)
    num_explorations = max(0, n_ep - n_trials_in_log)
    train_seed = cfg["wrapper_par"]["seed_offset"]

    print(f"run: {run_dir}  |  episodes: {n_ep}  |  trials in log: {n_trials_in_log}  |  "
         f"pivot: {cfg_kwargs.get('pivot_hours', 90.0):g} h")

    return Run(dir=run_dir, params=all_params, cfg=cfg, log=log, monitors=monitors,
              n_ep=n_ep, n_trials_in_log=n_trials_in_log,
              num_explorations=num_explorations, train_seed=train_seed)


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def decision_time_grid(N):
    """Physical decision-time axis (h) for a stored trajectory of N samples."""
    return K_WARM * STEP_IN_HOURS + np.arange(N) * T_SAMPLING


def _denorm(x, lo, hi):
    return lo + (np.asarray(x) + 1.0) * (hi - lo) / 2.0


def _denorm_delta(d, lo, hi):
    return np.asarray(d) * (hi - lo) / 2.0


def _denorm_phys(x, name):
    return decode_state_value(name, _denorm(x, *STATE_RANGES[name]))


def final_P(state_norm):
    P = _denorm_phys(np.asarray(state_norm)[:, P_IDX], "P")
    return float(P[-1]), float(P.mean())


def _ep_color(i, n_ep, n_expl):
    """grey for exploration episodes, viridis (early->late) for the trials."""
    if i < n_expl:
        return "0.72"
    span = max(1, n_ep - n_expl - 1)
    return cm.viridis((i - n_expl) / span)


def _episode_colorbar(fig, ax, n_ep, n_expl, label="trial (early -> late)"):
    lo, hi = n_expl, max(n_expl + 1, n_ep - 1)
    sm = ScalarMappable(cmap=cm.viridis, norm=Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def _mark_pivot(ax, pivot_hours, blend_half_width_hours=None):
    """Marker at the dual-phase pivot -- the CENTER of the sigmoid blend that combines
    phase-1/phase-2 GP predictions -- on any panel whose x-axis is batch time. When
    blend_half_width_hours is given, also shades the ~1%-99% transition window
    (pivot +- half_width) so the plot shows a region, not a hard cutoff that no longer exists."""
    for a in np.atleast_1d(ax).ravel():
        if blend_half_width_hours is not None:
            a.axvspan(pivot_hours - blend_half_width_hours, pivot_hours + blend_half_width_hours,
                      color="purple", alpha=0.06, zorder=0,
                      label=f"blend window (+-{blend_half_width_hours:g} h)")
        a.axvline(pivot_hours, color="purple", ls=":", lw=1.2, label=f"pivot ({pivot_hours:g} h)")


def run_arm(wrapper, seed, policy=None, pid_baseline=False):
    """Run one batch on `seed` and return its monitor dict."""
    wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0, seed=seed, pid_baseline=pid_baseline)
    return wrapper.monitor[-1]


def _finish(fig, out_dir, filename, show):
    fig.savefig(Path(out_dir) / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _assert_policy_state_dim_ok(np_policy, label):
    test_state = np.zeros(STATE_DIM)
    try:
        action = np.asarray(np_policy(test_state, 0), dtype=float).ravel()
    except Exception as e:
        raise RuntimeError(
            f"{label}: policy forward pass failed on a {STATE_DIM}-dim state "
            f"(STATE_NAMES={STATE_NAMES}). This checkpoint was very likely trained under a "
            f"different state space and cannot be evaluated without retraining."
        ) from e
    if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
        raise RuntimeError(
            f"{label}: policy output {action} is the wrong shape or non-finite for a "
            f"{STATE_DIM}-dim state."
        )


# ---------------------------------------------------------------------------
# Policy loading (Section A)
# ---------------------------------------------------------------------------

def build_policy_agent(run):
    """Build a PenSimMCPILCOMultiPhase, load the trained policy (last saved trial) from
    run.log, and roll a same-seed recipe reference batch. load_policy_from_log only touches
    control_policy (never model_learning), so it works unmodified for the dual-phase
    composite."""
    eval_wrapper = PenSimWrapper()
    policy_agent = PenSimMCPILCOMultiPhase(pensim_wrapper=eval_wrapper, **run.cfg["mc_pilco_init"])
    folder = str(run.dir).rstrip("/") + "/"
    with contextlib.redirect_stdout(io.StringIO()):
        policy_agent.load_policy_from_log(num_trial=run.n_trials_in_log, folder=folder)
    np_policy = policy_agent.control_policy.get_np_policy()
    _assert_policy_state_dim_ok(np_policy, f"trial {run.n_trials_in_log} policy")

    base_mon = run_arm(eval_wrapper, run.train_seed, policy=None, pid_baseline=True)
    ref = {k: (np.asarray(base_mon["t"]), np.asarray(base_mon[k]))
          for k in ("P", "PAA", "Viscosity", "Fpaa", "Wt", "Fs")}
    ref["yield"] = yield_kg(base_mon)
    ref["yield_gated"] = feasibility_gated_yield_kg(base_mon)
    ref["final_P"] = float(base_mon["P"][-1])
    ref_lbl = f"recipe (seed {run.train_seed})"
    print(f"loaded trial {run.n_trials_in_log} policy | recipe baseline on seed {run.train_seed}: "
         f"final_P={ref['final_P']:.2f} g/L, yield={ref['yield']:.1f} kg "
         f"(feasibility-gated: {ref['yield_gated']:.1f} kg)")
    return policy_agent, eval_wrapper, np_policy, ref, ref_lbl


def load_stage_policy(run, trial_k, folder=None):
    """Load trial-k's saved policy onto a FRESH staging agent (get_np_policy() is a live
    alias to control_policy, so reusing the main policy_agent here would mutate whatever
    np_policy build_policy_agent() already returned)."""
    folder = folder or (str(run.dir).rstrip("/") + "/")
    stage_agent = PenSimMCPILCOMultiPhase(pensim_wrapper=PenSimWrapper(), **run.cfg["mc_pilco_init"])
    with contextlib.redirect_stdout(io.StringIO()):
        stage_agent.load_policy_from_log(num_trial=trial_k, folder=folder)
    pol = stage_agent.control_policy.get_np_policy()
    _assert_policy_state_dim_ok(pol, f"trial {trial_k} policy")
    return pol


# ---------------------------------------------------------------------------
# Section A: RL vs PID/recipe baseline
# ---------------------------------------------------------------------------

def eval_held_out(run, np_policy, eval_wrapper, out_dir, n_eval_seeds=5, eval_base=700000):
    eval_seeds = [eval_base + i for i in range(n_eval_seeds)]
    assert run.train_seed not in eval_seeds, f"eval seeds must exclude train_seed={run.train_seed}"
    rows, mons_rl, mons_recipe = [], [], []
    for h in eval_seeds:
        m_rl = run_arm(eval_wrapper, h, policy=np_policy, pid_baseline=False)
        m_recipe = run_arm(eval_wrapper, h, policy=None, pid_baseline=True)
        mons_rl.append(m_rl); mons_recipe.append(m_recipe)
        y_rl, y_recipe = yield_kg(m_rl), yield_kg(m_recipe)
        row = {"seed": h, "yield_rl": y_rl, "yield_recipe": y_recipe, "delta": y_rl - y_recipe}
        row.update({f"rl_{k}": v for k, v in constraint_diagnostics(m_rl).items()})
        row.update({f"recipe_{k}": v for k, v in constraint_diagnostics(m_recipe).items()})
        rows.append(row)
        print(f"seed {h}: yield_rl={y_rl:8.2f}  yield_recipe={y_recipe:8.2f}  delta={y_rl - y_recipe:+8.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "A1_held_out.csv", index=False)
    return df, mons_rl, mons_recipe


def paired_stats(df, out_dir):
    delta = df["delta"].values
    n = len(delta)
    mean_d = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    ci = 1.96 * se
    t_p = float(stats.ttest_rel(df["yield_rl"], df["yield_recipe"]).pvalue) if n > 1 else float("nan")
    try:
        w_p = float(stats.wilcoxon(df["yield_rl"], df["yield_recipe"]).pvalue)
    except ValueError:
        w_p = float("nan")
    summary = pd.DataFrame([{
        "n_seeds": n,
        "mean_yield_rl": float(df["yield_rl"].mean()),
        "mean_yield_recipe": float(df["yield_recipe"].mean()),
        "mean_delta": mean_d,
        "delta_ci95_lo": mean_d - ci,
        "delta_ci95_hi": mean_d + ci,
        "ttest_p": t_p,
        "wilcoxon_p": w_p,
        "winrate_rl_gt_recipe": float((delta > 0).mean()),
    }])
    summary.to_csv(Path(out_dir) / "A2_paired_stats.csv", index=False)
    print("=== SUMMARY (RL vs recipe, held-out) ===")
    print(summary.T.to_string(header=False))
    return summary


def plot_paired_yield(df, mons_rl, mons_recipe, out_dir, show=False):
    t = np.asarray(mons_rl[0]["t"])
    avg = lambda mons, key: np.mean([m[key] for m in mons], axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    lo = float(min(df["yield_recipe"].min(), df["yield_rl"].min()))
    hi = float(max(df["yield_recipe"].max(), df["yield_rl"].max()))
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].scatter(df["yield_recipe"], df["yield_rl"], c="C0")
    ax[0].set_xlabel("recipe yield (kg)"); ax[0].set_ylabel("RL yield (kg)")
    ax[0].set_title("Paired yield (above y=x => RL wins)"); ax[0].grid(alpha=.3)

    ax[1].bar(np.arange(len(df)), df["delta"].values,
              color=["C2" if d > 0 else "C3" for d in df["delta"]])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("held-out seed idx"); ax[1].set_ylabel("delta yield RL-recipe (kg)")
    ax[1].set_title("Per-seed difference"); ax[1].grid(alpha=.3, axis="y")

    ax[2].plot(t, avg(mons_rl, "P"), label="RL", color="C0")
    ax[2].plot(t, avg(mons_recipe, "P"), label="recipe (= PID arm)", color="C1")
    ax[2].set_xlabel("time (h)"); ax[2].set_ylabel("P (g/L)")
    ax[2].set_title("Seed-averaged penicillin"); ax[2].grid(alpha=.3); ax[2].legend()
    fig.tight_layout()
    _finish(fig, out_dir, "A3_paired_yield.png", show)
    return fig


def plot_total_yield(df, out_dir, show=False):
    n = len(df)
    mean_rl, mean_recipe = float(df["yield_rl"].mean()), float(df["yield_recipe"].mean())
    sem_rl = float(df["yield_rl"].std(ddof=1) / np.sqrt(n))
    sem_recipe = float(df["yield_recipe"].std(ddof=1) / np.sqrt(n))
    mean_d = float(df["delta"].mean())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar([0, 1], [mean_recipe, mean_rl], yerr=[sem_recipe, sem_rl], capsize=6, color=["C1", "C0"])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["recipe (= PID arm)", "RL"])
    ax[0].set_ylabel("mean yield (kg)")
    ax[0].set_title(f"Mean held-out yield (n={n})   delta={mean_rl - mean_recipe:+.1f} kg")
    ax[0].grid(alpha=.3, axis="y")

    ax[1].hist(df["delta"].values, bins=max(5, n // 2), color="C0", alpha=.8, edgecolor="k")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(mean_d, color="crimson", ls="--", lw=2, label=f"mean {mean_d:+.1f} kg")
    ax[1].set_xlabel("delta yield RL-recipe (kg)"); ax[1].set_ylabel("count")
    ax[1].set_title("Distribution of per-seed yield delta"); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, out_dir, "A4_total_yield.png", show)
    return fig


def plot_seed_avg_vars(mons_rl, mons_recipe, out_dir, show=False):
    t = np.asarray(mons_rl[0]["t"])
    avg = lambda mons, key: np.mean([m[key] for m in mons], axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    panels = [("Fs", ax[0, 0], "Fs (L/h)"), ("PAA", ax[0, 1], "PAA (mg/L)"),
              ("Fpaa", ax[1, 0], "Fpaa (L/h)"), ("Viscosity", ax[1, 1], "viscosity (cP)")]
    for key, axx, yl in panels:
        axx.plot(t, avg(mons_rl, key), label="RL", color="C0")
        axx.plot(t, avg(mons_recipe, key), label="recipe (= PID arm)", color="C1")
        axx.axvline(WARMUP_H, color="gray", ls=":", lw=.8)
        axx.set_xlabel("time (h)"); axx.set_ylabel(yl); axx.set_title(f"Seed-averaged {key}")
        axx.grid(alpha=.3); axx.legend()
    ax[0, 1].axhspan(*PAA_BAND, color="green", alpha=.08)
    ax[1, 1].axhline(VISC_MAX, color="crimson", ls="--", lw=1)
    fig.suptitle("Seed-averaged control & constraint variables (RL vs recipe, held-out)")
    fig.tight_layout()
    _finish(fig, out_dir, "A5_seed_avg_vars.png", show)
    return fig


def plot_model_vs_recipe_single_seed(eval_wrapper, np_policy, compare_seed, out_dir, show=False):
    m_rl_s = run_arm(eval_wrapper, compare_seed, policy=np_policy, pid_baseline=False)
    m_recipe_s = run_arm(eval_wrapper, compare_seed, policy=None, pid_baseline=True)
    y_rl_s, y_recipe_s = yield_kg(m_rl_s), yield_kg(m_recipe_s)
    print(f"seed {compare_seed}: yield model={y_rl_s:8.2f} kg  recipe={y_recipe_s:8.2f} kg  "
         f"delta={y_rl_s - y_recipe_s:+8.2f} kg")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def cmp(a, key, ylabel, title):
        a.plot(m_recipe_s["t"], m_recipe_s[key], **{**REF_STYLE, "label": "recipe (same seed)"})
        a.plot(m_rl_s["t"], m_rl_s[key], **{**MODEL_STYLE, "label": "model"})
        a.axvline(WARMUP_H, color="gray", ls=":", lw=.8)
        a.set_xlabel("time (h)"); a.set_ylabel(ylabel); a.set_title(title)
        a.grid(alpha=.3); a.legend(fontsize=8)

    cmp(ax[0, 0], "P", "P (g/L)", "Penicillin (reward)")
    cmp(ax[0, 1], "Fs", "Fs (L/h)", "Sugar feed Fs (ACTION)")
    cmp(ax[1, 0], "PAA", "PAA (mg/L)", "PAA concentration")
    cmp(ax[1, 1], "Viscosity", "viscosity (cP)", "Viscosity")
    cmp(ax[1, 2], "Wt", "Wt (kg)", "Broth weight")
    ax[1, 0].axhspan(*PAA_BAND, color="green", alpha=.12)
    ax[1, 1].axhline(VISC_MAX, color="crimson", ls="--", lw=1)

    ax[0, 2].bar([0, 1], [y_recipe_s, y_rl_s], color=[REF_STYLE["color"], "C0"])
    ax[0, 2].set_xticks([0, 1]); ax[0, 2].set_xticklabels(["recipe", "model"])
    ax[0, 2].set_ylabel("yield (kg)")
    ax[0, 2].set_title(f"Yield (seed {compare_seed})  delta={y_rl_s - y_recipe_s:+.1f} kg")
    ax[0, 2].grid(alpha=.3, axis="y")
    fig.suptitle(f"Model vs recipe on shared seed {compare_seed} (identical batch realisation; only Fs differs)")
    fig.tight_layout()
    _finish(fig, out_dir, "A6_model_vs_recipe_single_seed.png", show)
    return fig, m_rl_s, m_recipe_s


def plot_fs_residual(m_rl_s, m_recipe_s, compare_seed, out_dir, show=False):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(m_recipe_s["t"], m_recipe_s["Fs"], label="recipe (baseline)", color="C1")
    ax.plot(m_rl_s["t"], m_rl_s["Fs"], label="RL (loaded policy)", color="C0")
    ax.axvline(WARMUP_H, color="gray", ls=":", lw=.8, label=f"RL on ({WARMUP_H:g} h)")
    ax.set_xlabel("time (h)"); ax.set_ylabel("Fs (L/h)")
    ax.set_title(f"Substrate feed Fs: recipe vs RL (seed {compare_seed})")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    _finish(fig, out_dir, "A7a_fs_residual.png", show)

    t_fs = np.asarray(m_rl_s["t"])
    ratio = np.asarray(m_rl_s["Fs"]) / np.maximum(np.asarray(m_recipe_s["Fs"]), 1e-9)
    a_fs = (ratio - 1.0) / FS_SCALE

    fig2, ax2 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax2[0].plot(t_fs, ratio, color="C0", drawstyle="steps-post")
    ax2[0].axhline(1.0, color="k", lw=1, ls="--")
    ax2[0].set_ylabel("Fs_rl / Fs_recipe")
    ax2[0].set_title("RL correction factor (each flat step = one decision)")
    ax2[1].plot(t_fs, a_fs, color="C3", drawstyle="steps-post")
    ax2[1].axhline(0.0, color="k", lw=1, ls="--")
    ax2[1].set_ylabel("recovered action a_fs"); ax2[1].set_xlabel("time (h)")
    for tb in np.arange(WARMUP_H, t_fs[-1], T_SAMPLING):
        for a in ax2:
            a.axvline(tb, color="grey", alpha=.2, lw=.6)
    fig2.tight_layout()
    _finish(fig2, out_dir, "A7b_recovered_action.png", show)

    n_steps = int(np.sum(np.abs(np.diff(a_fs)) > 1e-6)) + 1
    print(f"distinct action levels held over the batch: {n_steps} "
         f"(expected ~{int(CONTROL_H / T_SAMPLING)} at T_SAMPLING={T_SAMPLING:g} h)")
    return fig, fig2


def validation_learning_curve(run, eval_wrapper, out_dir, val_seeds, show=False):
    folder = str(run.dir).rstrip("/") + "/"
    base_by_seed = {s: yield_kg(run_arm(eval_wrapper, s, policy=None, pid_baseline=True)) for s in val_seeds}
    base_mean = float(np.mean(list(base_by_seed.values())))
    print(f"recipe baseline on {val_seeds}: {base_mean:.1f} kg")

    ys_by_seed = {s: [] for s in val_seeds}
    ds_by_seed = {s: [] for s in val_seeds}
    curve, spread, delta = [], [], []
    for k in range(1, run.n_trials_in_log + 1):
        pol = load_stage_policy(run, k, folder=folder)
        ys = np.array([yield_kg(run_arm(eval_wrapper, s, policy=pol, pid_baseline=False)) for s in val_seeds])
        ds = ys - np.array([base_by_seed[s] for s in val_seeds])
        for s, y, d in zip(val_seeds, ys, ds):
            ys_by_seed[s].append(y); ds_by_seed[s].append(d)
        curve.append(ys.mean()); spread.append(ys.std()); delta.append(ds.mean())
        print(f"  trial {k:>2}: yield={ys.mean():8.1f} +/- {ys.std():5.1f} kg | "
             f"paired delta vs recipe = {ds.mean():+8.1f} kg")

    curve, spread, delta = map(np.asarray, (curve, spread, delta))
    trials = np.arange(1, run.n_trials_in_log + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    seed_colors = plt.cm.tab10(np.linspace(0, 1, len(val_seeds)))
    for s, c in zip(val_seeds, seed_colors):
        ax[0].plot(trials, ys_by_seed[s], color=c, alpha=.5, lw=1, label=f"seed {s}")
        ax[1].plot(trials, ds_by_seed[s], color=c, alpha=.5, lw=1, label=f"seed {s}")
    ax[0].plot(trials, curve, marker="o", label="mean over seeds", **MODEL_STYLE)
    ax[0].axhline(base_mean, label="recipe (same seeds)", **REF_STYLE)
    ax[0].set_xlabel("trial (policy stage)"); ax[0].set_ylabel("batch yield (kg)")
    ax[0].set_title(f"Validation learning curve - fixed seeds {val_seeds}")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)

    ax[1].axhline(0.0, label="recipe (paired)", **REF_STYLE)
    ax[1].plot(trials, delta, marker="o", label="mean over seeds", **MODEL_STYLE)
    ax[1].set_xlabel("trial (policy stage)"); ax[1].set_ylabel("paired yield delta (kg)")
    ax[1].set_title("Paired improvement over recipe (seed variance cancelled)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _finish(fig, out_dir, "A8_validation_learning_curve.png", show)

    df = pd.DataFrame({"trial": trials, "mean_yield": curve, "std_yield": spread, "paired_delta": delta})
    df.to_csv(Path(out_dir) / "A8_validation_learning_curve.csv", index=False)

    verdict = "ABOVE" if delta[-1] > 0 else "BELOW"
    print(f"final policy: {curve[-1]:.1f} kg vs recipe {base_mean:.1f} kg "
         f"(paired delta {delta[-1]:+.1f} kg) -> {verdict} recipe")
    return fig, df


# ---------------------------------------------------------------------------
# Section B: training progression
# ---------------------------------------------------------------------------

def plot_training_progression(run, ref, ref_lbl, out_dir, show=False):
    if run.monitors is None:
        print("no monitor.pkl -> skipping per-episode monitor plots")
        return None
    monitors = run.monitors
    state_hist = run.log["state_samples_history"]
    n_ep, n_expl = run.n_ep, run.num_explorations

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    finals = np.array([final_P(b)[0] for b in state_hist])
    ax[0, 0].plot(np.arange(len(finals)), finals, marker="o", label=f"seed {run.train_seed}")
    ax[0, 0].axhline(ref["final_P"], label=ref_lbl, **REF_STYLE)

    yields, yields_gated = [], []
    for i, m in enumerate(monitors):
        c = _ep_color(i, n_ep, n_expl)
        ax[0, 1].plot(m["t"], m["PAA"], color=c, lw=1, alpha=.85)
        ax[0, 2].plot(m["t"], m["Viscosity"], color=c, lw=1, alpha=.85)
        ax[1, 0].plot(m["t"], m["Fpaa"], color=c, lw=1, alpha=.85)
        ax[1, 1].plot(m["t"], m["P"], color=c, lw=1, alpha=.85)
        yields.append(yield_kg(m))
        yields_gated.append(feasibility_gated_yield_kg(m))
    yields = np.array(yields)
    yields_gated = np.array(yields_gated)
    # Primary metric is feasibility-gated (an envelope breach zeroes the episode -- see
    # experiments/eval_utils.feasibility_gated_yield_kg); raw yield is only overlaid on the
    # episodes gating actually changed, so the "cost of infeasibility" stays visible rather than
    # silently disappearing behind the headline number.
    breached = yields_gated < yields
    ax[1, 2].bar(np.arange(n_ep), yields_gated, color=[_ep_color(i, n_ep, n_expl) for i in range(n_ep)])
    if breached.any():
        ax[1, 2].scatter(np.arange(n_ep)[breached], yields[breached], marker="x", color="crimson",
                         zorder=3, label="raw yield (envelope breach -> gated to 0)")
        print(f"{breached.sum()}/{n_ep} episodes breached the operating envelope "
              f"(Wt overflow or viscosity collapse) -> gated to 0 kg in the plot above")
    ax[1, 2].axhline(ref["yield_gated"], label=ref_lbl, **REF_STYLE)

    for axis, key in [(ax[0, 1], "PAA"), (ax[0, 2], "Viscosity"), (ax[1, 0], "Fpaa"), (ax[1, 1], "P")]:
        if key in ref:
            rt, ry = ref[key]
            axis.plot(rt, ry, label=ref_lbl, **REF_STYLE)

    ax[0, 0].set_title("Final penicillin conc per episode")
    ax[0, 0].set_xlabel("episode (exploration then trials)"); ax[0, 0].set_ylabel("P (g/L)")
    ax[0, 0].grid(alpha=.3)

    ax[0, 1].axhspan(*PAA_BAND, color="green", alpha=.12, label="allowed band")
    ax[0, 1].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[0, 1].set_title("PAA conc (all episodes)"); ax[0, 1].set_xlabel("time (h)")
    ax[0, 1].set_ylabel("PAA (mg/L)"); ax[0, 1].grid(alpha=.3)

    ax[0, 2].axhline(VISC_MAX, color="crimson", ls="--", label=f"limit {VISC_MAX:.0f} cP")
    ax[0, 2].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[0, 2].set_title("Viscosity (all episodes)"); ax[0, 2].set_xlabel("time (h)")
    ax[0, 2].set_ylabel("viscosity (cP)"); ax[0, 2].grid(alpha=.3)

    ax[1, 0].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[1, 0].axhspan(FPAA_MIN, FPAA_MAX, color="orange", alpha=.06,
                     label=f"clamp [{FPAA_MIN:.0f},{FPAA_MAX:.0f}]")
    ax[1, 0].set_title("Fpaa setpoint (all episodes)"); ax[1, 0].set_xlabel("time (h)")
    ax[1, 0].set_ylabel("Fpaa (L/h)"); ax[1, 0].grid(alpha=.3)

    ax[1, 1].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[1, 1].set_title("Penicillin trajectories (all episodes)"); ax[1, 1].set_xlabel("time (h)")
    ax[1, 1].set_ylabel("P (g/L)"); ax[1, 1].grid(alpha=.3)
    ax[1, 1].plot([], [], color="0.72", label="exploration")
    _episode_colorbar(fig, ax[1, 1], n_ep, n_expl, label="trial episode (early -> late)")

    ax[1, 2].set_title("Penicillin yield per episode (feasibility-gated)")
    ax[1, 2].set_xlabel("episode"); ax[1, 2].set_ylabel("feasibility-gated yield (kg)")
    ax[1, 2].grid(alpha=.3, axis="y")

    _mark_pivot([ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]], run.pivot_hours, run.blend_half_width_hours)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]):
        a.legend(fontsize=6, ncol=2)

    fig.suptitle("Dual-phase MC-PILCO training progression")
    fig.tight_layout()
    _finish(fig, out_dir, "B1_training_progression.png", show)
    return fig


def plot_all_observations(run, ref, ref_lbl, out_dir, show=False):
    state_hist = run.log["state_samples_history"]
    n_ep, n_expl = run.n_ep, run.num_explorations

    ncol = 3
    n_obs = len(STATE_NAMES)
    nrow = int(np.ceil(n_obs / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 3.6 * nrow), squeeze=False)
    axes = ax.ravel()
    for j, name in enumerate(STATE_NAMES):
        a = axes[j]
        lo, hi = STATE_RANGES[name]
        for i, b in enumerate(state_hist):
            y = _denorm_phys(np.asarray(b)[:, j], name)
            a.plot(decision_time_grid(len(y)), y, color=_ep_color(i, n_ep, n_expl), lw=1, alpha=.85)
        if name in ref:
            rt, ry = ref[name]
            a.plot(rt, ry, label=ref_lbl, **REF_STYLE)
        if name == "PAA":
            a.axhspan(*PAA_BAND, color="green", alpha=.12, label="allowed band")
        a.axvline(WARMUP_H, color="gray", ls=":", lw=.8)
        _mark_pivot(a, run.pivot_hours, run.blend_half_width_hours)
        _lo_p, _hi_p = decode_state_value(name, lo), decode_state_value(name, hi)
        a.set_title(name); a.set_xlabel("time (h)"); a.set_ylabel(f"{name} (range {_lo_p:g}..{_hi_p:g})")
        a.grid(alpha=.3)
        if name in ref or name == "PAA":
            a.legend(fontsize=7)
    axes[0].plot([], [], color="0.72", label="exploration")
    _episode_colorbar(fig, list(axes), n_ep, n_expl, label="trial episode (early -> late)")
    axes[0].legend(fontsize=7)
    for k in range(n_obs, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"All observations per episode — seed {run.params.get('seed')}", y=1.002)
    fig.tight_layout()
    _finish(fig, out_dir, "B2_all_observations.png", show)
    return fig


def plot_fs_all_episodes(run, out_dir, show=False):
    if run.monitors is None:
        print("no monitor.pkl -> skipping")
        return None
    monitors = run.monitors
    n_ep, n_expl = run.n_ep, run.num_explorations
    fs_recipe = Recipe(FS_DEFAULT_PROFILE, FS)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(monitors):
        ax.plot(m["t"], m["Fs"], color=_ep_color(i, n_ep, n_expl), lw=1, alpha=.85)
    t_ref = np.arange(WARMUP_H, 230.0 + STEP_IN_HOURS, STEP_IN_HOURS)
    y_ref = np.array([fs_recipe.get_value_at(float(tt)) for tt in t_ref])
    ax.plot(t_ref, y_ref, label="recipe profile", **REF_STYLE)
    ax.axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax.plot([], [], color="0.72", label="exploration")
    _episode_colorbar(fig, ax, n_ep, n_expl, label="trial episode (early -> late)")
    ax.set_title("Fs (sugar feed): recipe vs RL residual (all episodes)")
    ax.set_xlabel("time (h)"); ax.set_ylabel("Fs (L/h)")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _finish(fig, out_dir, "B3_fs_all_episodes.png", show)
    return fig


# ---------------------------------------------------------------------------
# Section C: GP model diagnostics
# ---------------------------------------------------------------------------

def reconstruct_gp_agent(run, idx=None, get_config_fn=None):
    """Build a PenSimMCPILCOMultiPhase and load the trial-`idx` GP model from run.log (no
    training). See module docstring point 1 for why this differs from the single-phase
    reconstruct(): DualPhaseModelLearning's gp_inputs/gp_output_list/gp_list/norm_list are
    READ-ONLY concatenations, so weights are assigned into .phase1/.phase2 directly.

    Rebuilds cfg from run.params itself rather than reusing run.cfg, so get_config_fn must
    match whatever was passed to load_run() for this run (regular vs baseline) -- otherwise
    this reconstructs phase1/phase2 with the WRONG model_learning class. Since
    RBF_WtMassBalance/RBF_RecipeMean share their exact parameter set with plain RBF (only
    get_mean differs), a mismatched class still load_state_dict's without error -- it just
    silently reattaches a prior mean the checkpoint was never fit against. See
    action_sensitivity_multi_phase_baseline.py for the baseline call site."""
    get_config_fn = get_config if get_config_fn is None else get_config_fn
    idx = _resolve_trial(run.log, idx)
    cfg = get_config_fn(**_build_cfg_kwargs(run.params))
    cfg["mc_pilco_init"]["log_path"] = None
    agent = PenSimMCPILCOMultiPhase(pensim_wrapper=PenSimWrapper(**cfg["wrapper_par"]),
                                    **cfg["mc_pilco_init"])
    agent.state_samples_history = run.log["state_samples_history"]
    agent.input_samples_history = run.log["input_samples_history"]
    agent.noiseless_states_history = run.log.get("noiseless_states_history",
                                                  run.log["state_samples_history"])

    ml = agent.model_learning
    n_gp = STATE_DIM
    gp_inputs_all = run.log[f"gp_inputs_{idx}"]
    gp_outputs_all = run.log[f"gp_output_list_{idx}"]
    params_all = run.log[f"parameters_gp_{idx}"]
    # gp_outputs_all/params_all: [phase1 x STATE_DIM, phase2 x STATE_DIM] (see
    # DualPhaseModelLearning.gp_output_list/gp_list). gp_inputs_all: [phase1 rows ++ phase2
    # rows] (see .gp_inputs) -- split its row boundary using each phase's own channel-0
    # output length rather than recomputing an episode count.
    n1 = gp_outputs_all[0].shape[0]
    n2 = gp_outputs_all[n_gp].shape[0]
    assert gp_inputs_all.shape[0] == n1 + n2, (
        f"gp_inputs row count {gp_inputs_all.shape[0]} != phase1 ({n1}) + phase2 ({n2})")

    for sub, out_slice, par_slice, inp_slice in [
        (ml.phase1, gp_outputs_all[:n_gp], params_all[:n_gp], gp_inputs_all[:n1]),
        (ml.phase2, gp_outputs_all[n_gp:], params_all[n_gp:], gp_inputs_all[n1:n1 + n2]),
    ]:
        sub.gp_inputs = inp_slice
        sub.gp_output_list = out_slice
        sub.num_samples = inp_slice.shape[0]
        sub.dim_state = len(STATE_NAMES)
        sub.init_gp_models()
        for k in range(sub.num_gp):
            sub.gp_list[k].load_state_dict(par_slice[k])
            sub.norm_list[k] = (torch.max(torch.abs(sub.gp_output_list[k]))
                                if getattr(sub, "flg_norm", False)
                                else torch.tensor(1.0, dtype=agent.dtype))
        with torch.no_grad():
            for k in range(sub.num_gp):
                sub.pretrain_gp(k)
        sub.set_eval_mode()
    return agent, idx


def _kstep_errors(agent, batch_idx, horizons, target_origins=60):
    """Sliding-origin k-step prediction error for one batch (open-loop, recorded actions).
    Returns rmse[len(horizons), STATE_DIM] in NORMALISED state space."""
    ml = agent.model_learning
    true = torch.tensor(agent.state_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    inp = torch.tensor(agent.input_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    T = true.shape[0]
    kmax = max(horizons)
    stride = max(1, int(np.ceil((T - 1) / target_origins)))
    sq = {k: [] for k in horizons}
    for t0 in range(0, T - 1, stride):
        steps = min(kmax, T - 1 - t0)
        cur = true[t0:t0 + 1, :]
        ml.reset_step_counter(t0)  # dual-phase: route decisions t0, t0+1, ... through
                                    # whichever phase actually owns them
        for j in range(1, steps + 1):
            cur, _, _ = ml.get_next_state(current_state=cur,
                                          current_input=inp[t0 + j - 1:t0 + j, :],
                                          particle_pred=False)
            if j in sq:
                err = (cur - true[t0 + j:t0 + j + 1, :]).ravel()
                sq[j].append((err ** 2).detach().cpu().numpy())
    rmse = np.full((len(horizons), STATE_DIM), np.nan)
    for r, k in enumerate(horizons):
        if sq[k]:
            rmse[r] = np.sqrt(np.mean(np.stack(sq[k], 0), axis=0))
    return rmse


def one_step_fit(gp_agent, gp_idx, out_dir, show=False):
    """C.1/C.3/C.4 combined: one-step GP fit against the trial-gp_idx training batch, scored
    per (phase, channel). get_model_learning_performance's targets/means/per_dim_mse are
    2*STATE_DIM long (phase1's channels, then phase2's), since
    DualPhaseModelLearning.get_gp_estimate_from_data evaluates each phase's GPs on its own
    data segment rather than forcing the whole trajectory through one model."""
    with torch.no_grad():
        _, targets, means, _ = gp_agent.get_model_learning_performance(gp_idx)
    per_dim_mse = [float(((targets[k] - means[k]) ** 2).mean()) for k in range(len(targets))]

    loX, hiX = STATE_RANGES["X"]; loP, hiP = STATE_RANGES["P"]
    results = {}
    for phase_name, xi, pi in [("phase1", X_IDX_P1, P_IDX_P1), ("phase2", X_IDX_P2, P_IDX_P2)]:
        tgt_dx = _denorm_delta(targets[xi].ravel(), loX, hiX); prd_dx = _denorm_delta(means[xi].ravel(), loX, hiX)
        tgt_dp = _denorm_delta(targets[pi].ravel(), loP, hiP); prd_dp = _denorm_delta(means[pi].ravel(), loP, hiP)
        r2_x = 1.0 - float(((tgt_dx - prd_dx) ** 2).sum()) / (float(((tgt_dx - tgt_dx.mean()) ** 2).sum()) or 1.0)
        r2_p = 1.0 - float(((tgt_dp - prd_dp) ** 2).sum()) / (float(((tgt_dp - tgt_dp.mean()) ** 2).sum()) or 1.0)
        results[phase_name] = dict(tgt_dx=tgt_dx, prd_dx=prd_dx, tgt_dp=tgt_dp, prd_dp=prd_dp,
                                   r2_x=r2_x, r2_p=r2_p)

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for col, phase_name in enumerate(("phase1", "phase2")):
        r = results[phase_name]
        a = ax[0, col]
        a.scatter(r["tgt_dx"], r["prd_dx"], s=12, alpha=.6, color="C0")
        lim = [min(r["tgt_dx"].min(), r["prd_dx"].min()), max(r["tgt_dx"].max(), r["prd_dx"].max())]
        a.plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
        a.set_title(f"{phase_name}: one-step dX (R^2={r['r2_x']:.3f})")
        a.set_xlabel("actual dX (g/L per step)"); a.set_ylabel("GP predicted dX")
        a.grid(alpha=.3); a.legend(fontsize=8)

        a = ax[1, col]
        a.scatter(r["tgt_dp"], r["prd_dp"], s=12, alpha=.6, color="C1")
        lim = [min(r["tgt_dp"].min(), r["prd_dp"].min()), max(r["tgt_dp"].max(), r["prd_dp"].max())]
        a.plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
        a.set_title(f"{phase_name}: one-step dP (R^2={r['r2_p']:.3f})")
        a.set_xlabel("actual dP (g/L per step)"); a.set_ylabel("GP predicted dP")
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"C.1/C.3 — one-step dX/dP fit by phase, trial {gp_idx}")
    fig.tight_layout()
    _finish(fig, out_dir, "C1_one_step_scatter.png", show)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = ["C0"] * STATE_DIM + ["C2"] * STATE_DIM
    ax2.bar(range(len(per_dim_mse)), per_dim_mse, color=colors)
    ax2.set_xticks(range(len(DUAL_STATE_NAMES))); ax2.set_xticklabels(DUAL_STATE_NAMES, rotation=45, ha="right")
    ax2.set_title(f"Per-dim one-step MSE by phase — trial {gp_idx}"); ax2.set_ylabel("MSE (normalised)")
    ax2.grid(alpha=.3, axis="y")
    ax2.bar([], [], color="C0", label="phase 1"); ax2.bar([], [], color="C2", label="phase 2")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    _finish(fig2, out_dir, "C4_per_dim_mse.png", show)

    pd.DataFrame({"channel": DUAL_STATE_NAMES, "mse_norm": per_dim_mse}).to_csv(
        Path(out_dir) / "C4_per_dim_mse.csv", index=False)

    print(f"one-step R^2 -- phase1: X={results['phase1']['r2_x']:.3f} P={results['phase1']['r2_p']:.3f} | "
         f"phase2: X={results['phase2']['r2_x']:.3f} P={results['phase2']['r2_p']:.3f}")
    return per_dim_mse, results


def plot_multistep_rollout(gp_agent, gp_idx, ho_idx, has_ho, out_dir, pivot_hours,
                           blend_half_width_hours=None, show=False):
    with torch.no_grad():
        pred, true, _ = gp_agent.get_rollout_prediction_performance(gp_idx)
        if has_ho:
            pred_ho, true_ho, _ = gp_agent.get_rollout_prediction_performance(ho_idx)

    t_ms = decision_time_grid(pred.shape[0])
    x_pred = _denorm_phys(pred[:, X_IDX], "X"); x_true = _denorm_phys(true[:, X_IDX], "X")
    p_pred = _denorm_phys(pred[:, P_IDX], "P"); p_true = _denorm_phys(true[:, P_IDX], "P")
    if has_ho:
        t_ho = decision_time_grid(pred_ho.shape[0])
        x_pred_ho = _denorm_phys(pred_ho[:, X_IDX], "X"); x_true_ho = _denorm_phys(true_ho[:, X_IDX], "X")
        p_pred_ho = _denorm_phys(pred_ho[:, P_IDX], "P"); p_true_ho = _denorm_phys(true_ho[:, P_IDX], "P")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(t_ms, x_true, "k-", lw=2, label=f"in-sample true (batch {gp_idx})")
    ax[0].plot(t_ms, x_pred, "C1--", lw=2, label="in-sample GP")
    if has_ho:
        ax[0].plot(t_ho, x_true_ho, "-", color="steelblue", lw=2, label=f"held-out true (batch {ho_idx})")
        ax[0].plot(t_ho, x_pred_ho, "--", color="crimson", lw=2, label="held-out GP")
    ax[0].set_title("Multi-step X (biomass): in-sample vs held-out")
    ax[0].set_xlabel("time (h)"); ax[0].set_ylabel("X (g/L)"); ax[0].grid(alpha=.3)

    ax[1].plot(t_ms, p_true, "k-", lw=2, label=f"in-sample true (batch {gp_idx})")
    ax[1].plot(t_ms, p_pred, "C1--", lw=2, label="in-sample GP")
    if has_ho:
        ax[1].plot(t_ho, p_true_ho, "-", color="steelblue", lw=2, label=f"held-out true (batch {ho_idx})")
        ax[1].plot(t_ho, p_pred_ho, "--", color="crimson", lw=2, label="held-out GP")
    ax[1].set_title("Multi-step P: in-sample vs held-out")
    ax[1].set_xlabel("time (h)"); ax[1].set_ylabel("P (g/L)"); ax[1].grid(alpha=.3)

    _mark_pivot(ax, pivot_hours, blend_half_width_hours)
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle(f"GP-vs-simulator multi-step diagnostic — model@trial {gp_idx}"
                + (f" (held-out batch {ho_idx})" if has_ho else " (in-sample only)"))
    fig.tight_layout()
    _finish(fig, out_dir, "C2_multistep_rollout.png", show)
    print(f"X rollout in-sample: final actual={x_true[-1]:.2f} g/L, GP={x_pred[-1]:.2f} g/L")
    print(f"P rollout in-sample: final actual={p_true[-1]:.2f} g/L, GP={p_pred[-1]:.2f} g/L")
    return fig


def _particle_rollout(agent, idx, N=100, seed=0):
    """N particles through the RECORDED action sequence of batch `idx` (particle_pred=True).
    Sequential walk from decision 0 -- reset_step_counter(0) starts the phase router at the
    true origin (see module docstring point 2). get_next_state now returns full predictive
    variance (epistemic + observation-noise sigma_n^2, see Model_learning.get_next_state /
    _update_calib_factor), so the sampled particles already carry the noise floor -- no
    eval-side correction here."""
    ml = agent.model_learning
    ml.set_eval_mode()
    S = np.asarray(agent.state_samples_history[idx])
    U = np.asarray(agent.input_samples_history[idx])
    T, D = S.shape
    torch.manual_seed(seed)
    x = torch.tensor(np.tile(S[0], (N, 1)), dtype=agent.dtype, device=agent.device)
    out = np.zeros((T, N, D)); out[0] = x.detach().cpu().numpy()
    ml.reset_step_counter()
    with torch.no_grad():
        for t in range(1, T):
            u = torch.tensor(np.tile(U[t - 1], (N, 1)), dtype=agent.dtype, device=agent.device)
            x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=True)
            out[t] = x.detach().cpu().numpy()
    return out


def plot_particle_bands(gp_agent, gp_idx, ho_idx, has_ho, out_dir, pivot_hours,
                        blend_half_width_hours=None, n_part=100, show=False):
    with torch.no_grad():
        pred, _, _ = gp_agent.get_rollout_prediction_performance(gp_idx)
    cols = [(gp_idx, "IN-SAMPLE", pred)]
    if has_ho:
        with torch.no_grad():
            pred_ho, _, _ = gp_agent.get_rollout_prediction_performance(ho_idx)
        cols.append((ho_idx, "HELD-OUT", pred_ho))

    tr = {idx: _particle_rollout(gp_agent, idx, N=n_part) for idx, _, _ in cols}
    tt = {idx: decision_time_grid(tr[idx].shape[0]) for idx, _, _ in cols}
    xlim = (min(tt[i][0] for i, _, _ in cols), max(tt[i][-1] for i, _, _ in cols))

    nrow, ncol = len(STATE_NAMES), len(cols)
    fig, ax = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 2.5 * nrow), squeeze=False)
    rows = []
    for r, name in enumerate(STATE_NAMES):
        k = STATE_NAMES.index(name)
        lo, hi = STATE_RANGES[name]
        lo_p, hi_p = decode_state_value(name, lo), decode_state_value(name, hi)
        pad = 0.05 * (hi_p - lo_p)
        for c, (bidx, tag, meanroll) in enumerate(cols):
            v = _denorm_phys(tr[bidx][:, :, k], name)
            p10, p50, p90 = (np.percentile(v, 10, axis=1), np.median(v, axis=1), np.percentile(v, 90, axis=1))
            truth = _denorm_phys(np.asarray(gp_agent.state_samples_history[bidx])[:, k], name)
            mline = _denorm_phys(np.asarray(meanroll)[:, k], name)
            t_ = tt[bidx]
            a = ax[r, c]
            a.fill_between(t_, p10, p90, color="C0", alpha=.22, label=f"GP 10-90% ({n_part} particles)")
            a.plot(t_, truth, "k-", lw=1.8, label="simulator (truth)")
            a.plot(t_, mline, "C1--", lw=1.8, label="GP mean rollout")
            a.plot(t_, p50, "C0-", lw=1.5, label="GP particle median")
            _mark_pivot(a, pivot_hours, blend_half_width_hours)
            a.set_xlim(*xlim); a.set_ylim(lo_p - pad, hi_p + pad)
            a.set_title(f"{name} - {tag} (batch {bidx})", fontsize=9)
            a.set_ylabel(name, fontsize=8); a.grid(alpha=.3); a.tick_params(labelsize=7)
            if r == nrow - 1:
                a.set_xlabel("time (h)", fontsize=8)
            if r == 0 and c == 0:
                a.legend(fontsize=6.5, loc="upper left")
            cov = float(np.mean((truth >= p10) & (truth <= p90)))
            gap = abs(float(mline[-1]) - float(p50[-1]))
            rows.append({"channel": name, "batch_tag": tag, "batch_idx": bidx,
                        "coverage_10_90": cov, "mean_median_gap_final": gap})
    fig.suptitle(f"C.2b — GP multi-step uncertainty, all channels — model@trial {gp_idx}", y=1.001)
    fig.tight_layout()
    _finish(fig, out_dir, "C2b_particle_bands.png", show)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "C2b_particle_bands.csv", index=False)
    return fig


def _one_step_std_residuals(agent, batch_idx):
    """Teacher-forced one-step standardised residuals, per channel. One-off probe AT each t0
    (not a sequential walk), so reset_step_counter(t0) is called before every single call --
    see module docstring point 2. get_next_state now returns full predictive variance
    (epistemic + observation-noise sigma_n^2, see Model_learning.get_next_state /
    _update_calib_factor), so dvar is already the right denominator -- no eval-side
    correction here."""
    ml = agent.model_learning
    tr = torch.tensor(agent.state_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    ip = torch.tensor(agent.input_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    N = tr.shape[0]
    z = np.full((N - 1, STATE_DIM), np.nan)
    with torch.no_grad():
        for t0 in range(N - 1):
            ml.reset_step_counter(t0)
            nxt, _dmean, dvar = ml.get_next_state(current_state=tr[t0:t0 + 1, :],
                                                  current_input=ip[t0:t0 + 1, :], particle_pred=False)
            std = torch.sqrt(dvar).ravel()
            resid = (nxt - tr[t0 + 1:t0 + 2, :]).ravel()
            z[t0] = (resid / std).detach().cpu().numpy()
    return z


def plot_calibration(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False):
    z_in = _one_step_std_residuals(gp_agent, gp_idx)
    z_ho = _one_step_std_residuals(gp_agent, ho_idx) if has_ho else None
    xg = np.linspace(-4, 4, 200)
    npdf = np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(1, STATE_DIM, figsize=(3.2 * STATE_DIM, 3.4), squeeze=False)
    series = [(z_in, "in-sample", "C1")] + ([(z_ho, "held-out", "steelblue")] if has_ho else [])
    rows = []
    for k, name in enumerate(STATE_NAMES):
        a = ax[0, k]
        for z, tag, col in series:
            zc = z[:, k]; zc = zc[np.isfinite(zc)]
            if zc.size == 0:
                continue
            a.hist(zc, bins=20, range=(-4, 4), density=True, alpha=.5, color=col, label=tag)
            cov = float(np.mean(np.abs(zc) < 1.96))
            rows.append({"channel": name, "tag": tag, "mean": float(zc.mean()), "std": float(zc.std()),
                        "coverage_1.96": cov})
        a.plot(xg, npdf, "k--", lw=1.2, label="N(0,1)")
        a.set_title(name); a.set_xlabel("standardised residual z"); a.grid(alpha=.3)
        if k == 0:
            a.set_ylabel("density"); a.legend(fontsize=7)
    fig.suptitle(f"C.4b — one-step teacher-forced standardised residuals — model@trial {gp_idx}")
    fig.tight_layout()
    _finish(fig, out_dir, "C4b_calibration.png", show)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "C4b_calibration.csv", index=False)
    return fig


def _one_step_abs_err(agent, batch_idx):
    ml = agent.model_learning
    tr = torch.tensor(agent.state_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    ip = torch.tensor(agent.input_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    N = tr.shape[0]
    grid = decision_time_grid(N)
    errs = np.full((N - 1, STATE_DIM), np.nan)
    with torch.no_grad():
        for t0 in range(N - 1):
            ml.reset_step_counter(t0)
            nxt, _, _ = ml.get_next_state(current_state=tr[t0:t0 + 1, :],
                                          current_input=ip[t0:t0 + 1, :], particle_pred=False)
            errs[t0] = (nxt - tr[t0 + 1:t0 + 2, :]).abs().ravel().detach().cpu().numpy()
    eX = _denorm_delta(errs[:, X_IDX], *STATE_RANGES["X"])
    eP = _denorm_delta(errs[:, P_IDX], *STATE_RANGES["P"])
    return grid[:-1], eX, eP


def plot_local_error(gp_agent, gp_idx, ho_idx, has_ho, out_dir, pivot_hours,
                     blend_half_width_hours=None, show=False):
    t_in, eX_in, eP_in = _one_step_abs_err(gp_agent, gp_idx)
    if has_ho:
        t_ho, eX_ho, eP_ho = _one_step_abs_err(gp_agent, ho_idx)

    nbins = min(10, len(t_in))
    edges = np.linspace(t_in.min(), t_in.max(), nbins + 1)
    bc = 0.5 * (edges[:-1] + edges[1:])

    def rmse_by_bin(tt, e):
        b = np.clip(np.digitize(tt, edges) - 1, 0, nbins - 1)
        return np.array([np.sqrt((e[b == kk] ** 2).mean()) if (b == kk).any() else np.nan for kk in range(nbins)])

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, name, ein, eho in [(ax[0], "X", eX_in, eX_ho if has_ho else None),
                              (ax[1], "P", eP_in, eP_ho if has_ho else None)]:
        a.scatter(t_in, ein, s=16, alpha=.35, color="C1", label="per-step in-sample")
        a.plot(bc, rmse_by_bin(t_in, ein), "-o", color="C1", lw=2, label="in-sample binned RMSE")
        if has_ho:
            a.scatter(t_ho, eho, s=16, alpha=.35, color="steelblue", label="per-step held-out")
            a.plot(bc, rmse_by_bin(t_ho, eho), "-o", color="steelblue", lw=2, label="held-out binned RMSE")
        a.set_title(f"One-step |error| vs batch time — {name}")
        a.set_xlabel("batch time at prediction origin (h)"); a.set_ylabel(f"one-step |error| ({name}, g/L)")
        a.grid(alpha=.3); a.legend(fontsize=7)
    _mark_pivot(ax, pivot_hours, blend_half_width_hours)
    fig.suptitle("Local (one-step) GP error across the batch")
    fig.tight_layout()
    _finish(fig, out_dir, "C5_local_error.png", show)
    return fig


def plot_kstep_growth(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False,
                      horizons=(1, 5, 10, 15, 20, 25, 30, 35, 40, 45)):
    Tb = gp_agent.state_samples_history[gp_idx].shape[0]
    horizons = np.array([k for k in horizons if 1 <= k <= Tb - 1])
    rmse_in = _kstep_errors(gp_agent, gp_idx, horizons)
    rmse_ho = _kstep_errors(gp_agent, ho_idx, horizons) if has_ho else None

    fig, ax = plt.subplots(figsize=(8, 6))
    rows = []
    for dim, name, color in [(X_IDX, "X", "crimson"), (P_IDX, "P", "steelblue")]:
        lo, hi = STATE_RANGES[name]
        if has_ho:
            y = _denorm_delta(rmse_ho[:, dim], lo, hi); m = np.isfinite(y)
            ax.plot(horizons[m], y[m], "-o", color=color, lw=2, label=f"{name} held-out (batch {ho_idx})")
        y_in = _denorm_delta(rmse_in[:, dim], lo, hi); m_in = np.isfinite(y_in)
        ax.plot(horizons[m_in], y_in[m_in], "--o", color=color, lw=1.5, alpha=.45,
                label=f"{name} in-sample (batch {gp_idx})")
        for r, k in enumerate(horizons):
            rows.append({"dim": name, "horizon_steps": int(k), "horizon_hours": float(k * T_SAMPLING),
                        "rmse_in_sample": float(_denorm_delta(rmse_in[r, dim], lo, hi)),
                        "rmse_held_out": float(_denorm_delta(rmse_ho[r, dim], lo, hi)) if has_ho else None})
    ax.set_xticks(horizons); ax.set_xticklabels([str(k) for k in horizons])
    ax.set_xlabel(f"prediction horizon k (steps)  [1 step = {T_SAMPLING:g} h]"); ax.set_ylabel("RMSE (g/L)")
    ax.set_title("k-step-ahead prediction error growth\n(sliding-origin, open-loop)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, out_dir, "C6_kstep_growth.png", show)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "C6_kstep_growth.csv", index=False)
    return fig

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

from evaluations import torch_cpu_compat  # noqa: F401  -- load GPU-trained runs on a CPU box
from utils.recipe import Recipe
from utils.constants import STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import FS, FS_DEFAULT_PROFILE

from mcpilco.config_dual_phase import get_config
from mcpilco.model_learning_det_time import DETERMINISTIC_CHANNELS
from mcpilco.model_learning_dual_phase import _BLEND_SKIP_EPS
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
                    "pms_visc_delay", "use_offline_measurements",
                    # Absent in note.txt for runs predating these params -- correctly falls
                    # through to get_config's own hardcoded defaults in _build_cfg_kwargs below,
                    # which IS what those older runs actually trained with (no setdefault needed,
                    # unlike the two Viscosity-delay flags above).
                    "cost_function", "num_explorations", "num_high_feed_probes",
                    # Training-split coordinate (see model_learning_dual_phase.py). Absent in
                    # note.txt for runs predating it -- falling through to get_config's own
                    # default ("time") is correct there, since that IS what they trained with.
                    "pivot_mode", "pivot_bm",
                    # --onEachRollout (rollout blend on biomass). Absent in every
                    # note.txt predating the flag -> get_config default False ->
                    # those runs reconstruct with the time sigmoid, as trained.
                    "on_each_rollout", "blend_half_width_bm")


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

    @property
    def pivot_mode(self):
        """"time" (split every batch at pivot_hours) or "biomass" (split each batch at its own
        X*Wt crossing). Absent in note.txt for runs predating the flag -- those are all "time"."""
        return str(self.params.get("pivot_mode", "time"))

    @property
    def pivot_bm(self):
        from mcpilco.model_learning_dual_phase import BM_PIVOT_DEFAULT
        return float(self.params.get("pivot_bm", BM_PIVOT_DEFAULT))

    @property
    def on_each_rollout(self):
        """True if the ROLLOUT BLEND (not just the training split) ran on the biomass
        coordinate. Absent in note.txt for every run predating --onEachRollout -> False."""
        return str(self.params.get("on_each_rollout", "False")).lower() == "true"


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


def _mark_pivot(ax, pivot_hours, blend_half_width_hours=None, pivot_mode="time"):
    """Marker at the dual-phase pivot -- the CENTER of the sigmoid blend that combines
    phase-1/phase-2 GP predictions -- on any panel whose x-axis is batch time. When
    blend_half_width_hours is given, also shades the ~1%-99% transition window
    (pivot +- half_width) so the plot shows a region, not a hard cutoff that no longer exists.

    pivot_mode only changes the LABEL, never the geometry: the blend is the time sigmoid centred
    on pivot_hours in both modes. Under pivot_mode="time" pivot_hours legitimately means two
    things at once -- the training-split point AND the blend centre -- so the bare word "pivot"
    is unambiguous. Under "biomass" the split moves to each batch's own X*Wt crossing (see
    model_learning_dual_phase.py) and pivot_hours means ONLY the blend centre, so labelling this
    line "pivot" invites reading it as the split point and concluding the biomass split was
    ignored. Say "blend centre" there instead; C.0a/C.0f report the actual split."""
    lbl = "blend centre" if pivot_mode == "biomass" else "pivot"
    for a in np.atleast_1d(ax).ravel():
        if blend_half_width_hours is not None:
            a.axvspan(pivot_hours - blend_half_width_hours, pivot_hours + blend_half_width_hours,
                      color="purple", alpha=0.06, zorder=0,
                      label=f"blend window (+-{blend_half_width_hours:g} h)")
        a.axvline(pivot_hours, color="purple", ls=":", lw=1.2, label=f"{lbl} ({pivot_hours:g} h)")


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

    _mark_pivot([ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]], run.pivot_hours,
                run.blend_half_width_hours, pivot_mode=run.pivot_mode)
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
        _mark_pivot(a, run.pivot_hours, run.blend_half_width_hours, pivot_mode=run.pivot_mode)
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


def _bm_max0_at(ml, states, t0):
    """Running-max biomass over `states[:t0+1]`, for seeding a jump to decision t0.

    Several diagnostics here deliberately probe decision t0 directly -- reset_step_counter(t0)
    then one or a few get_next_state calls -- instead of replaying the batch from 0, because
    they sweep every origin and replaying each would be quadratic. Under --onEachRollout the
    blend weight is a function of biomass ACCUMULATED SINCE THE START OF THE BATCH, so such a
    jump starts with an empty running max; DualPhaseModelLearning._blend_weight raises rather
    than return the resulting under-weighted phase 2 silently. `states` is the real logged
    trajectory those probes are teacher-forced on, which is exactly what the running max is
    defined over, so the correct seed is available for free at every call site.

    Returns None -- and reset_step_counter then behaves exactly as before -- whenever the flag
    is off, so every pre-existing (time / stage-1 biomass) run is untouched."""
    if not getattr(ml, "on_each_rollout", False):
        return None
    from mcpilco.model_learning_dual_phase import _bm_from_states
    bm = _bm_from_states(states[:t0 + 1])
    m = float(bm.max()) if torch.is_tensor(bm) else float(np.max(bm))
    return torch.tensor([m], dtype=ml.dtype, device=ml.device)


def _load_split_log(run):
    """This run's per-trajectory training splits as a DataFrame [step, hours, crossed], or None.

    None covers both legitimate absences: pivot_mode="time" (the split is the constant
    pivot_step, so the driver writes nothing) and runs trained before the driver dumped
    split_log.pkl at all. Callers treat None as "no split info available", never as an error.
    """
    if run.pivot_mode != "biomass":
        return None
    path = Path(run.dir) / "split_log.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pd.DataFrame(pickle.load(fh), columns=["step", "hours", "crossed"])


def _weights_equal(a, b):
    """Blend weights compare as plain floats under the time sigmoid and as per-particle tensors
    under --onEachRollout, so equality needs a type-aware helper: `a == b` on two tensors yields
    a tensor, and bool() of that raises for anything but a 1-element result."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b)
                and a.shape == b.shape and bool(torch.equal(a, b)))
    return a == b


def _fmt_weight(w):
    return f"{float(w.reshape(-1)[0]):.6f}" if torch.is_tensor(w) else f"{w:.6f}"


def _weights_along_rollout(ml, S, U, dtype, device, n_rows=1, particle_pred=False):
    """Every (decision_step, weight) `_blend_weight` computes while walking ONE deterministic
    rollout from S[0] under the recorded inputs U.

    Needed because under --onEachRollout the weight is no longer a pure function of the decision
    index -- it depends on the biomass the rollout has actually accumulated -- so it cannot be
    sampled by calling _blend_weight on a bare time grid. Two consequences that make an external
    call outright WRONG there, not merely uninformative: _blend_weight requires current_state
    (it raises without one), and it MUTATES ml._bm_max, so calling it outside get_next_state
    would double-advance the running max and corrupt the very rollout being measured.

    Recording from inside the real get_next_state loop sidesteps both, and gives the identical
    answer under the time sigmoid, where the weight ignores the state entirely."""
    calls = []
    orig = ml._blend_weight

    def _rec(t_step, current_state=None):
        w = orig(t_step, current_state)
        # clone: the tensor returned under --onEachRollout is ml._bm_max-derived and would
        # otherwise alias state that later steps overwrite.
        calls.append((t_step, w.detach().clone() if torch.is_tensor(w) else w))
        return w

    x = torch.tensor(np.tile(np.asarray(S)[0], (n_rows, 1)), dtype=dtype, device=device)
    ml._blend_weight = _rec
    try:
        ml.reset_step_counter(0)
        with torch.no_grad():
            for t in range(1, np.asarray(S).shape[0]):
                u = torch.tensor(np.asarray(U)[t - 1:t], dtype=dtype, device=device).expand(n_rows, -1)
                x, _, _ = ml.get_next_state(current_state=x, current_input=u,
                                            particle_pred=particle_pred)
    finally:
        ml._blend_weight = orig
    return calls


def check_blend_weight_sanity(gp_agent, gp_idx, ho_idx, has_ho, out_dir,
                              pivot_hours, blend_half_width_hours, show=False,
                              pivot_mode="time", on_each_rollout=False, run_for_split=None):
    """C.0 -- blend-weight sanity check for DualPhaseModelLearning's sigmoid phase blend.

    _blend_weight(t_step) is a pure function of (t_step, pivot_hours, blend_half_width_hours,
    T_SAMPLING) -- it does not depend on trained GP state at all -- so this only needs
    gp_agent.model_learning and the actual per-batch trajectory LENGTH (in decision steps) of
    a representative rollout, taken from the real in-sample (and held-out, if available)
    batches already loaded onto gp_agent by reconstruct_gp_agent.

    Reports two distinct weight series per batch, because get_next_state itself uses both:
      - raw:  w2(t) = model_learning._blend_weight(t), w1(t) = 1 - w2(t) -- the sigmoid itself.
      - eff:  what get_next_state ACTUALLY applies once the _BLEND_SKIP_EPS threshold snaps it
              to exactly 0/1 outside +-blend_half_width_hours of the pivot (see its w<=EPS /
              w>=1-EPS branches) -- this is when a phase becomes the SOLE predictor, not just
              dominant.

    NOTE on scope: training itself is unaffected by any of this -- add_data still hard-splits
    each trajectory at pivot_step, so phase1/phase2's own GPs are fit purely on their own
    side's data regardless of how wide the rollout-time blend is (see add_data's docstring).
    What a wide blend window actually risks is a rollout-time effect: during planning/particle
    rollouts, imagined next-states over that window are a genuine MIX of both phases' one-step
    predictions (see get_next_state's mixture-variance formula), so an off-batch phase can
    still leak error into the trajectory the policy is optimised against, even though neither
    phase's own fit was ever "blurred" by the other phase's training data.

    Verifies, and prints a pass/fail line for each:
      (i)   phase1_weight + phase2_weight == 1 at every step (exact by construction; a
            mismatch here would mean the blend formula itself has drifted, not a tuning issue).
      (ii)  weight_phase1 -> 1 early, weight_phase2 -> 1 late (monotonicity of the sigmoid).
      (iii) a contiguous EARLY window exists with weight_phase1 >= 0.95, and a contiguous LATE
            window exists with weight_phase2 >= 0.95 -- if either is empty, that phase never
            reaches near-sole-predictor status anywhere in the batch.
    Also prints min/max of each weight across the batch, and saves a per-step CSV + a plot."""
    cols = [(gp_idx, "IN-SAMPLE")]
    if has_ho:
        cols.append((ho_idx, "HELD-OUT"))

    fig, ax = plt.subplots(1, len(cols), figsize=(7.5 * len(cols), 4.0), squeeze=False)
    rows = []
    summary_lines = []
    for c, (bidx, tag) in enumerate(cols):
        T = np.asarray(gp_agent.state_samples_history[bidx]).shape[0]
        t_ = decision_time_grid(T)
        if on_each_rollout:
            # The weight depends on accumulated biomass, not on t, so it has to be READ OFF a
            # real rollout rather than sampled on a time grid (see _weights_along_rollout).
            # Deterministic single-row walk, matching C.0d's, so the two agree by construction.
            _rec = _weights_along_rollout(
                gp_agent.model_learning,
                gp_agent.state_samples_history[bidx], gp_agent.input_samples_history[bidx],
                gp_agent.dtype, gp_agent.device)
            _by_step = {t: float(np.reshape(w.cpu().numpy() if torch.is_tensor(w) else w, -1)[0])
                        for t, w in _rec}
            # The walk makes T-1 get_next_state calls, so it records decision steps 0..T-2; the
            # final step T-1 is a state the rollout arrives at but never predicts FROM, so no
            # weight is computed for it. Forward-fill rather than defaulting it to 0 -- a zero
            # there is a spurious drop at the end of an otherwise monotone series, which trips
            # both the monotonicity check (ii) and the phase-2 pure-window search (iii).
            _last, raw_w2 = 0.0, []
            for _t in range(T):
                _last = _by_step.get(_t, _last)
                raw_w2.append(_last)
            raw_w2 = np.array(raw_w2)
        else:
            raw_w2 = np.array([gp_agent.model_learning._blend_weight(t) for t in range(T)])
        raw_w1 = 1.0 - raw_w2
        eff_w2 = np.where(raw_w2 <= _BLEND_SKIP_EPS, 0.0,
                          np.where(raw_w2 >= 1.0 - _BLEND_SKIP_EPS, 1.0, raw_w2))
        eff_w1 = 1.0 - eff_w2

        for k in range(T):
            rows.append({"batch_tag": tag, "batch_idx": bidx, "t_step": k, "t_hours": t_[k],
                        "w1_phase1_raw": raw_w1[k], "w2_phase2_raw": raw_w2[k],
                        "w1_phase1_eff": eff_w1[k], "w2_phase2_eff": eff_w2[k]})

        # (i) sums to 1 -- exact by construction, checked anyway (formula-drift guard)
        max_sum_err = float(np.max(np.abs((raw_w1 + raw_w2) - 1.0)))
        chk_sum = max_sum_err < 1e-12

        # (ii) monotonic: phase1 weight non-increasing, phase2 weight non-decreasing
        chk_mono = bool(np.all(np.diff(raw_w1) <= 1e-12) and np.all(np.diff(raw_w2) >= -1e-12))

        # (iii) contiguous windows at >=0.95 (using the EFFECTIVE weight -- what the model
        # actually runs on -- since that's what "sole predictor" means operationally)
        early_mask = eff_w1 >= 0.95
        late_mask = eff_w2 >= 0.95
        has_early = bool(early_mask[0]) and bool(early_mask.any())
        has_late = bool(late_mask[-1]) and bool(late_mask.any())
        early_hi = float(t_[early_mask][-1]) if early_mask.any() else float("nan")
        late_lo = float(t_[late_mask][0]) if late_mask.any() else float("nan")

        line = (
            f"[blend sanity | {tag} batch {bidx}, T={T} steps, {t_[0]:.1f}-{t_[-1]:.1f}h]\n"
            f"  (i)   sum-to-1:      max|w1+w2-1| = {max_sum_err:.2e}  -> {'PASS' if chk_sum else 'FAIL'}\n"
            f"  (ii)  monotonicity:  phase1 non-increasing & phase2 non-decreasing -> "
            f"{'PASS' if chk_mono else 'FAIL'}\n"
            f"  (iii) pure-phase windows (effective weight >= 0.95):\n"
            f"        phase1: {'0.0-' + format(early_hi, '.1f') + 'h' if has_early else 'NONE'}"
            f" (w1 min={raw_w1.min():.3f}, max={raw_w1.max():.3f})\n"
            f"        phase2: {format(late_lo, '.1f') + '-' + format(t_[-1], '.1f') + 'h' if has_late else 'NONE'}"
            f" (w2 min={raw_w2.min():.3f}, max={raw_w2.max():.3f})\n"
            f"        -> {'PASS' if (has_early and has_late) else 'FAIL'}"
        )
        print(line)
        summary_lines.append(line)

        a = ax[0, c]
        a.plot(t_, raw_w1, "-", color="C0", lw=1.8, label="phase1 weight (raw sigmoid)")
        a.plot(t_, raw_w2, "-", color="C1", lw=1.8, label="phase2 weight (raw sigmoid)")
        a.plot(t_, eff_w1, "--", color="C0", lw=1.2, alpha=.7, label="phase1 weight (effective)")
        a.plot(t_, eff_w2, "--", color="C1", lw=1.2, alpha=.7, label="phase2 weight (effective)")
        a.axhline(0.95, color="grey", ls=":", lw=1)
        _mark_pivot(a, pivot_hours, blend_half_width_hours, pivot_mode=pivot_mode)
        a.set_ylim(-0.05, 1.05); a.set_xlabel("time (h)"); a.set_ylabel("blend weight")
        a.set_title(f"{tag} (batch {bidx})", fontsize=9); a.grid(alpha=.3)
        if c == 0:
            a.legend(fontsize=7, loc="center left")
    # Under pivot_mode="biomass" this check is UNCHANGED by design -- stage 1 moves only the
    # training split, not the rollout blend -- so C.0 looks identical to a time-mode run and has
    # been misread as "the biomass pivot was ignored". Say so on the figure and in the .txt.
    _blend_lbl = ("blend centre" if pivot_mode == "biomass" else "pivot")
    fig.suptitle(f"C.0 — blend-weight sanity — {_blend_lbl}={pivot_hours:g}h "
                f"+-{blend_half_width_hours:g}h — model@trial {gp_idx}")
    fig.tight_layout()
    _finish(fig, out_dir, "C0_blend_weight_sanity.png", show)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "C0_blend_weight_sanity.csv", index=False)
    # Where this run's TRAINING data was actually split, printed next to the blend window above.
    # Without it the two live in different files and the stage-1 mismatch reads as a
    # contradiction: the blend window is identical across seeds (it is a pure function of
    # pivot_hours/blend_half_width_hours) while the splits genuinely differ per seed and per
    # trajectory. Seeing both on one page is the whole diagnosis.
    _sl = _load_split_log(run_for_split) if run_for_split is not None else None
    if _sl is not None and len(_sl):
        _h = _sl["hours"].to_numpy(dtype=float)
        _gap = pivot_hours - float(np.median(_h))
        summary_lines.append(
            f"training split (this run, {len(_h)} trajectories): median {np.median(_h):.1f}h, "
            f"range [{_h.min():.1f}, {_h.max():.1f}]h\n"
            + (f"  blend centre is {pivot_hours:g}h -- {_gap:+.1f}h vs the median split; phase 1 "
               f"holds majority weight past the end of its own training data"
               if not on_each_rollout else
               f"  blend now tracks each rollout's own biomass, so it follows these splits "
               f"rather than the {pivot_hours:g}h clock"))

    # Stage-1 only. Under --onEachRollout the blend is NOT the time sigmoid, so this note would
    # be actively wrong there -- the windows above are then read off a real rollout and DO move.
    if pivot_mode == "biomass" and not on_each_rollout:
        summary_lines.append(
            "NOTE (pivot_mode=biomass): the numbers above describe the ROLLOUT BLEND ONLY, which\n"
            "is the time sigmoid centred on pivot_hours in BOTH modes -- so this check is expected\n"
            "to look identical to a pivot_mode=time run. It is NOT evidence the biomass split was\n"
            "ignored. The training split is per-batch and reported in C0a_training_split_sanity\n"
            "(look for overlapping phase1/phase2 TIME ranges) and C0f_split_distribution.")
    (Path(out_dir) / "C0_blend_weight_sanity.txt").write_text("\n\n".join(summary_lines) + "\n")
    return fig


def check_training_split_sanity(gp_agent, gp_idx, out_dir, pivot_hours, show=False,
                                pivot_mode="time", pivot_bm=None):
    """C.0a -- training-DATA assignment sanity check for DualPhaseModelLearning.

    Separate concern from check_blend_weight_sanity (C.0), which only checks the ROLLOUT-time
    prediction blend. This checks the actual TRAINING sets: does phase1's GP only ever see
    early-batch transitions and phase2's only late-batch ones, with no overlap/leak?

    add_data hard-splits at pivot_step (model_learning_dual_phase.py:103-111): phase1 gets
    state_samples[:pivot_step+1], phase2 gets state_samples[pivot_step:]. The base
    Model_learning.data_to_gp_IO (MC-PILCO/model_learning/Model_learning.py:502-506) then
    turns an N-row state slice into (N-1) GP input rows via states[:-1] (dropping the last
    row, which only ever serves as a TARGET, never an input) -- so the two phases' input rows
    should be cleanly adjacent, not overlapping: phase1's last input at decision index
    (pivot_step - 1), phase2's first input at decision index pivot_step. No shared input row,
    unlike a naive split might suggest.

    Reads the ACTUAL accumulated training inputs already loaded onto
    gp_agent.model_learning.phase{1,2} by reconstruct_gp_agent for trial gp_idx (ground truth
    of what those GPs were actually fit on -- not a replay), decodes each row's `time` channel
    (STATE_NAMES.index("time") column of the [state..., action] input vector) back to physical
    batch-hours, and reports count + [min, max] time range per phase.

    The leak check is DISJOINTNESS (phase1's latest input time < phase2's earliest), not a
    comparison against the raw `pivot_hours` float: PIVOT_STEP = round(pivot_hours /
    T_SAMPLING) (pensim_wrapper.py) snaps the split to the nearest decision step, so the
    actual boundary in physical time is generically a bit off nominal pivot_hours (e.g.
    pivot_hours=95.5h with T_SAMPLING=5h rounds to step 19 -> phase1 ends at 90.2h, phase2
    starts at 95.2h -- a clean one-T_SAMPLING-step gap, not a leak, even though 95.2 < 95.5).
    Comparing against pivot_hours directly would misfire on every run whose pivot isn't an
    exact multiple of T_SAMPLING.

    pivot_mode="biomass": TIME-disjointness is the WRONG invariant and this check switches
    coordinates rather than reporting a spurious FAIL. Each batch is split at its own X*Wt
    crossing, so a batch splitting at 40h and another at 95h put phase1 rows as late as 95h
    alongside phase2 rows as early as 40h -- the time ranges genuinely overlap, and that
    overlap IS the feature. The invariant that must still hold is in BIOMASS space: every
    phase1 input row was taken before its own trajectory's running max reached pivot_bm, so
    (running max being monotone and >= the raw value) every phase1 row must have raw
    BM < pivot_bm. Phase2 rows carry no matching bound -- the raw signal peaks ~134h and
    declines, so a phase2 row legitimately dips back below pivot_bm -- hence phase2 is
    reported for information only, not asserted on."""
    time_col = STATE_NAMES.index("time")
    lo_t, hi_t = STATE_RANGES["time"]
    rows = []
    summary_lines = []
    from mcpilco.model_learning_dual_phase import _bm_from_states, BM_PIVOT_DEFAULT
    pivot_bm = BM_PIVOT_DEFAULT if pivot_bm is None else float(pivot_bm)
    for name, sub in [("phase1", gp_agent.model_learning.phase1),
                      ("phase2", gp_agent.model_learning.phase2)]:
        gi = sub.gp_inputs.detach().cpu().numpy() if torch.is_tensor(sub.gp_inputs) else np.asarray(sub.gp_inputs)
        t_hours = decode_state_value("time", _denorm(gi[:, time_col], lo_t, hi_t))
        n = int(gi.shape[0])
        t_min, t_max = float(t_hours.min()), float(t_hours.max())
        # gp_inputs rows are [state..., action], so the state columns _bm_from_states needs
        # (X, Wt) are present and correctly positioned -- slice them off the action column.
        bm = _bm_from_states(gi[:, :len(STATE_NAMES)])
        rows.append({"phase": name, "n_points": n, "time_min_h": t_min, "time_max_h": t_max,
                     "bm_min": float(bm.min()), "bm_max": float(bm.max()),
                     "frac_bm_below_pivot": float((bm < pivot_bm).mean())})

    p1, p2 = rows[0], rows[1]
    tol = 1e-6
    gap_h = p2["time_min_h"] - p1["time_max_h"]
    nominal_step = round(pivot_hours / T_SAMPLING) * T_SAMPLING

    if pivot_mode == "biomass":
        # See docstring: assert in biomass space, report time overlap as expected, not a leak.
        disjoint = p1["frac_bm_below_pivot"] > 1.0 - 1e-9
        line = (
            f"[training split sanity | model@trial {gp_idx}, pivot_mode=biomass, "
            f"pivot_bm={pivot_bm:g}]\n"
            f"  phase1: n={p1['n_points']} points, time [{p1['time_min_h']:.2f}, "
            f"{p1['time_max_h']:.2f}]h, BM [{p1['bm_min']:.0f}, {p1['bm_max']:.0f}]\n"
            f"  phase2: n={p2['n_points']} points, time [{p2['time_min_h']:.2f}, "
            f"{p2['time_max_h']:.2f}]h, BM [{p2['bm_min']:.0f}, {p2['bm_max']:.0f}]\n"
            f"  time ranges overlap by {max(0.0, -gap_h):.2f}h -- EXPECTED under a biomass "
            f"split (each batch is cut at its own crossing, so a late-crossing batch "
            f"contributes phase1 rows later than an early-crossing batch's phase2 rows)\n"
            f"  phase1 rows with BM < pivot_bm: {100*p1['frac_bm_below_pivot']:.2f}% "
            f"(must be 100% -- every phase1 row predates its own trajectory's crossing)\n"
            f"  phase2 rows with BM < pivot_bm: {100*p2['frac_bm_below_pivot']:.2f}% "
            f"(informational: the raw signal peaks ~134h then declines, so dips back below "
            f"are expected and are NOT a leak)\n"
            f"  -> {'PASS (biomass-space split is clean)' if disjoint else 'FAIL (phase1 contains rows at/above pivot_bm)'}"
        )
        print(line)
        summary_lines.append(line)
        df = pd.DataFrame(rows)
        df.to_csv(Path(out_dir) / "C0a_training_split_sanity.csv", index=False)
        (Path(out_dir) / "C0a_training_split_sanity.txt").write_text("\n".join(summary_lines) + "\n")
        return df

    disjoint = p1["time_max_h"] < p2["time_min_h"] - tol

    line = (
        f"[training split sanity | model@trial {gp_idx}, pivot_hours={pivot_hours:g}h "
        f"(rounds to step boundary ~{nominal_step:g}h)]\n"
        f"  phase1: n={p1['n_points']} training points, time range "
        f"[{p1['time_min_h']:.2f}, {p1['time_max_h']:.2f}]h\n"
        f"  phase2: n={p2['n_points']} training points, time range "
        f"[{p2['time_min_h']:.2f}, {p2['time_max_h']:.2f}]h\n"
        f"  gap between phase1's last input and phase2's first input: {gap_h:.2f}h "
        f"(expect ~{T_SAMPLING:g}h, one decision step -- the boundary step itself is a "
        f"TARGET-only row for phase1 and phase2's first INPUT row, never double-counted)\n"
        f"  -> {'PASS (disjoint, no overlap)' if disjoint else 'FAIL (data-assignment leak: ranges overlap/invert)'}"
    )
    print(line)
    summary_lines.append(line)

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "C0a_training_split_sanity.csv", index=False)
    (Path(out_dir) / "C0a_training_split_sanity.txt").write_text("\n".join(summary_lines) + "\n")
    return df


def check_split_distribution(run, out_dir, show=False):
    """C.0f -- WHERE the per-batch training split actually landed (pivot_mode="biomass" only).

    C.0a proves the split is CLEAN (no biomass-space leak); this reports its DISTRIBUTION, which
    is what tells you whether the biomass pivot did anything interesting. Under pivot_mode="time"
    the split is the constant pivot_step in every batch, so there is nothing to plot -- returns
    None and writes no files.

    Reads split_log.pkl (written by the 03_mcpilco_dual_phase_baseline.py driver from
    DualPhaseModelLearning._split_log). Absent for runs trained before that dump existed, in
    which case this reports that and returns None rather than failing the whole eval.

    The number to watch is `n_fallback`: trajectories whose biomass never reached pivot_bm, which
    silently fall back to the fixed pivot_step (see _biomass_pivot_step). A handful is fine -- a
    stunted batch genuinely never enters the production regime -- but a large count means
    pivot_bm is set too high and most batches are being split on the clock after all, which would
    make a biomass-vs-time A/B compare two nearly identical things."""
    if run.pivot_mode != "biomass":
        return None
    df = _load_split_log(run)
    if df is None:
        print(f"[split distribution] no split_log.pkl in {run.dir.name} "
              f"(trained before the dump existed) -- skipping C.0f")
        return None
    hrs = df["hours"].to_numpy(dtype=float)
    n_fb = int((~df["crossed"]).sum())
    cv = float(hrs.std() / abs(hrs.mean())) if hrs.mean() else float("nan")

    line = (
        f"[split distribution | {run.dir.name}, pivot_mode=biomass, pivot_bm={run.pivot_bm:g}]\n"
        f"  {len(df)} trajectories split at: median {np.median(hrs):.1f}h, "
        f"range [{hrs.min():.1f}, {hrs.max():.1f}]h, CV {cv:.3f}\n"
        f"  (a pivot_mode=time run would show every trajectory at the same "
        f"{run.pivot_hours:g}h, CV 0)\n"
        f"  never crossed pivot_bm, fell back to the fixed pivot_step: {n_fb}/{len(df)}"
        f"{'  <-- pivot_bm may be set too high' if n_fb > 0.25 * len(df) else ''}"
    )
    print(line)
    df.to_csv(Path(out_dir) / "C0f_split_distribution.csv", index=False)
    (Path(out_dir) / "C0f_split_distribution.txt").write_text(line + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(hrs, bins=min(20, max(5, len(hrs) // 2)), color="purple", alpha=.65)
    ax[0].axvline(run.pivot_hours, color="k", ls="--", lw=1.4,
                  label=f"fixed pivot would be {run.pivot_hours:g}h")
    ax[0].set_xlabel("training-split time (h)"); ax[0].set_ylabel("trajectories")
    ax[0].set_title("Where each batch was actually split"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].plot(np.arange(len(hrs)), hrs, marker="o", ms=3, lw=1, color="purple")
    ax[1].axhline(run.pivot_hours, color="k", ls="--", lw=1.4)
    ax[1].set_xlabel("trajectory (order added to the GP training set)")
    ax[1].set_ylabel("split time (h)")
    ax[1].set_title("Split time over training"); ax[1].grid(alpha=.3)
    fig.suptitle(f"C.0f — biomass training-split distribution — {run.dir.name} "
                 f"(pivot_bm={run.pivot_bm:g})")
    fig.tight_layout()
    _finish(fig, out_dir, "C0f_split_distribution.png", show)
    return df


def check_rollout_pivot_distribution(gp_agent, gp_idx, run, out_dir, n_particles=None, show=False):
    """C.0g -- WHERE EACH IMAGINED ROLLOUT crosses the pivot (--onEachRollout only).

    Distinct from both neighbours, and the gap between them is the point of this check:
      * C.0f reports the TRAINING split -- one number per real logged trajectory, from add_data.
      * C.0  reports the ROLLOUT blend along a single DETERMINISTIC walk -- one curve.
      * this reports the rollout blend across N PARTICLES of one imagined rollout, i.e. the
        spread the policy optimiser actually sees.

    That spread is the quantity --onEachRollout exists to create: under the time sigmoid every
    particle crosses at the same hour by construction (CV 0), so a run whose particles all still
    cross together has gained nothing from the flag and pivot_bm is probably mis-placed for the
    policy's operating point.

    Returns None and writes nothing under pivot_mode="time" or without --onEachRollout, where
    the answer is degenerate by construction."""
    if not run.on_each_rollout:
        return None
    if n_particles is None:
        # The run's OWN particle count, not a constant that happens to be nearby: this measures
        # the spread the policy optimiser actually saw, so sampling it with a different number of
        # particles than training used would under- (or over-) resolve exactly the quantity being
        # reported. Same read-back-from-cfg discipline the drivers use for note.txt.
        n_particles = int(run.cfg.get("reinforce_par", {})
                             .get("policy_optimization_dict", {})
                             .get("num_particles", 200))
    ml = gp_agent.model_learning
    S = np.asarray(gp_agent.state_samples_history[gp_idx])
    U = np.asarray(gp_agent.input_samples_history[gp_idx])
    T = S.shape[0]

    # Particle rollout, recording the FULL per-particle weight vector at every step (the
    # deterministic single-row walk C.0/C.0d use would collapse exactly the spread of interest).
    calls = []
    orig = ml._blend_weight

    def _rec(t_step, current_state=None):
        w = orig(t_step, current_state)
        calls.append((t_step, w.detach().clone() if torch.is_tensor(w) else w))
        return w

    x = torch.tensor(np.tile(S[0], (n_particles, 1)), dtype=gp_agent.dtype, device=gp_agent.device)
    ml._blend_weight = _rec
    try:
        ml.reset_step_counter(0)
        torch.manual_seed(0)
        with torch.no_grad():
            for t in range(1, T):
                u = torch.tensor(U[t - 1:t], dtype=gp_agent.dtype,
                                 device=gp_agent.device).expand(n_particles, -1)
                x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=True)
    finally:
        ml._blend_weight = orig

    steps = np.array([t for t, _ in calls])
    W = np.stack([np.broadcast_to(np.asarray(w.cpu()) if torch.is_tensor(w) else np.asarray(w),
                                  (n_particles,)) for _, w in calls])          # (n_steps, n_part)
    hours = steps * T_SAMPLING
    # first step at which each particle's weight reaches 0.5 -- the weight is monotone in the
    # running max, so the first crossing is well defined and needs no smoothing.
    crossed = W >= 0.5
    has = crossed.any(axis=0)
    idx = np.where(has, crossed.argmax(axis=0), -1)
    t_cross = np.where(has, hours[np.clip(idx, 0, len(hours) - 1)], np.nan)
    ok = np.isfinite(t_cross)
    cv = float(np.nanstd(t_cross) / abs(np.nanmean(t_cross))) if ok.any() else float("nan")

    line = (
        f"[rollout pivot distribution | model@trial {gp_idx}, batch {gp_idx}, "
        f"{n_particles} particles, pivot_bm={run.pivot_bm:g}]\n"
        f"  particles crossing w=0.5: {int(ok.sum())}/{n_particles}\n"
        f"  crossing time: median {np.nanmedian(t_cross):.1f} h, "
        f"range [{np.nanmin(t_cross):.1f}, {np.nanmax(t_cross):.1f}] h, CV {cv:.3f}\n"
        f"  (a time-sigmoid run would put EVERY particle at {run.pivot_hours:g} h, CV 0)\n"
        f"  never crossed (stay on phase 1 all batch): {int((~ok).sum())}"
        f"{'   <-- phase 1 has no training data that late; pivot_bm likely too high' if (~ok).sum() else ''}"
    )
    print(line)
    pd.DataFrame({"particle": np.arange(n_particles), "t_cross_h": t_cross}).to_csv(
        Path(out_dir) / "C0g_rollout_pivot_distribution.csv", index=False)
    (Path(out_dir) / "C0g_rollout_pivot_distribution.txt").write_text(line + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax[0].hist(t_cross[ok], bins=min(25, max(5, int(ok.sum()) // 8)), color="purple", alpha=.7)
    ax[0].axvline(run.pivot_hours, color="k", ls="--", lw=1.4,
                  label=f"time sigmoid would be {run.pivot_hours:g} h")
    ax[0].set_xlabel("w=0.5 crossing time (h)"); ax[0].set_ylabel("particles")
    ax[0].set_title("Per-particle pivot, one imagined rollout"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    lo, med, hi = np.nanpercentile(W, [5, 50, 95], axis=1)
    ax[1].fill_between(hours, lo, hi, color="purple", alpha=.25, label="5-95th pct")
    ax[1].plot(hours, med, color="purple", lw=2, label="median")
    ax[1].axhline(0.5, color="grey", ls=":", lw=1)
    ax[1].set_xlabel("batch time (h)"); ax[1].set_ylabel("phase-2 weight")
    ax[1].set_title("Weight spread across particles"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle(f"C.0g — per-rollout pivot spread — {run.dir.name}")
    fig.tight_layout()
    _finish(fig, out_dir, "C0g_rollout_pivot_distribution.png", show)
    return t_cross


def check_gp_independence_sanity(gp_agent, gp_idx, out_dir, cfg, show=False):
    """C.0b -- GP-instance independence sanity check for DualPhaseModelLearning.

    Catches aliasing: phase1/phase2 are meant to be two fully independent
    Model_learning_RBF_det_time instances (DualPhaseModelLearning.__init__:
    self.phase1 = phase_model_cls(**phase1_par); self.phase2 = phase_model_cls(**phase2_par)
    -- two separate constructor calls), each holding its own per-channel RBF GP with its own
    torch.nn.Parameter tensors (Stationary_GP.__init__/RBF.__init__ in
    MC-PILCO/gpr_lib/GP_prior/Stationary_GP.py -- log_lengthscales_par, log_lambda_par,
    sigma_n_log). Nothing in that path shares a tensor between phases by construction, but
    this verifies it at runtime rather than by code-reading alone:

      (a) object identity      -- phase1 is not phase2; no gp_list[k] object shared.
      (b) parameter storage    -- no phase1/phase2 parameter pair shares underlying memory
                                   (data_ptr equality) -- catches a `.view()`/shared-buffer
                                   bug that (a) alone would miss (still two Python objects,
                                   but backed by the same storage).
      (c) trained values differ -- prints lengthscales/lambda/sigma_n for both phases per
                                   channel; flags any pair identical to machine precision
                                   (near-impossible after independent training on different
                                   data unless something is aliased or one side never trained).
      (d) moved from init      -- flags any phase/channel whose trained value is still
                                   (near-)exactly its init value -- a phase stuck at init
                                   likely never received a real gradient update.
      (e) gradients flow       -- runs ONE forward+backward per phase/channel (its own
                                   forward(), the same Marginal_log_likelihood criterion real
                                   training uses) against that phase's OWN loaded training
                                   data, WITHOUT calling optimizer.step() (so this does not
                                   mutate the loaded checkpoint's parameter values -- .grad is
                                   explicitly cleared again afterward), and confirms every
                                   trainable parameter's .grad is populated and non-zero for
                                   BOTH phases independently."""
    ml = gp_agent.model_learning
    p1, p2 = ml.phase1, ml.phase2
    n_gp = p1.num_gp
    lines = []

    # (a) object identity
    ok_obj = p1 is not p2
    shared_gp_objs = [k for k in range(n_gp) if p1.gp_list[k] is p2.gp_list[k]]
    lines.append(
        f"(a) object identity: phase1 is not phase2 -> {'PASS' if ok_obj else 'FAIL (SAME OBJECT)'}; "
        f"shared gp_list[k] objects: {shared_gp_objs or 'none'} -> "
        f"{'PASS' if not shared_gp_objs else 'FAIL'}")

    # (b) parameter storage identity
    shared_storage = []
    for k in range(n_gp):
        params1 = dict(p1.gp_list[k].named_parameters())
        params2 = dict(p2.gp_list[k].named_parameters())
        for name, t1 in params1.items():
            t2 = params2.get(name)
            if t2 is not None and t1.data_ptr() == t2.data_ptr():
                shared_storage.append((k, name))
    lines.append(f"(b) parameter storage: shared data_ptr pairs: {shared_storage or 'none'} -> "
                f"{'PASS' if not shared_storage else 'FAIL (ALIASED STORAGE)'}")

    # (c)/(d) trained hyperparameters: differ between phases, and moved from init
    init_dict_1 = cfg["mc_pilco_init"]["model_learning_par"]["phase1_par"]["init_dict_list"]
    init_dict_2 = cfg["mc_pilco_init"]["model_learning_par"]["phase2_par"]["init_dict_list"]
    rows = []
    identical_pairs = []
    stuck_at_init = []
    for k in range(n_gp):
        name = STATE_NAMES[k] if k < len(STATE_NAMES) else f"gp{k}"
        for tag, sub, init_dict in [("phase1", p1, init_dict_1[k]), ("phase2", p2, init_dict_2[k])]:
            gp = sub.gp_list[k]
            ls = np.exp(gp.log_lengthscales_par.detach().cpu().numpy())
            lam = float(np.exp(gp.log_lambda_par.detach().cpu().numpy().ravel()[0]))
            sn = float(np.exp(gp.sigma_n_log.detach().cpu().numpy().ravel()[0]))
            ls_init = np.asarray(init_dict["lengthscales_init"], dtype=float)
            lam_init = float(np.asarray(init_dict["lambda_init"]).ravel()[0])
            sn_init = float(np.asarray(init_dict["sigma_n_init"]).ravel()[0])
            moved = not (np.allclose(ls, ls_init, atol=1e-6, rtol=0)
                        and np.isclose(lam, lam_init, atol=1e-6, rtol=0)
                        and np.isclose(sn, sn_init, atol=1e-6, rtol=0))
            if not moved:
                stuck_at_init.append((name, tag))
            rows.append({"channel": name, "phase": tag, "lengthscales": ls.tolist(),
                        "lambda": lam, "sigma_n": sn, "moved_from_init": moved})
        ls1 = np.exp(p1.gp_list[k].log_lengthscales_par.detach().cpu().numpy())
        ls2 = np.exp(p2.gp_list[k].log_lengthscales_par.detach().cpu().numpy())
        lam1 = float(np.exp(p1.gp_list[k].log_lambda_par.detach().cpu().numpy().ravel()[0]))
        lam2 = float(np.exp(p2.gp_list[k].log_lambda_par.detach().cpu().numpy().ravel()[0]))
        sn1 = float(np.exp(p1.gp_list[k].sigma_n_log.detach().cpu().numpy().ravel()[0]))
        sn2 = float(np.exp(p2.gp_list[k].sigma_n_log.detach().cpu().numpy().ravel()[0]))
        if np.array_equal(ls1, ls2) and lam1 == lam2 and sn1 == sn2:
            identical_pairs.append(name)
        print(f"  [{name}] phase1: lengthscales={np.array2string(ls1, precision=4)} "
             f"lambda={lam1:.4g} sigma_n={sn1:.4g}")
        print(f"  [{name}] phase2: lengthscales={np.array2string(ls2, precision=4)} "
             f"lambda={lam2:.4g} sigma_n={sn2:.4g}")
    lines.append(f"(c) identical-to-machine-precision phase1/phase2 pairs: "
                f"{identical_pairs or 'none'} -> {'PASS' if not identical_pairs else 'FAIL (possible alias)'}")
    lines.append(f"(d) stuck at init (never moved) (channel, phase) pairs: "
                f"{stuck_at_init or 'none'} -> {'PASS' if not stuck_at_init else 'FAIL (may not have trained)'}")

    # (e) gradient flow: one forward+backward per phase/channel, no optimizer.step() (does
    # not mutate the checkpoint's parameter VALUES -- .grad is cleared again immediately after
    # inspection). reconstruct_gp_agent leaves every GP in eval mode, which -- per
    # GP_prior.set_eval_mode (MC-PILCO/gpr_lib/GP_prior/GP_prior.py:75-80) -- forces
    # requires_grad=False on EVERY parameter (stashing the prior flags for set_training_mode
    # to restore) as a safety measure for inference-only rollouts. That's not a training bug;
    # it just means this probe must call set_training_mode() first (matching what real
    # training actually ran under) or every parameter would trivially show "no grad" for a
    # reason that has nothing to do with phase independence. set_eval_mode() is restored
    # afterward so this leaves the reconstructed agent exactly as it was found.
    criterion_cls = cfg["reinforce_par"]["model_optimization_opt_list"][0]["criterion"]
    no_grad_flags = []
    for tag, sub in [("phase1", p1), ("phase2", p2)]:
        for k in range(n_gp):
            name = STATE_NAMES[k] if k < len(STATE_NAMES) else f"gp{k}"
            gp = sub.gp_list[k]
            gp.set_training_mode()
            for p in gp.parameters():
                p.grad = None
            X = sub.gp_inputs
            Y = sub.gp_output_list[k] / sub.norm_list[k]
            out = gp(X)
            loss = criterion_cls()(out, Y)
            loss.backward()
            for pname, p in gp.named_parameters():
                if not p.requires_grad:
                    continue
                g = p.grad
                gmax = 0.0 if g is None else float(g.abs().max())
                if g is None or gmax == 0.0:
                    no_grad_flags.append((tag, name, pname))
            for p in gp.parameters():
                p.grad = None  # leave checkpoint's grad state as found (untouched values)
            gp.set_eval_mode()  # restore the mode reconstruct_gp_agent left this GP in
    lines.append(f"(e) zero/missing-gradient (phase, channel, param) triples: "
                f"{no_grad_flags or 'none'} -> {'PASS' if not no_grad_flags else 'FAIL'}")

    summary = f"[GP independence sanity | model@trial {gp_idx}]\n" + "\n".join(lines)
    print(summary)

    pd.DataFrame(rows).to_csv(Path(out_dir) / "C0b_gp_independence_sanity.csv", index=False)
    (Path(out_dir) / "C0b_gp_independence_sanity.txt").write_text(summary + "\n")
    return rows


def check_rollout_gradient_flow(gp_agent, gp_idx, out_dir, N=20, show=False):
    """C.0c -- gradient-flow-through-the-blend sanity check for DualPhaseModelLearning.

    Different failure mode from check_gp_independence_sanity (C.0b), which only tested
    gradients during MODEL fitting (train_gp_likelihood's one-GP marginal-log-likelihood
    loss, no blend involved at all). This checks the thing POLICY optimisation actually runs:
    a multi-step PARTICLE rollout through get_next_state (particle_pred=True, i.e. through
    Normal.rsample() -- the reparameterization trick MC-PILCO relies on for pathwise policy
    gradients: model_learning_det_time.py's get_next_state_from_gp_output uses `.rsample()`,
    not `.sample()`, so sampling itself is differentiable) across MANY sequential steps,
    through DualPhaseModelLearning.get_next_state's blend
    (next_states = (1-w)*next1 + w*next2 -- model_learning_dual_phase.py:161-166) and its
    EPS-skip branches (lines 156-159, which route to exactly one phase outside the blend
    window -- not a gradient bug, just means that specific STEP's gradient can only credit
    the phase actually evaluated there; see the printed per-phase step-coverage below).

    A real reparameterization break (an accidental .detach()/.numpy()/torch.no_grad() on one
    phase's branch, or a hard argmax-style gate instead of the smooth sigmoid) would show up
    as exactly-zero gradient on every parameter of the affected phase, even though that phase
    IS evaluated for a large chunk of the trajectory -- this test would catch it where C.0b
    could not, since C.0b never invokes the blend at all.

    Procedure: starting from batch gp_idx's own recorded initial state (N particles, replicated
    like _particle_rollout), replays the RECORDED action sequence but as a single differentiable
    leaf tensor (action_probe, requires_grad=True) standing in for "whatever produced the
    action" (the real policy network, in actual training) -- this isolates the model/blend
    graph's differentiability from the policy network's own, which is a separate, standard
    nn.Module concern outside this check's scope. Both phases' GPs are temporarily switched to
    set_training_mode() (restoring requires_grad=True on their own hyperparameters, undone by
    reconstruct_gp_agent's set_eval_mode()) so gradient reaching THEIR parameters directly
    localises which phase's forward computation the graph actually passed through -- a stronger
    localisation than just checking action_probe.grad, which could stay non-zero even if only
    one phase secretly contributed. Runs loss.backward() once on a sign-safe scalar
    (next_states**2).sum() (avoids accidental cross-particle/dim sign cancellation to zero) and
    reports non-None/non-zero verdicts for action_probe and every trainable parameter of both
    phases. Restores set_eval_mode() and clears .grad afterward -- does not mutate the loaded
    checkpoint's parameter VALUES."""
    ml = gp_agent.model_learning
    p1, p2 = ml.phase1, ml.phase2
    n_gp = p1.num_gp
    dtype, device = gp_agent.dtype, gp_agent.device

    for sub in (p1, p2):
        for gp in sub.gp_list:
            gp.set_training_mode()

    S = np.asarray(gp_agent.state_samples_history[gp_idx])
    U = np.asarray(gp_agent.input_samples_history[gp_idx])
    T = S.shape[0]
    action_probe = torch.tensor(U, dtype=dtype, device=device, requires_grad=True)

    torch.manual_seed(0)
    x = torch.tensor(np.tile(S[0], (N, 1)), dtype=dtype, device=device)
    ml.reset_step_counter(0)
    phase1_calls, phase2_calls = 0, 0

    # The per-phase call counts are RECORDED from inside get_next_state rather than recomputed
    # by calling _blend_weight externally. Under --onEachRollout an external call is not just
    # uninformative but actively harmful: _blend_weight needs current_state and MUTATES
    # ml._bm_max, so an extra call per step would double-advance the running max and change the
    # very rollout whose gradient flow is being measured. Recording gives the identical counts
    # under the time sigmoid, where the weight is state-independent.
    _orig_bw = ml._blend_weight

    def _counting_blend_weight(t_step, current_state=None):
        nonlocal phase1_calls, phase2_calls
        w = _orig_bw(t_step, current_state)
        w_max = float(w.max()) if torch.is_tensor(w) else w
        w_min = float(w.min()) if torch.is_tensor(w) else w
        # "phase N was evaluated at this step" -- mirrors get_next_state's own skip shortcut,
        # which under --onEachRollout requires unanimity across particles.
        phase1_calls += int(w_max < 1.0 - _BLEND_SKIP_EPS)
        phase2_calls += int(w_min > _BLEND_SKIP_EPS)
        return w

    ml._blend_weight = _counting_blend_weight
    try:
        for t in range(1, T):
            u = action_probe[t - 1:t, :].expand(N, -1)
            x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=True)
    finally:
        ml._blend_weight = _orig_bw
    loss = (x ** 2).sum()
    loss.backward()

    lines = [
        f"[rollout gradient-flow sanity | model@trial {gp_idx}, batch {gp_idx}, T={T} steps]",
        f"  phase1 evaluated on {phase1_calls}/{T - 1} steps, phase2 evaluated on "
        f"{phase2_calls}/{T - 1} steps (both > (T-1) - blend-only steps, since each phase is "
        f"evaluated for its pure region PLUS the shared blend region, not skipped there)",
    ]

    ap_grad = action_probe.grad
    ap_max = 0.0 if ap_grad is None else float(ap_grad.abs().max())
    lines.append(f"  action_probe.grad: {'None' if ap_grad is None else f'max|grad|={ap_max:.3e}'} "
                f"-> {'PASS' if ap_max > 0.0 else 'FAIL (no gradient reaches the actions at all)'}")

    no_grad_flags = []
    expected_no_grad = []
    for tag, sub in [("phase1", p1), ("phase2", p2)]:
        for k in range(n_gp):
            name = STATE_NAMES[k] if k < len(STATE_NAMES) else f"gp{k}"
            gp = sub.gp_list[k]
            for pname, p in gp.named_parameters():
                if not p.requires_grad:
                    continue
                g = p.grad
                gmax = 0.0 if g is None else float(g.abs().max())
                if g is None or gmax == 0.0:
                    # DETERMINISTIC_CHANNELS (the `time` clock) has its GP's output SPLICED
                    # OUT and replaced with an exact deterministic delta every step
                    # (model_learning_det_time.get_next_state_from_gp_output) -- its own GP's
                    # forward pass never influences next_states, so zero gradient there is
                    # correct/expected for BOTH phases, not a reparameterization break.
                    (expected_no_grad if k in DETERMINISTIC_CHANNELS else no_grad_flags).append(
                        (tag, name, pname))
    if expected_no_grad:
        lines.append(f"  zero-gradient on deterministic channel(s) (expected, GP output is "
                    f"discarded/spliced there regardless of phase): {expected_no_grad}")
    lines.append(f"  zero/missing-gradient on NON-deterministic (phase, channel, param) triples: "
                f"{no_grad_flags or 'none'} -> "
                f"{'PASS (gradient reaches both phases)' if not no_grad_flags else 'FAIL (a phase is effectively frozen during rollout)'}")

    # cleanup: leave the reconstructed agent's grad/mode state as we found it
    action_probe.grad = None
    for sub in (p1, p2):
        for gp in sub.gp_list:
            for p in gp.parameters():
                p.grad = None
            gp.set_eval_mode()

    summary = "\n".join(lines)
    print(summary)
    (Path(out_dir) / "C0c_rollout_gradient_flow.txt").write_text(summary + "\n")
    return no_grad_flags


def check_train_eval_blend_consistency(gp_agent, gp_idx, run, out_dir, show=False):
    """C.0d -- prediction-time / training-time blend consistency check.

    Verifies the SAME blend function (same pivot, same half-width, same per-step weight) is
    used identically in (i) the particle rollout POLICY OPTIMISATION drives and (ii) the
    rollout HELD-OUT EVALUATION/PREDICTION diagnostics use -- a mismatch there would silently
    degrade multi-phase specifically (the single-phase model has no such split to drift).

    Two independent lines of evidence, not just one:

    (a)/(b) VALUE check: compares gp_agent.model_learning.pivot_hours/blend_half_width_hours
    (the values actually baked into the reconstructed model doing every computation below)
    against run.pivot_hours/run.blend_half_width_hours (parsed straight from this run's own
    note.txt -- see Run.pivot_hours/blend_half_width_hours, eval_multi_phase_lib.py:156-162).
    Catches a config-reconstruction bug (e.g. a dropped kwarg in _build_cfg_kwargs) that would
    silently evaluate with different blend parameters than the run actually trained with.

    (c)/(d) STRUCTURAL check, empirical not just read from code: PenSimMCPILCOMultiPhase
    overrides BOTH apply_policy (policy optimisation's rollout) AND rollout (the method behind
    get_rollout_prediction_performance, hence every held-out plot in this file) to call
    self.model_learning.reset_step_counter() before delegating to the SAME base MC_PILCO loop,
    which calls self.model_learning.get_next_state(...) -- see pensim_wrapper.py's
    PenSimMCPILCOMultiPhase docstring and its apply_policy/rollout overrides. Structurally
    there is exactly one blend implementation (DualPhaseModelLearning._blend_weight) and no
    parallel reimplementation exists for evaluation. This step verifies that guarantee
    empirically: instruments _blend_weight to record every (decision_step, weight) pair it
    computes, once while replaying batch gp_idx via a hand-rolled policy-optimisation-SHAPED
    loop (reset + sequential get_next_state, matching apply_policy's own loop shape) and once
    via gp_agent.rollout(data_collection_index=gp_idx, ...) itself (the actual method every
    held-out evaluation diagnostic here calls), then checks the two recorded sequences are
    identical step-for-step. particle_pred=False (deterministic mean rollout) in both, so this
    isolates blend-routing consistency from any RNG-driven sampling variation."""
    ml = gp_agent.model_learning
    lines = []

    pivot_match = abs(ml.pivot_hours - run.pivot_hours) < 1e-9
    width_match = abs(ml.blend_half_width_hours - run.blend_half_width_hours) < 1e-9
    lines.append(f"(a) reconstructed model pivot_hours={ml.pivot_hours:g}h vs run.pivot_hours "
                f"(note.txt)={run.pivot_hours:g}h -> {'PASS' if pivot_match else 'FAIL (MISMATCH)'}")
    lines.append(f"(b) reconstructed model blend_half_width_hours={ml.blend_half_width_hours:g}h "
                f"vs run.blend_half_width_hours (note.txt)={run.blend_half_width_hours:g}h -> "
                f"{'PASS' if width_match else 'FAIL (MISMATCH)'}")

    calls = []
    orig_blend_weight = ml._blend_weight

    def _recording_blend_weight(t_step, current_state=None):
        # Must accept current_state: get_next_state passes it under --onEachRollout (and omits
        # it otherwise), so a one-argument wrapper would TypeError there. Forwarding it
        # unconditionally is safe -- _blend_weight ignores it under the time sigmoid.
        w = orig_blend_weight(t_step, current_state)
        # clone, or every recorded entry would alias the same running-max tensor.
        calls.append((t_step, w.detach().clone() if torch.is_tensor(w) else w))
        return w

    S = np.asarray(gp_agent.state_samples_history[gp_idx])
    U = np.asarray(gp_agent.input_samples_history[gp_idx])
    T = S.shape[0]

    ml._blend_weight = _recording_blend_weight
    try:
        # (i) policy-optimisation-SHAPED loop: reset + sequential get_next_state, matching
        # apply_policy's own particle-rollout loop shape (see PenSimMCPILCOMultiPhase docstring).
        x = torch.tensor(S[0:1], dtype=gp_agent.dtype, device=gp_agent.device)
        ml.reset_step_counter(0)
        calls.clear()
        with torch.no_grad():
            for t in range(1, T):
                u = torch.tensor(U[t - 1:t], dtype=gp_agent.dtype, device=gp_agent.device)
                x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=False)
        w_policy_style = list(calls)

        # (ii) the ACTUAL agent.rollout() method behind get_rollout_prediction_performance --
        # i.e. every held-out/prediction diagnostic in this file.
        calls.clear()
        with torch.no_grad():
            gp_agent.rollout(data_collection_index=gp_idx, particle_pred=False)
        w_eval_style = list(calls)
    finally:
        ml._blend_weight = orig_blend_weight

    same_len = len(w_policy_style) == len(w_eval_style)
    # Bit-identity is still the right bar under --onEachRollout: both loops start from the SAME
    # S[0], replay the SAME recorded inputs U, run deterministically (particle_pred=False) and
    # reset the step counter first (PenSimMCPILCOMultiPhase.rollout overrides do so), so the two
    # state trajectories -- and hence the state-dependent weights -- must coincide exactly. Only
    # the COMPARISON needs to be type-aware, because the weight is a tensor there.
    identical = same_len and all(
        ta == tb and _weights_equal(wa, wb)
        for (ta, wa), (tb, wb) in zip(w_policy_style, w_eval_style))
    lines.append(f"(c) weight-sequence length: policy-opt-shaped loop={len(w_policy_style)} calls, "
                f"agent.rollout()={len(w_eval_style)} calls -> {'PASS' if same_len else 'FAIL'}")
    if same_len and not identical:
        mismatches = [(a, b) for a, b in zip(w_policy_style, w_eval_style)
                      if not (a[0] == b[0] and _weights_equal(a[1], b[1]))]
        (ta, wa), (tb, wb) = mismatches[0]
        lines.append(f"(d) per-step (decision_step, weight) pairs identical -> FAIL, "
                    f"{len(mismatches)}/{len(w_policy_style)} mismatches, first: "
                    f"step {ta} w={_fmt_weight(wa)} vs step {tb} w={_fmt_weight(wb)}")
    else:
        lines.append(f"(d) per-step (decision_step, weight) pairs identical, bit-for-bit -> "
                    f"{'PASS' if identical else 'FAIL'}")

    ok = pivot_match and width_match and identical
    summary = (f"[train/eval blend consistency | model@trial {gp_idx}, batch {gp_idx}]\n"
              + "\n".join(lines) + f"\n  -> OVERALL: {'PASS' if ok else 'FAIL'}")
    print(summary)
    (Path(out_dir) / "C0d_train_eval_blend_consistency.txt").write_text(summary + "\n")
    return ok


def check_active_dims_consistency(gp_agent, run, single_phase_get_config_fn, out_dir, show=False):
    """C.0e -- active_dims / state-layout consistency check: phase1 vs phase2 vs the matching
    single-phase baseline.

    Given STATE_NAMES = ["Wt", "X", "P", "Viscosity", "time"], each of the num_gp per-channel
    GPs reads the SAME shared `active_dims` (a fixed subset of the [state..., action] input
    columns -- see config_dual_phase.py's init_dict_RBF, reused for every gp_index), and
    gp_list[k] is BY CONSTRUCTION the GP that predicts STATE_NAMES[k]'s delta (data_to_gp_output
    in MC-PILCO/model_learning/Model_learning.py:495-500 builds targets via
    `states[1:,i]-states[:-1,i]) for i in range(dim_state)` -- index i IS the channel index, no
    relabelling possible). An "off-by-one" bug here would mean some gp_index's active_dims (or
    which physical channel it targets) has silently drifted between phase1, phase2, and/or the
    single-phase config it's meant to match -- e.g. one phase accidentally excluding a different
    channel from its regressors, or the two ablation families (time-dropped vs time-kept)
    disagreeing about which config family a run actually belongs to.

    single_phase_get_config_fn's kwargs are filtered from run.params via inspect.signature (not
    hardcoded) since single-phase and dual-phase get_config accept different kwarg sets (e.g.
    single-phase has no pivot_hours/blend_half_width_hours) -- this only needs the CONFIG
    (init_dict_list is fixed at config-build time, identical across every trial), not a trained
    agent, so no single-phase run needs to have ever actually been trained for this check to run."""
    import inspect

    ml = gp_agent.model_learning
    p1, p2 = ml.phase1, ml.phase2
    n_gp = p1.num_gp

    accepted = set(inspect.signature(single_phase_get_config_fn).parameters)
    sp_kwargs = {k: v for k, v in run.params.items() if k in accepted}
    sp_cfg = single_phase_get_config_fn(**sp_kwargs)
    sp_init_dict_list = sp_cfg["mc_pilco_init"]["model_learning_par"]["init_dict_list"]

    lines = ["[active_dims / state-layout consistency]"]
    time_idx = STATE_NAMES.index("time")
    lines.append(f"STATE_NAMES={STATE_NAMES}, TIME_IDX={time_idx}, "
                f"DETERMINISTIC_CHANNELS keys={list(DETERMINISTIC_CHANNELS.keys())} -> "
                f"{'PASS' if list(DETERMINISTIC_CHANNELS.keys()) == [time_idx] else 'FAIL (TIME_IDX mismatch)'}")
    lines.append(f"num_gp: phase1={p1.num_gp}, phase2={p2.num_gp}, single_phase={len(sp_init_dict_list)}, "
                f"STATE_DIM={STATE_DIM} -> "
                f"{'PASS' if p1.num_gp == p2.num_gp == len(sp_init_dict_list) == STATE_DIM else 'FAIL'}")

    rows = []
    mismatched_channels = []
    for k in range(n_gp):
        name = STATE_NAMES[k] if k < len(STATE_NAMES) else f"gp{k}"
        ad1 = p1.gp_list[k].active_dims.cpu().numpy().tolist()
        ad2 = p2.gp_list[k].active_dims.cpu().numpy().tolist()
        ad_sp = np.asarray(sp_init_dict_list[k]["active_dims"]).tolist()
        row_ok = (ad1 == ad2 == ad_sp)
        rows.append({"channel": name, "gp_index": k, "phase1_active_dims": ad1,
                    "phase2_active_dims": ad2, "single_phase_active_dims": ad_sp,
                    "all_match": row_ok})
        line = (f"  [gp_index={k} / {name}] phase1={ad1}  phase2={ad2}  "
               f"single_phase={ad_sp}  -> {'PASS' if row_ok else 'FAIL'}")
        print(line)
        lines.append(line)
        if not row_ok:
            mismatched_channels.append(name)

    time_present_1 = time_idx in p1.gp_list[0].active_dims.cpu().numpy().tolist()
    time_present_2 = time_idx in p2.gp_list[0].active_dims.cpu().numpy().tolist()
    time_present_sp = time_idx in np.asarray(sp_init_dict_list[0]["active_dims"]).tolist()
    lines.append(f"`time` (idx {time_idx}) present as a GP INPUT regressor: phase1={time_present_1}, "
                f"phase2={time_present_2}, single_phase={time_present_sp} -> "
                f"{'PASS (all agree)' if time_present_1 == time_present_2 == time_present_sp else 'FAIL (ablation family mismatch)'}")

    ok = (not mismatched_channels) and (time_present_1 == time_present_2 == time_present_sp) \
        and (p1.num_gp == p2.num_gp == len(sp_init_dict_list) == STATE_DIM) \
        and (list(DETERMINISTIC_CHANNELS.keys()) == [time_idx])
    lines.append(f"-> OVERALL: {'PASS' if ok else f'FAIL (mismatched channels: {mismatched_channels})'}")

    summary = "\n".join(lines)
    for l in lines:
        if not l.startswith("  [gp_index"):  # already printed inside the loop above
            print(l)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "C0e_active_dims_consistency.csv", index=False)
    (Path(out_dir) / "C0e_active_dims_consistency.txt").write_text(summary + "\n")
    return ok


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
        ml.reset_step_counter(t0, bm_max0=_bm_max0_at(ml, true, t0))
        # dual-phase: route decisions t0, t0+1, ... through whichever phase actually owns them;
        # bm_max0 supplies the biomass this jump skipped over (see _bm_max0_at).
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
                           blend_half_width_hours=None, show=False, pivot_mode="time"):
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

    _mark_pivot(ax, pivot_hours, blend_half_width_hours, pivot_mode=pivot_mode)
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
                        blend_half_width_hours=None, n_part=100, show=False, pivot_mode="time"):
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
            _mark_pivot(a, pivot_hours, blend_half_width_hours, pivot_mode=pivot_mode)
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
            ml.reset_step_counter(t0, bm_max0=_bm_max0_at(ml, tr, t0))
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
            ml.reset_step_counter(t0, bm_max0=_bm_max0_at(ml, tr, t0))
            nxt, _, _ = ml.get_next_state(current_state=tr[t0:t0 + 1, :],
                                          current_input=ip[t0:t0 + 1, :], particle_pred=False)
            errs[t0] = (nxt - tr[t0 + 1:t0 + 2, :]).abs().ravel().detach().cpu().numpy()
    eX = _denorm_delta(errs[:, X_IDX], *STATE_RANGES["X"])
    eP = _denorm_delta(errs[:, P_IDX], *STATE_RANGES["P"])
    return grid[:-1], eX, eP


def plot_local_error(gp_agent, gp_idx, ho_idx, has_ho, out_dir, pivot_hours,
                     blend_half_width_hours=None, show=False, pivot_mode="time"):
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
    _mark_pivot(ax, pivot_hours, blend_half_width_hours, pivot_mode=pivot_mode)
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

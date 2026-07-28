"""Shared analysis library for single-phase MC-PILCO evaluation.

Both evaluations_single_phase.py (CLI) and evaluations.ipynb (notebook) import from here, so
there is exactly one implementation of every plot/table -- the two entry points cannot
silently drift apart. Every plot/table function always saves its output into `out_dir` (PNG
for plots, CSV for tables); pass show=True (the notebook does) to also display inline.

Mirrors mcpilco_mcpilco/evaluations/eval_multi_phase_lib.py's structure, simplified back down
to a flat (non-composite) model_learning -- no phase splitting, no reset_step_counter dance.
C.7 (short-horizon/recipe-anchor diagnostic), which the dual-phase lib drops as structurally
inapplicable, IS ported here since anchors/optim_horizon are real, supported knobs for
single-phase runs (--num_anchor_batches/--optim_horizon on experiments/02_mcpilco_single_phase.py).

This replaces the previous inline evaluations.ipynb content, and along the way fixes two
latent bugs that inline version had: several names it used (torch, STATE_RANGES, X_IDX/P_IDX,
etc.) were never actually imported/defined anywhere in the committed notebook, and it hardcoded
T_SAMPLING = 2.0 in its setup cell despite the real trained decision spacing being 5h
(pensim_wrapper.T_SAMPLING) -- both would have broken/mislabelled a from-scratch run. This
module imports everything it needs explicitly and uses the real T_SAMPLING throughout.
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
from matplotlib.lines import Line2D
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

from mcpilco.config_single_phase import get_config
from mcpilco.pensim_wrapper import (
    PenSimWrapper, PenSimMCPILCO, STATE_NAMES, STATE_DIM, ACTION_DIM,
    STATE_RANGES, decode_state_value, T_SAMPLING, CONTROL_H, K_WARM, VISC_MAX, WARMUP_H,
    PAA_BAND, FS_SCALE, FPAA_MIN, FPAA_MAX, initial_state_norm,
)
from experiments.eval_utils import yield_kg, constraint_diagnostics

X_IDX = STATE_NAMES.index("X")
P_IDX = STATE_NAMES.index("P")
ACTION_IDX = STATE_DIM  # gp_input columns are [states..., action]

RESULTS_ROOT = Path(_ROOT) / "results" / "single_phase"

REF_STYLE = dict(color="red", lw=2.2, ls="--", zorder=6)
MODEL_STYLE = dict(color="C0", lw=2.0, zorder=5)

# get_config kwargs recoverable from note.txt's "== run parameters ==" block (see the
# run_params dict in experiments/02_mcpilco_single_phase.py). "optim_horizon" is the CLI/
# note.txt name but get_config's own kwarg is "optim_horizon_steps" -- renamed here rather
# than in get_config, since the CLI flag name is the more stable public surface.
_GET_CONFIG_KEY_RENAME = {"optim_horizon": "optim_horizon_steps"}
_GET_CONFIG_KEYS = ("seed", "num_trials", "fast", "optim_horizon_steps", "num_anchor_batches",
                    "num_anchors", "anchor_var", "risk_weight", "visc_penalty",
                    "constraint_strength", "harvest_reward", "num_high_feed_probes",
                    "pms_visc_delay")


def _build_cfg_kwargs(params):
    out = {}
    for k, v in params.items():
        k2 = _GET_CONFIG_KEY_RENAME.get(k, k)
        if k2 in _GET_CONFIG_KEYS:
            out[k2] = v
    # pms_visc_delay predates this key existing in note.txt at all: runs written before it was
    # added to run_params have no such line, and MUST NOT fall through to get_config's own
    # default (True) -- absent means "trained before this feature existed", i.e. no delay.
    # get_config's default stays True because THAT default governs fresh/manual get_config()
    # calls (e.g. a brand new training run), a separate concern from reconstructing a past run.
    out.setdefault("pms_visc_delay", False)
    return out


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def resolve_run_dir(run_id_or_path):
    """Accepts a bare run name ("seed3_104") resolved under RESULTS_ROOT, or an existing
    absolute/relative path directly."""
    p = Path(run_id_or_path)
    if p.exists():
        return p
    candidate = RESULTS_ROOT / run_id_or_path
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
    def optim_horizon(self):
        return self.params.get("optim_horizon")

    @property
    def num_anchors(self):
        return self.params.get("num_anchors")

    @property
    def has_anchors(self):
        """C.7 (short-horizon/recipe-anchor diagnostic) is only meaningful when the run
        actually used anchors -- both optim_horizon and num_anchor_batches/num_anchors must
        be set and non-zero."""
        return bool(self.optim_horizon) and bool(self.num_anchors) and bool(self.params.get("num_anchor_batches"))


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


def load_run(run_id_or_path):
    """The one function both entry points call first. Resolves the dir, parses note.txt,
    rebuilds cfg via get_config(**params), loads log.pkl/monitor.pkl, checks state-dim
    compatibility."""
    run_dir = resolve_run_dir(run_id_or_path)
    note_path = run_dir / "note.txt"
    if not note_path.exists():
        raise FileNotFoundError(f"{run_dir}: no note.txt (needed to recover run params)")
    all_params = parse_run_params(note_path)
    cfg = get_config(**_build_cfg_kwargs(all_params))

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

    print(f"run: {run_dir}  |  episodes: {n_ep}  |  trials in log: {n_trials_in_log}")

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
    """Build a PenSimMCPILCO, load the trained policy (last saved trial) from run.log, and
    roll a same-seed recipe reference batch."""
    # run.cfg["wrapper_par"]["pms_visc_delay"] reflects what THIS run actually used (read from
    # note.txt via _build_cfg_kwargs, defaulting to False for pre-existing runs that predate the
    # key) -- NOT get_config's own default, which would silently misjudge older runs.
    eval_wrapper = PenSimWrapper(**run.cfg["wrapper_par"])
    policy_agent = PenSimMCPILCO(pensim_wrapper=eval_wrapper, **run.cfg["mc_pilco_init"])
    folder = str(run.dir).rstrip("/") + "/"
    with contextlib.redirect_stdout(io.StringIO()):
        policy_agent.load_policy_from_log(num_trial=run.n_trials_in_log, folder=folder)
    np_policy = policy_agent.control_policy.get_np_policy()
    _assert_policy_state_dim_ok(np_policy, f"trial {run.n_trials_in_log} policy")

    base_mon = run_arm(eval_wrapper, run.train_seed, policy=None, pid_baseline=True)
    ref = {k: (np.asarray(base_mon["t"]), np.asarray(base_mon[k]))
          for k in ("P", "PAA", "Viscosity", "Fpaa", "Wt", "Fs")}
    ref["yield"] = yield_kg(base_mon)
    ref["final_P"] = float(base_mon["P"][-1])
    ref_lbl = f"recipe (seed {run.train_seed})"
    print(f"loaded trial {run.n_trials_in_log} policy | recipe baseline on seed {run.train_seed}: "
         f"final_P={ref['final_P']:.2f} g/L, yield={ref['yield']:.1f} kg")
    return policy_agent, eval_wrapper, np_policy, ref, ref_lbl


def load_stage_policy(run, trial_k, folder=None):
    """Load trial-k's saved policy onto a FRESH staging agent (get_np_policy() is a live
    alias to control_policy, so reusing the main policy_agent here would mutate whatever
    np_policy build_policy_agent() already returned)."""
    folder = folder or (str(run.dir).rstrip("/") + "/")
    stage_agent = PenSimMCPILCO(pensim_wrapper=PenSimWrapper(), **run.cfg["mc_pilco_init"])
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

    yields = []
    for i, m in enumerate(monitors):
        c = _ep_color(i, n_ep, n_expl)
        ax[0, 1].plot(m["t"], m["PAA"], color=c, lw=1, alpha=.85)
        ax[0, 2].plot(m["t"], m["Viscosity"], color=c, lw=1, alpha=.85)
        ax[1, 0].plot(m["t"], m["Fpaa"], color=c, lw=1, alpha=.85)
        ax[1, 1].plot(m["t"], m["P"], color=c, lw=1, alpha=.85)
        yields.append(yield_kg(m))
    yields = np.array(yields)
    ax[1, 2].bar(np.arange(n_ep), yields, color=[_ep_color(i, n_ep, n_expl) for i in range(n_ep)])
    ax[1, 2].axhline(ref["yield"], label=ref_lbl, **REF_STYLE)

    for axis, key in [(ax[0, 1], "PAA"), (ax[0, 2], "Viscosity"), (ax[1, 0], "Fpaa"), (ax[1, 1], "P")]:
        if key in ref:
            rt, ry = ref[key]
            axis.plot(rt, ry, label=ref_lbl, **REF_STYLE)

    ax[0, 0].set_title("Final penicillin conc per episode")
    ax[0, 0].set_xlabel("episode (exploration then trials)"); ax[0, 0].set_ylabel("P (g/L)")
    ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=8)

    ax[0, 1].axhspan(*PAA_BAND, color="green", alpha=.12, label="allowed band")
    ax[0, 1].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[0, 1].set_title("PAA conc (all episodes)"); ax[0, 1].set_xlabel("time (h)")
    ax[0, 1].set_ylabel("PAA (mg/L)"); ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize=6, ncol=2)

    ax[0, 2].axhline(VISC_MAX, color="crimson", ls="--", label=f"limit {VISC_MAX:.0f} cP")
    ax[0, 2].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[0, 2].set_title("Viscosity (all episodes)"); ax[0, 2].set_xlabel("time (h)")
    ax[0, 2].set_ylabel("viscosity (cP)"); ax[0, 2].grid(alpha=.3); ax[0, 2].legend(fontsize=6, ncol=2)

    ax[1, 0].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[1, 0].axhspan(FPAA_MIN, FPAA_MAX, color="orange", alpha=.06,
                     label=f"clamp [{FPAA_MIN:.0f},{FPAA_MAX:.0f}]")
    ax[1, 0].set_title("Fpaa setpoint (all episodes)"); ax[1, 0].set_xlabel("time (h)")
    ax[1, 0].set_ylabel("Fpaa (L/h)"); ax[1, 0].grid(alpha=.3); ax[1, 0].legend(fontsize=6, ncol=2)

    ax[1, 1].axvline(WARMUP_H, color="gray", ls=":", label=f"RL on ({WARMUP_H:g} h)")
    ax[1, 1].set_title("Penicillin trajectories (all episodes)"); ax[1, 1].set_xlabel("time (h)")
    ax[1, 1].set_ylabel("P (g/L)"); ax[1, 1].grid(alpha=.3)
    ax[1, 1].plot([], [], color="0.72", label="exploration")
    _episode_colorbar(fig, ax[1, 1], n_ep, n_expl, label="trial episode (early -> late)")
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].set_title("Penicillin yield per episode")
    ax[1, 2].set_xlabel("episode"); ax[1, 2].set_ylabel("yield (kg)")
    ax[1, 2].grid(alpha=.3, axis="y"); ax[1, 2].legend(fontsize=8)

    fig.suptitle("Single-run MC-PILCO training progression")
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

def reconstruct_gp_agent(run, idx=None):
    """Build a PenSimMCPILCO and load the trial-`idx` GP model from run.log (no training)."""
    idx = _resolve_trial(run.log, idx)
    cfg = get_config(**_build_cfg_kwargs(run.params))
    cfg["mc_pilco_init"]["log_path"] = None
    agent = PenSimMCPILCO(pensim_wrapper=PenSimWrapper(**cfg["wrapper_par"]), **cfg["mc_pilco_init"])
    agent.state_samples_history = run.log["state_samples_history"]
    agent.input_samples_history = run.log["input_samples_history"]
    agent.noiseless_states_history = run.log.get("noiseless_states_history",
                                                  run.log["state_samples_history"])
    ml = agent.model_learning
    ml.gp_inputs = run.log[f"gp_inputs_{idx}"]
    ml.gp_output_list = run.log[f"gp_output_list_{idx}"]
    ml.num_samples = ml.gp_inputs.shape[0]
    ml.dim_state = len(STATE_NAMES)
    ml.init_gp_models()
    params = run.log[f"parameters_gp_{idx}"]
    for k in range(ml.num_gp):
        ml.gp_list[k].load_state_dict(params[k])
        # Faithful to training: config runs flg_norm=False, so GPs were fit on raw deltas
        # (norm_list=1). Mirror flg_norm rather than assuming it.
        ml.norm_list[k] = (torch.max(torch.abs(ml.gp_output_list[k]))
                           if getattr(ml, "flg_norm", False) else torch.tensor(1.0, dtype=agent.dtype))
    with torch.no_grad():
        for k in range(ml.num_gp):
            ml.pretrain_gp(k)
    ml.set_eval_mode()
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
    """C.1/C.3/C.4: one-step GP fit against the trial-gp_idx training batch (dX/dP scatter +
    per-dim MSE bars)."""
    with torch.no_grad():
        _, targets, means, _ = gp_agent.get_model_learning_performance(gp_idx)
    per_dim_mse = [float(((targets[k] - means[k]) ** 2).mean()) for k in range(len(targets))]

    loX, hiX = STATE_RANGES["X"]; loP, hiP = STATE_RANGES["P"]
    tgt_dx = _denorm_delta(targets[X_IDX].ravel(), loX, hiX); prd_dx = _denorm_delta(means[X_IDX].ravel(), loX, hiX)
    tgt_dp = _denorm_delta(targets[P_IDX].ravel(), loP, hiP); prd_dp = _denorm_delta(means[P_IDX].ravel(), loP, hiP)
    r2_x = 1.0 - float(((tgt_dx - prd_dx) ** 2).sum()) / (float(((tgt_dx - tgt_dx.mean()) ** 2).sum()) or 1.0)
    r2_p = 1.0 - float(((tgt_dp - prd_dp) ** 2).sum()) / (float(((tgt_dp - tgt_dp.mean()) ** 2).sum()) or 1.0)

    gp_inputs = gp_agent.model_learning.data_to_gp_input(
        torch.tensor(gp_agent.state_samples_history[gp_idx], dtype=gp_agent.dtype),
        torch.tensor(gp_agent.input_samples_history[gp_idx], dtype=gp_agent.dtype))[:-1, :].detach().cpu().numpy()
    action_col = gp_inputs[:, ACTION_IDX]

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    sc = ax[0].scatter(tgt_dx, prd_dx, c=action_col, cmap="viridis", s=14, alpha=.8)
    lim = [min(tgt_dx.min(), prd_dx.min()), max(tgt_dx.max(), prd_dx.max())]
    ax[0].plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
    fig.colorbar(sc, ax=ax[0], label="Fs action (-1..+1)")
    ax[0].set_title(f"One-step dX: GP vs actual (R^2={r2_x:.3f})")
    ax[0].set_xlabel("actual dX (g/L per step)"); ax[0].set_ylabel("GP predicted dX")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    sc2 = ax[1].scatter(tgt_dp, prd_dp, c=action_col, cmap="viridis", s=14, alpha=.8)
    lim = [min(tgt_dp.min(), prd_dp.min()), max(tgt_dp.max(), prd_dp.max())]
    ax[1].plot(lim, lim, "r--", lw=1.5, label="perfect (y=x)")
    fig.colorbar(sc2, ax=ax[1], label="Fs action (-1..+1)")
    ax[1].set_title(f"One-step dP: GP vs actual (R^2={r2_p:.3f})")
    ax[1].set_xlabel("actual dP (g/L per step)"); ax[1].set_ylabel("GP predicted dP")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"C.1/C.3 — one-step dX/dP fit, trial {gp_idx}")
    fig.tight_layout()
    _finish(fig, out_dir, "C1_one_step_scatter.png", show)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(range(len(per_dim_mse)), per_dim_mse, color="steelblue")
    bars[X_IDX].set_color("crimson"); bars[P_IDX].set_color("darkorange")
    ax2.set_xticks(range(len(STATE_NAMES))); ax2.set_xticklabels(STATE_NAMES, rotation=45)
    ax2.set_title(f"Per-dim one-step MSE — trial {gp_idx}"); ax2.set_ylabel("MSE (normalised)")
    ax2.grid(alpha=.3, axis="y")
    for c, lbl in [("crimson", "X (Fs-driven)"), ("darkorange", "P (reward)"), ("steelblue", "other")]:
        ax2.bar([], [], color=c, label=lbl)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    _finish(fig2, out_dir, "C4_per_dim_mse.png", show)

    pd.DataFrame({"channel": STATE_NAMES, "mse_norm": per_dim_mse}).to_csv(
        Path(out_dir) / "C4_per_dim_mse.csv", index=False)
    print(f"one-step R^2: X={r2_x:.3f} P={r2_p:.3f}")
    return per_dim_mse, dict(r2_x=r2_x, r2_p=r2_p)


def plot_multistep_rollout(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False):
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
    """N particles through the RECORDED action sequence of batch `idx` (particle_pred=True)."""
    ml = agent.model_learning
    ml.set_eval_mode()
    S = np.asarray(agent.state_samples_history[idx])
    U = np.asarray(agent.input_samples_history[idx])
    T, D = S.shape
    torch.manual_seed(seed)
    x = torch.tensor(np.tile(S[0], (N, 1)), dtype=agent.dtype, device=agent.device)
    out = np.zeros((T, N, D)); out[0] = x.detach().cpu().numpy()
    with torch.no_grad():
        for t in range(1, T):
            u = torch.tensor(np.tile(U[t - 1], (N, 1)), dtype=agent.dtype, device=agent.device)
            x, _, _ = ml.get_next_state(current_state=x, current_input=u, particle_pred=True)
            out[t] = x.detach().cpu().numpy()
    return out


def plot_particle_bands(gp_agent, gp_idx, ho_idx, has_ho, out_dir, n_part=100, show=False):
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
    """Teacher-forced one-step standardised residuals, per channel."""
    ml = agent.model_learning
    tr = torch.tensor(agent.state_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    ip = torch.tensor(agent.input_samples_history[batch_idx], dtype=agent.dtype, device=agent.device)
    N = tr.shape[0]
    z = np.full((N - 1, STATE_DIM), np.nan)
    with torch.no_grad():
        for t0 in range(N - 1):
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
            nxt, _, _ = ml.get_next_state(current_state=tr[t0:t0 + 1, :],
                                          current_input=ip[t0:t0 + 1, :], particle_pred=False)
            errs[t0] = (nxt - tr[t0 + 1:t0 + 2, :]).abs().ravel().detach().cpu().numpy()
    eX = _denorm_delta(errs[:, X_IDX], *STATE_RANGES["X"])
    eP = _denorm_delta(errs[:, P_IDX], *STATE_RANGES["P"])
    return grid[:-1], eX, eP


def plot_local_error(gp_agent, gp_idx, ho_idx, has_ho, out_dir, show=False):
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


def plot_short_horizon(gp_agent, gp_idx, ho_idx, has_ho, out_dir, optim_horizon, num_anchors, show=False):
    """C.7: error growth inside vs beyond the H-step optimiser window, per-anchor H-step
    accuracy, and each anchor's H-step pure-GP rollout over the true recipe trajectory. Only
    meaningful for runs that actually used anchors -- see Run.has_anchors; caller should skip
    this otherwise."""
    Tb = gp_agent.state_samples_history[gp_idx].shape[0]
    H = int(optim_horizon)
    horizons2 = np.array(sorted({k for k in (1, 5, 10, 15, 20, H, int(1.5 * H), 2 * H, 50)
                                 if 1 <= k <= Tb - 1}))
    rmse_in2 = _kstep_errors(gp_agent, gp_idx, horizons2)
    rmse_ho2 = _kstep_errors(gp_agent, ho_idx, horizons2) if has_ho else None

    def _recipe_batch(agent, seed):
        w = PenSimWrapper(seed_offset=agent.system.seed_offset)
        a0 = lambda state, decision_idx: np.array([0.0])
        states, inputs, _ = w.rollout(s0=initial_state_norm(), policy=a0, T=CONTROL_H,
                                      dt=agent.T_sampling, noise=agent.std_meas_noise, seed=seed)
        return states, inputs

    def _window_rollouts(agent, true, inp, origins, H):
        ml = agent.model_learning
        true_t = torch.tensor(true, dtype=agent.dtype, device=agent.device)
        inp_t = torch.tensor(inp, dtype=agent.dtype, device=agent.device)
        T = true_t.shape[0]
        preds, rmseH = [], []
        for t0 in origins:
            h = int(min(H, T - 1 - t0))
            cur = true_t[t0:t0 + 1, :]
            traj, sq = [cur], []
            for j in range(1, h + 1):
                cur, _, _ = ml.get_next_state(current_state=cur,
                                              current_input=inp_t[t0 + j - 1:t0 + j, :], particle_pred=False)
                traj.append(cur)
                sq.append(((cur - true_t[t0 + j:t0 + j + 1, :]).ravel() ** 2).detach().cpu().numpy())
            preds.append(torch.cat(traj, 0).detach().cpu().numpy())
            rmseH.append(np.sqrt(np.mean(np.stack(sq, 0), axis=0)) if sq else np.full(STATE_DIM, np.nan))
        return preds, np.stack(rmseH)

    rs, ri = _recipe_batch(gp_agent, gp_agent.system.seed_offset)
    Tr = rs.shape[0]
    anchor_t0 = np.linspace(0, max(1, Tr - 1 - H), num_anchors).round().astype(int)
    preds_a, rmseH = _window_rollouts(gp_agent, rs, ri, anchor_t0, H)
    anchor_hours = WARMUP_H + anchor_t0 * T_SAMPLING
    xH = _denorm_delta(rmseH[:, X_IDX], *STATE_RANGES["X"])
    pH = _denorm_delta(rmseH[:, P_IDX], *STATE_RANGES["P"])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); hcol = "purple"
    a00 = ax[0, 0]
    for dim, name, color in [(X_IDX, "X", "crimson"), (P_IDX, "P", "steelblue")]:
        if has_ho:
            y = _denorm_delta(rmse_ho2[:, dim], *STATE_RANGES[name]); m = np.isfinite(y)
            a00.plot(horizons2[m], y[m], "-o", color=color, lw=2, label=f"{name} held-out")
        y = _denorm_delta(rmse_in2[:, dim], *STATE_RANGES[name]); m = np.isfinite(y)
        a00.plot(horizons2[m], y[m], "--o", color=color, lw=1.4, alpha=.45, label=f"{name} in-sample")
    a00.axvspan(horizons2.min(), H, color="green", alpha=.06)
    a00.axvline(H, color=hcol, ls=":", lw=1.6, label=f"optimiser horizon H={H}")
    a00.set_xlabel(f"horizon k (steps)  [1 step = {T_SAMPLING:g} h]"); a00.set_ylabel("RMSE (g/L)")
    a00.set_title(f"Error growth: inside (green) vs beyond the H={H} optimiser window")
    a00.grid(alpha=.3); a00.legend(fontsize=7)

    a01 = ax[0, 1]
    a01.plot(anchor_hours, xH, "-o", color="crimson", lw=2, label="X")
    a01.plot(anchor_hours, pH, "-o", color="steelblue", lw=2, label="P")
    a01.set_xlabel("anchor launch time (h)"); a01.set_ylabel(f"{H}-step-window RMSE (g/L)")
    a01.set_title(f"Per-anchor {H}-step accuracy across the batch")
    a01.grid(alpha=.3); a01.legend(fontsize=8)

    norm = plt.Normalize(vmin=float(anchor_hours.min()), vmax=float(anchor_hours.max()))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
    for a1, dim, name in [(ax[1, 0], X_IDX, "X"), (ax[1, 1], P_IDX, "P")]:
        t_full = WARMUP_H + np.arange(Tr) * T_SAMPLING
        true_phys = _denorm_phys(rs[:, dim], name)
        (rl,) = a1.plot(t_full, true_phys, "k-", lw=2, label="recipe true", zorder=3)
        for a, t0 in enumerate(anchor_t0):
            c = plt.cm.viridis(norm(anchor_hours[a]))
            tw = WARMUP_H + (t0 + np.arange(preds_a[a].shape[0])) * T_SAMPLING
            a1.plot(tw, _denorm_phys(preds_a[a][:, dim], name), "-", color=c, lw=1.6, alpha=.9)
            a1.axvline(anchor_hours[a], color="0.7", ls=":", lw=.6, alpha=.5, zorder=0)
            a1.plot(anchor_hours[a], true_phys[t0], "o", color=c, ms=5, mec="k", mew=.5, zorder=4)
        a1.set_xlabel("time (h)"); a1.set_ylabel(f"{name} (g/L)")
        a1.set_title(f"{H}-step GP rollouts from recipe anchors — {name}")
        a1.grid(alpha=.3); fig.colorbar(sm, ax=a1).set_label("anchor launch time (h)")
        marker_proxy = Line2D([0], [0], marker="o", color="w", mec="k", mfc="0.6", ms=6,
                              label="anchor launch (truth injected)")
        a1.legend(handles=[rl, marker_proxy], fontsize=7, loc="upper left")
    fig.suptitle(f"Short-horizon GP diagnostic — model@trial {gp_idx}, optimiser H={H} steps ({H * T_SAMPLING:g} h)")
    fig.tight_layout()
    _finish(fig, out_dir, "C7_short_horizon.png", show)
    return fig

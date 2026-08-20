
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MC-PILCO"))

import policy_learning.MC_PILCO as MCP

import numpy as np
import torch

from utils.peni_env_setup import PenSimEnv
from utils.recipe import Recipe, RecipeCombo
from utils.constants import NUM_STEPS, STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)

from utils.ode_patch import patch_fastodeint
patch_fastodeint()

# Minimal RL state: agent-relevant channels only. T, pH, O2, CO2outgas, PAA are PID/recipe-held
# (the agent neither drives nor needs them as feedback), so they are dropped from the observed state.
# `time` stays as the deterministic clock (see TIME_IDX). STATE_RANGES below is kept as a superset
# lookup table; only the channels named here are actually modelled/observed/conditioned on.
#
# `Viscosity` is here because it is the observed FAILURE MODE. Measured over 8 held-out batches of a
# trained policy, every batch peaking below ~111 cP yielded >=2678 kg while the two peaking at 158
# and 164 cP collapsed to ~1000 kg (and a training episode at 184 cP gave 606 kg) -- sustained feed
# thickens the broth, oxygen transfer fails, and penicillin DEGRADES instead of accumulating.
# Without this channel the GP cannot represent that mechanism, the policy cannot react to it, and no
# risk term can price it: the failure is literally invisible to every part of the agent.
# Every other module indexes channels either by name (STATE_NAMES.index(...)) or via
# `len(STATE_NAMES)` for the action column (see wt_mass_balance.py's ACTION_COL and
# penicillin_cost.py's P_IDX/WT_IDX/VISC_IDX/TIME_IDX), so both are derived from this list rather
# than hardcoded.
STATE_NAMES = ["Wt", "X", "P", "Viscosity", "time"]
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 1

# X, P, Viscosity are offline lab assays on the real plant (12h sampling + 4h analysis
# delay -- see peni_env_setup.py's X_offline/P_offline/Viscosity_offline, which reads the
# simulator's own delayed/held channel). Wt and time stay online/undelayed.
#
# This is the SIMPLE, uniform mechanism: whatever's in this set is read from the delayed
# proxy EVERYWHERE (GP training, cost, and the policy alike -- see _read/extract_state's
# use_offline_measurements below) -- there is no true-vs-measured split, so it's self-
# consistent by construction (the GP just learns dynamics on the held signal directly).
# This is DIFFERENT from, and MUST NOT be combined with, the MC-PILCO4PMS-style asymmetric
# split (pms_visc_delay / _ZOHPolicyProxy / PenSimMCPILCODelayed below), which keeps the GP
# on the TRUE signal and only holds what the policy is called with -- PenSimWrapper.__init__
# asserts these two are never both targeting Viscosity at once. Only Viscosity is in this
# set (X/P were explored earlier but descoped -- see git history if ever wanted again).
DELAYED_OFFLINE_NAMES = {"Viscosity"}

# Includes "S": inert for every current STATE_NAMES (none of them carry "S"), only takes effect
# once a script opts S into STATE_NAMES via set_state_names() -- see that function's docstring.
STATE_LOG_CHANNELS = {"Wt", "X", "P", "S"}
STATE_LOG_FLOOR = 1e-6

WARMUP_H = 0
T_SAMPLING = 5
STEPS_PER_DECISION = int(round(T_SAMPLING / STEP_IN_HOURS))

K_WARM = max(1, int(round(WARMUP_H / STEP_IN_HOURS)))
WARMUP_H_EFF = K_WARM * STEP_IN_HOURS
CONTROL_H = 230.0 - WARMUP_H_EFF

# Dual-phase pivot: PIVOT_HOURS does double duty for DualPhaseModelLearning (see
# model_learning_dual_phase.py) -- (1) PIVOT_STEP is where training data is HARD-split between
# phase1/phase2 (unchanged, still a hard boundary), and (2) PIVOT_HOURS is also the CENTER of
# the sigmoid blend weight w(t) that combines phase1/phase2's predictions at rollout time (see
# BLEND_HALF_WIDTH_HOURS below). PIVOT_STEP (decision index, not hour count) is what
# PenSimMCPILCOMultiPhase.apply_policy/rollout iterate over.
PIVOT_HOURS = 90.0
PIVOT_STEP = int(round(PIVOT_HOURS / T_SAMPLING))

# Sigmoid blend half-width (hours): with the default PIVOT_HOURS=90, w(t) is ~0.01 at
# PIVOT_HOURS - BLEND_HALF_WIDTH_HOURS = 50h and ~0.99 at PIVOT_HOURS + BLEND_HALF_WIDTH_HOURS
# = 130h (the standard "1%/99%" convention for a logistic's effective width -- see
# DualPhaseModelLearning._blend_weight).
BLEND_HALF_WIDTH_HOURS = 40.0

FPAA_MIN, FPAA_MAX = 0.0, 15.0

# 1 = 100%
# making FS 50% because the initial explorations are 10%, so policy going outside the exploration is risky,
# as there is no data behind them.
# Also, it would be sensible to compare it with the BO baselines in this way.
FS_SCALE = 0.5
# FS_SCALE = 1 #for ceiling test

# Recipe channels treated as NUTRITION, and therefore shifted by rollout(feed_delay_h=...) --
# an intervention that delays the culture by moving the whole feed programme later, instead of
# cutting Fs and then jumping back into the middle of the unshifted recipe.
#
# Deliberately excluded:
#   PAA   ctrl_flags.Raman_spec == 2 and rollout sets bypass_paa_pid = False, so a PID drives
#         PAA to a setpoint of 1200 and the commanded Fpaa is overridden. Shifting it would
#         look like an intervention while changing essentially nothing.
#   WATER dilution/evaporation control, not nutrition. Its profile is large and non-monotone
#         (0 -> 500 @75h -> 100 -> 0 -> 400 @170h) and it drives Wt, a CONSTRAINED variable, so
#         shifting it would entangle a yield effect with a constraint effect.
#   FG / PRES / DISCHARGE   plant schedule rather than feed. Holding these on the wall clock is
#         the whole point: it isolates a delay of the CULTURE from a delay of the operation.
FEED_CHANNELS = (FS, FOIL)

# Two mutually-exclusive action parameterisations, selected PER PenSimWrapper INSTANCE via
# `action_mode` (see __init__). Default "residual" is what every run predating this flag used, so
# nothing that doesn't explicitly opt in changes behaviour.
#
#   "residual"  Fs = recipe_Fs(t) * (1 + FS_SCALE * a).  a = 0 IS the recipe, and the action bound
#               itself confines the policy to a +/-FS_SCALE band around it -- the parameterisation
#               is doing safety work.
#   "absolute"  Fs = FS_ABS_MIN + (FS_ABS_MAX - FS_ABS_MIN) * (a + 1) / 2, i.e. feed is free to
#               fluctuate over the whole [FS_ABS_MIN, FS_ABS_MAX] band and the recipe is NOT
#               reachable as a single action value (a = 0 is mid-range feed, ~100 L/h by default,
#               against a recipe that runs 8 L/h at 3h and 80-116 L/h late). Consequences worth
#               knowing before using it:
#                 - the policy's reachable set is now much larger than its data support, so the
#                   optimiser can exploit GP extrapolation; only state_clamp and the cost penalties
#                   push back.
#                 - `fs_k` is constant across a whole T_SAMPLING window, so no action sequence
#                   reproduces the recipe exactly (the profile has 4h resolution before 24h).
#                 - INCOMPATIBLE with the Wt mass-balance prior mean: wt_mass_balance.py's
#                   `d_wt = fs_i * (1 + FS_SCALE * a)` hardcodes the residual formula against the
#                   recipe's integrated feed. Use a plain-RBF config (model_learning_baseline.py),
#                   which is what config_single_phase_absolute.py asserts.
ACTION_MODES = ("residual", "absolute")
FS_ABS_MIN, FS_ABS_MAX = 0.0, 200.0


def fs_from_action(a, fs_recipe, action_mode="residual",
                   fs_abs_min=FS_ABS_MIN, fs_abs_max=FS_ABS_MAX):
    """Physical substrate feed (L/h) for a normalised action a in [-1, 1]. THE single
    action->Fs map: PenSimWrapper.rollout calls it, and eval/diagnostic code that needs to go
    the other way (or shade the reachable band) should invert THIS rather than re-deriving the
    formula -- see eval_single_phase_lib.py's `a_fs = (ratio - 1) / FS_SCALE`, which is a
    residual-only inversion and is meaningless under "absolute"."""
    if action_mode == "residual":
        return fs_recipe * (1.0 + FS_SCALE * a)
    return fs_abs_min + (fs_abs_max - fs_abs_min) * (a + 1.0) / 2.0


STATE_RANGES = {

    "T":         (296.0, 302.0),
    "O2":        (0.15,  0.25),
    "CO2outgas": (0.0,   4.0),
    "pH":        (5.5,   7.5),
    "Wt":        (np.log(5.0e4), np.log(1.3e5)),
    "PAA":       (600,   1800.0),
    # Substrate: fed-batch keeps it near zero (median ~1.6e-3 g/L) but it SPIKES to ~14 g/L under
    # overfeeding -- ~5 orders of magnitude, so it is log-encoded (a linear band would pin the whole
    # operating range at z=-1). The spike is the substrate-accumulation warning that precedes a crash.
    # Upper bound 20 leaves headroom above the ~14 g/L observed spike; lower bound is
    # STATE_LOG_FLOOR since (unlike X/P below) no reachable-operating-floor measurement for S has
    # been taken here yet -- re-derive from real batches before trusting resolution near zero.
    # ONLY used when S is switched into STATE_NAMES via set_state_names() -- see that function.
    "S":         (np.log(STATE_LOG_FLOOR), np.log(20.0)),
    # Lower bound is the reachable operating floor, NOT STATE_LOG_FLOOR (the encode-clamp used to
    # avoid log(0) when a value is genuinely zero, e.g. P at batch start). Using STATE_LOG_FLOOR=1e-6
    # here made the normalisation range span 17.5 log units while the real production band (X~15-35
    # g/L) only spans ~0.07 of [-1,1] -- crushing the policy-relevant region into a sliver the RBF/GP
    # lengthscales can't resolve. These floors are set to the smallest physically reachable values
    # instead.
    "X":         (np.log(0.05), np.log(40.0)),
    "P":         (np.log(0.01), np.log(45.0)),
    # Measured over 619 logged batches: min 4.1, median 49.5, p99 142.6, max 188.8 cP. Linear, NOT
    # log-encoded -- the span is only ~46x (so a single RBF lengthscale copes), the decision-relevant
    # region is the 100-190 top end where linear gives the better resolution, and staying off the
    # exp() decode path keeps this channel out of the rollout blow-up mode that {Wt,X,P} needed
    # clamping for. Upper bound 200 leaves headroom above the worst observed batch.
    "Viscosity": (0.0,   120.0),
    "time":      (0.0,   230.0),
    # Carbon evolution rate: online off-gas, no lab assay and no measurement delay (see
    # peni_env_setup.py's x.CER, computed from Fg/CO2outgas each native step). Measured over the
    # 40-batch diagnostic sweep in evaluations/ryu_mu_check: 0.028 at inoculation, median 1.30,
    # p99 2.25, max 2.27. Linear, NOT log-encoded -- same reasoning as Viscosity: the span is only
    # ~80x, and the decision-relevant band is the 0.5-2.3 top end where linear gives the better
    # resolution. Upper bound 2.5 leaves headroom above the worst observed batch.
    # ONLY used when CER is switched into STATE_NAMES via set_state_names() -- see that function.
    "CER":       (0.0,   2.5),
}

# CHANGED_THIS added
TIME_IDX = STATE_NAMES.index("time")
_t_lo, _t_hi = STATE_RANGES["time"]
TIME_DELTA_NORM = 2.0 * T_SAMPLING / (_t_hi - _t_lo)
# time is deterministic, but torch's Gaussian samplers reject a variance/scale of exactly 0
# (MultivariateNormal needs a positive-definite diagonal at t=0; Normal needs scale > 0 at each
# GP step). So the time channel carries a tiny positive jitter wherever a distribution is built
# over the full state; the drawn time value is negligible at t=0 and overwritten thereafter.
TIME_INIT_VAR = 1e-6


def set_t_sampling(new_t_sampling):
    """Override T_SAMPLING for the rest of this process, recomputing every module constant
    derived from it: STEPS_PER_DECISION (PenSimWrapper.rollout's underlying-ODE-step multiplier
    -- see its own `spd = STEPS_PER_DECISION` line, which is NOT re-derived from the `dt` the
    caller passes in, so leaving it stale desyncs the simulator's decision cadence from what the
    agent thinks T_sampling is), PIVOT_STEP (informational only -- config_dual_phase.get_config
    recomputes its own pivot_step from pivot_hours/T_SAMPLING rather than reading this), and
    TIME_DELTA_NORM (the deterministic `time` channel's fixed per-decision delta -- see
    model_learning_det_time.DETERMINISTIC_CHANNELS).

    MUST be called before config_single_phase[_baseline[_time]] / config_dual_phase[_baseline
    [_time]] (or anything else that does `from mcpilco.pensim_wrapper import T_SAMPLING` /
    `TIME_DELTA_NORM` / etc.) is imported for the FIRST TIME in this process -- `from X import Y`
    snapshots Y's value at that moment, so importing one of those modules first and calling this
    after leaves them silently using the stale default. See the four
    experiments/0{2,3}_mcpilco_*_baseline*.py drivers' --t_sampling handling: they defer their
    `from mcpilco.config_... import get_config` import to happen after this call.

    Only the FIRST call in a given process is guaranteed correct for downstream modules that
    haven't been imported yet; a second call with a DIFFERENT value won't retroactively fix
    modules already imported (and hence already cached) with the first value. Irrelevant for the
    normal CLI use case (one process per run) -- only matters if calling this repeatedly with
    different values inside one long-lived Python session (e.g. a notebook)."""
    global T_SAMPLING, STEPS_PER_DECISION, PIVOT_STEP, TIME_DELTA_NORM
    T_SAMPLING = new_t_sampling
    STEPS_PER_DECISION = int(round(T_SAMPLING / STEP_IN_HOURS))
    PIVOT_STEP = int(round(PIVOT_HOURS / T_SAMPLING))
    TIME_DELTA_NORM = 2.0 * T_SAMPLING / (_t_hi - _t_lo)


def set_state_names(new_state_names):
    """Override STATE_NAMES for the rest of this process, recomputing every derived constant.

    Same import-order contract as set_t_sampling() above, and for exactly the same reason: the
    channel INDICES are snapshotted at import time by `X_IDX = STATE_NAMES.index("X")`-style
    module-level statements in penicillin_cost.py (P/WT/VISC/TIME_IDX), wt_mass_balance.py
    (WT_IDX/TIME_IDX/ACTION_COL), model_learning_det_time.py (WT_GP_IDX/RECIPE_MEAN_GP_IDX) and
    recipe_trajectory_mean.py (TIME_IDX). So this MUST be called before config_single_phase[
    _baseline[_time]] / config_dual_phase[...] -- or anything they transitively import -- is
    imported for the FIRST TIME in this process. See
    experiments/02_mcpilco_single_phase_baseline_cer.py for the deferred-import pattern.

    APPEND-ONLY. Every existing channel must keep its current position: the indices above are
    read from the NEW list, but training artefacts, monitor dumps and eval code written against
    the old layout are not, so reordering silently reinterprets old columns. Enforced below.

    Nothing else needs touching to add a channel: `active_dims`/`lengthscales_init` are built
    from `gp_input_dim = STATE_DIM + ACTION_DIM` in config_single_phase.py, and x0 mean/variance
    are MEASURED per-channel by _measure_init_state_stats(), so both pick the new column up
    automatically. The new name does need a STATE_RANGES entry (checked below) and must be a real
    PenSimPy batch-data channel, since _read() reaches it via getattr(batch_x, name).
    """
    global STATE_NAMES, STATE_DIM, TIME_IDX, VISC_IDX, TIME_DELTA_NORM
    new_state_names = list(new_state_names)
    if new_state_names[:len(STATE_NAMES)] != STATE_NAMES:
        raise ValueError(
            f"set_state_names is append-only: {new_state_names} does not start with the current "
            f"{STATE_NAMES}. Reordering or removing channels would silently reinterpret every "
            f"index baked into penicillin_cost / wt_mass_balance / model_learning_det_time.")
    missing = [n for n in new_state_names if n not in STATE_RANGES]
    if missing:
        raise ValueError(f"no STATE_RANGES entry for {missing}; add one before using it as a state")
    STATE_NAMES = new_state_names
    STATE_DIM = len(STATE_NAMES)
    TIME_IDX = STATE_NAMES.index("time")
    VISC_IDX = STATE_NAMES.index("Viscosity")
    _lo, _hi = STATE_RANGES["time"]
    TIME_DELTA_NORM = 2.0 * T_SAMPLING / (_hi - _lo)


INIT_STATE_PHYS = {"T": 297.98, "DO2": 12.33, "O2": 0.189, "CO2outgas": 1.86,
                   "pH": 6.49, "Wt": 97907.0, "PAA": 1200.0, "X": 22.80, "P": 16.73,
                   "time": WARMUP_H}


WT_SOFT = (7.0e4, 1.1e5) # same as indpensim - valid
WT_OVERFLOW = 1.2e5
P_CRASH = 55.0
PAA_BAND = (800.0, 1600.0)
VISC_MAX = 100.0 # was 150, but indpensim use 100, changing.

# Exploration screening: a batch whose total penicillin yield lands below FAILED_YIELD_KG has
# collapsed. Such batches are discarded and re-rolled rather than fed to the GPs, so the initial
# model is not built on failed batches (see PenSimMCPILCO.get_data_from_system).
# FAILED_YIELD_KG = 2000.0
# MAX_EXPLORATION_RETRIES = 20

# Reserved seed block for population-level measurement rollouts (x0, recipe-trajectory prior means).
# These measure statistics that should be independent of the run seed, so they must stay clear of
# every run's training/eval range (seed_offset = seed*1000 + episode, i.e. seed*1000..seed*1000+~900)
# and of setup_high_feed_probes' seed_offset+900 block. 900_000 requires seed >= 900 before any
# collision is even possible.
MEASUREMENT_SEED_BASE = 900_000


def _normalise(value, lo, hi):
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def encode_state_value(name, value):
    """Map a physical state value into the latent value used by the GP/model."""
    if name in STATE_LOG_CHANNELS:
        if torch.is_tensor(value):
            return torch.log(torch.clamp(value, min=STATE_LOG_FLOOR))
        return np.log(np.maximum(value, STATE_LOG_FLOOR))
    return value


def decode_state_value(name, value):
    """Map a latent GP value back to physical units."""
    if name in STATE_LOG_CHANNELS:
        if torch.is_tensor(value):
            return torch.exp(value)
        return np.exp(value)
    return value


def batch_yield_kg(mon):
    """Total penicillin yield (kg) of one batch, from its monitor dict.

    Sums the per-step `yield_per_run` captured from PenSimEnv.step -- exactly PenSimPy's
    `batch_yield`, so it is the same number the recipe/BO baselines and eval_utils.yield_kg report.
    """
    return float(np.sum(mon["yield_per_run"]))



def _read_offline(batch_x, name, i):
    """Most recently RELEASED offline reading at/before native index i (causal
    zero-order hold between lab-assay results). Falls back to the true online
    value if no offline sample has been released yet -- only possible in the
    first ~1h of a batch, before peni_env_setup.py's earliest release fires."""
    y = getattr(batch_x, name + "_offline").y
    for j in range(i, -1, -1):
        if not np.isnan(y[j]):
            return float(y[j])
    return float(getattr(batch_x, name).y[i])


def _read(batch_x, name, i, use_offline_measurements=False):
    """Physical value of `name` at native index i. pH is stored as 10^(-pH)
    mid-batch, so invert it back to pH units here. `time` is not a batch channel;
    native index i maps to absolute batch time (i+1)*STEP_IN_HOURS hours.
    When use_offline_measurements is True, DELAYED_OFFLINE_NAMES channels are read
    via the delayed/held lab-assay proxy (_read_offline) instead of the always-
    available online ODE state."""
    if name == "time":
        return (i + 1) * STEP_IN_HOURS
    if name == "pH":
        return -np.log10(max(float(getattr(batch_x, "pH").y[i]), 1e-12))
    if use_offline_measurements and name in DELAYED_OFFLINE_NAMES:
        return _read_offline(batch_x, name, i)
    return float(getattr(batch_x, name).y[i])


def extract_state(batch_x, k, use_offline_measurements=False):
    i = max(k - 1, 0)
    return np.array([
        _normalise(encode_state_value(n, _read(batch_x, n, i, use_offline_measurements)), *STATE_RANGES[n])
        for n in STATE_NAMES
    ])


_init_state_cache = {}

# Floor on the empirical initial_state_var so no channel (e.g. one with near-zero measured
# batch-to-batch spread at K_WARM) collapses to a literal 0, which the Gaussian particle sampler
# cannot handle.
INIT_STATE_VAR_FLOOR = 1e-4


def _measure_init_state_stats(num_batches=6):
    """Normalised handover state mean AND variance (x0), MEASURED by rolling pure-recipe batches
    up to K_WARM.

    Derived from the live WARMUP_H / STATE_RANGES so it can never go stale. (INIT_STATE_PHYS below is
    a hard-coded snapshot taken for WARMUP_H=120; it silently became wrong the moment WARMUP_H changed,
    which made policy optimisation launch its imagined rollouts from a fully-grown reactor while the
    real batch started at inoculation.)

    Uses MEASUREMENT_SEED_BASE rather than seed 0: this measures a population-level statistic that
    should be independent of the run seed, but with the old hardcoded seed_offset=0 a run started
    with --seed 0 (wrapper_par seed_offset=0) would measure x0 from the EXACT SAME simulator
    realisations (seeds 0..num_batches-1) as its own training/exploration episodes.

    NOT CACHED ON DISK, on purpose. A previous version wrote results/_init_state_cache/x0_kwarm{K}.npy
    keyed only by K_WARM -- so editing STATE_RANGES (x0 is stored NORMALISED) silently reused a stale
    x0. Nothing in the training path may depend on anything under results/: those artefacts carry the
    STATE_RANGES / WARMUP_H / code version of whenever they happened to be written. Measuring fresh
    costs `num_batches` recipe rollouts once per process, memoised below.
    """
    if K_WARM in _init_state_cache:
        return _init_state_cache[K_WARM]

    recipe_policy = lambda state, decision_idx: np.array([0.0])
    fresh = PenSimWrapper(seed_offset=MEASUREMENT_SEED_BASE)
    np_state = np.random.get_state()
    x0_samples = np.stack([fresh.rollout(s0=None, policy=recipe_policy, T=CONTROL_H, dt=T_SAMPLING,
                                         noise=None, seed=MEASUREMENT_SEED_BASE + i)[0][0]
                           for i in range(num_batches)])
    np.random.set_state(np_state)

    x0_mean = np.mean(x0_samples, axis=0)
    x0_var = np.maximum(np.var(x0_samples, axis=0), INIT_STATE_VAR_FLOOR)

    _init_state_cache[K_WARM] = (x0_mean, x0_var)
    return x0_mean, x0_var


def initial_state_norm():
    """Normalised state at the WARMUP_H turn-on point (RL phase x0), measured from the recipe."""
    return _measure_init_state_stats()[0]


def initial_state_var_norm():
    """Per-channel empirical variance of the normalised x0, measured from the recipe.

    Replaces a uniform hardcoded value (previously 0.01 for every channel) with the actual measured
    batch-to-batch spread at K_WARM -- channels with a compressed normalised range (see STATE_RANGES)
    or genuinely low real-world spread no longer get an artificially large particle-initialisation
    variance relative to what they can resolve. Caller overrides TIME_IDX separately (see
    config_single_phase.py) since time is deterministic.
    """
    return _measure_init_state_stats()[1]


# Two dither timescales (see PROBE_PLAN). At the default T_SAMPLING=5 these round to 8- and
# 2-decision blocks (40h and 10h) over a 45-decision batch. Both are balanced -- equal time at
# +level and -level, up to the odd trailing block, which leaves a residual mean action of only
# ~+0.02..0.04 -- so what the pair varies is the TIMESCALE on which feed moves, not how much of it
# there is on average.
PROBE_SLOW_BLOCK_HOURS = 40.0
PROBE_FAST_BLOCK_HOURS = 12.0

# Where the sustained +/- steps switch on. PIVOT_HOURS is the growth->production boundary the
# dual-phase model already splits on, so the step lands entirely inside the production window --
# the region where the feed->P response reverses sign and where a step's effect ACCUMULATES rather
# than showing up within a decision or two. Deliberately bound to the module-level default and NOT
# to config_dual_phase's --pivot_hours: that flag moves where the MODEL splits, while this is a
# property of the PROCESS, and letting it drift per run would make probe sets incomparable across
# runs that differ only in their model split.
PROBE_PROD_START_HOURS = PIVOT_HOURS


def _alternating_blocks_profile(level, n, block_hours):
    """+level/-level square wave in `block_hours`-wide blocks."""
    block_decisions = max(1, int(round(block_hours / T_SAMPLING)))
    block = (np.arange(n) // block_decisions) % 2
    return np.where(block == 0, level, -level)


def _prod_step_profile(level, n, start_hours=PROBE_PROD_START_HOURS):
    """0 (pure recipe) through the growth phase, then a single sustained `level` to batch end.

    One switch, held for the whole production window -- the opposite extreme from the dithers:
    they answer "how does P respond to feed moving", this answers "how does P respond to feed
    STAYING moved", which is the accumulating response the dithers deliberately average out.
    """
    start = min(max(int(round(start_hours / T_SAMPLING)), 0), n)
    return np.concatenate([np.zeros(start), np.full(n - start, level)])


def _dither_slow_profile(level, n):
    return _alternating_blocks_profile(level, n, PROBE_SLOW_BLOCK_HOURS)


def _dither_fast_profile(level, n):
    return _alternating_blocks_profile(level, n, PROBE_FAST_BLOCK_HOURS)


def _prod_step_up_profile(level, n):
    return _prod_step_profile(level, n)


def _prod_step_down_profile(level, n):
    return _prod_step_profile(-level, n)


PROBE_SHAPES = {
    "dither_slow": _dither_slow_profile,
    "dither_fast": _dither_fast_profile,
    "prod_step_up": _prod_step_up_profile,
    "prod_step_down": _prod_step_down_profile,
}

# The probe SET, in order (see setup_high_feed_probes): (shape, index into `levels`). Probe i takes
# entry i % 4, so the default num_probes=4 rolls exactly one of each and higher counts replicate the
# set under fresh batch realisations. Read as a designed experiment rather than four independent
# probes:
#   0/1  balanced dither at two timescales. Same zero-mean excitation, different frequency, so the
#        pair separates the fast (within-decision) response from the slow (accumulated) one -- the
#        effect of interest here being the slow one.
#   2/3  sustained +step and -step over the production window, at the SAME magnitude so the pair
#        brackets the feed->P sign reversal symmetrically from both sides. This is what the GP
#        cannot get anywhere else: exploration re-draws its level every ~19h and the dithers flip
#        sign by construction, so neither ever holds feed off-recipe long enough for the late-phase
#        response to integrate and reverse.
# The two step entries share level index 2 on purpose -- a +/- pair at different magnitudes would
# confound "which side of nominal" with "how far from nominal", which is exactly the confound the
# bracket exists to remove.
PROBE_PLAN = (
    ("dither_slow", 0),
    ("dither_fast", 1),
    ("prod_step_up", 2),
    ("prod_step_down", 2),
)


# MC-PILCO4PMS-style Viscosity delay: 12h lab-assay sampling cadence + 4h analysis turnaround
# (matches peni_env_setup.py's Off_line_m/Off_line_delay). Unlike DELAYED_OFFLINE_NAMES above,
# this ONLY affects what control_policy is called with -- see _ZOHPolicyProxy and
# PenSimMCPILCODelayed -- never extract_state's output, which stays the true instantaneous value.
VISC_SAMPLE_INTERVAL_H = 12.0
VISC_ANALYSIS_DELAY_H = 4.0
VISC_IDX = STATE_NAMES.index("Viscosity")


def build_release_table(num_decisions, T_sampling, sample_interval_h, analysis_delay_h):
    """held_source[d] = decision index of the most recently RELEASED lab sample as of decision
    d (None if no sample has ever been released yet -- e.g. very early decisions when
    analysis_delay_h > T_sampling -- which naturally falls back to the true current state,
    same as the real protocol: there's nothing to hold yet).

    `source_decision = floor(epoch / T_sampling)` snaps each sample down to the latest
    decision-grid point at/before it was actually drawn (never a decision after -- staying
    causal), which makes the realized delay slightly MORE conservative (staler) than the
    continuous-time 4-16h sawtooth, never optimistic. The table is periodic with period
    lcm(T_sampling, sample_interval_h) (60h / 12 decisions for the defaults) -- useful as a
    unit-test invariant.
    """
    releases = []
    n = 0
    while True:
        epoch = n * sample_interval_h
        source_decision = int(np.floor(epoch / T_sampling))
        release_decision = int(np.ceil((epoch + analysis_delay_h) / T_sampling))
        if source_decision >= num_decisions and release_decision >= num_decisions:
            break
        releases.append((release_decision, source_decision))
        n += 1
    held = [None] * num_decisions
    for d in range(num_decisions):
        candidates = [src for (rel, src) in releases if rel <= d]
        held[d] = candidates[-1] if candidates else None
    return held


class _ZOHPolicyProxy:
    """Wraps a policy callable so `idx` channels are zero-order-held from the most recently
    released sample (per held_source[t]); every other channel passes through unchanged.
    Shape-agnostic (`state[..., idx]`) -- works for the real rollout's 1D per-decision numpy
    state and the particle rollout's 2D (num_particles, state_dim) torch tensor alike, and for
    both call conventions (`policy(state, t)` and `control_policy(state, t=t, p_dropout=...)`)
    since `t` binds positionally-or-by-keyword either way."""

    def __init__(self, policy, held_source, idx):
        self._policy = policy
        self._held_source = held_source
        self._idx = list(idx)
        self._buf = []

    def __call__(self, state, t, **kwargs):
        self._buf.append(state)
        # self._buf[i] is only guaranteed to be the state at decision i because calls arrive in
        # strict 0,1,2,... order (true for every current caller -- see class docstring). If a
        # future change to the vendored apply_policy loop ever violates that, this must fail
        # loudly (wrong buffer index -> silently wrong held state) rather than train on garbage.
        assert len(self._buf) == t + 1, (
            f"_ZOHPolicyProxy called out of order: t={t} but this is call #{len(self._buf)} "
            "-- buffer indexing assumes strictly sequential calls starting at t=0"
        )
        src = self._held_source[t] if t < len(self._held_source) else self._held_source[-1]
        if src is not None and src != t:
            held = self._buf[src]
            state = state.clone() if torch.is_tensor(state) else state.copy()
            state[..., self._idx] = held[..., self._idx]
        return self._policy(state, t, **kwargs)

    def __getattr__(self, name):
        # Dunder lookups (__deepcopy__, __reduce_ex__, ...) must fail fast, not delegate: since
        # _policy is set in __init__'s first line, __getattr__ should never fire for it under
        # normal use -- but copy.deepcopy/pickle probe dunders via getattr BEFORE __init__ runs
        # (e.g. on a bare __new__'d instance), and delegating would recurse into looking up
        # `self._policy` itself (also missing), causing infinite recursion instead of a clean
        # AttributeError.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._policy, name)


class PenSimWrapper:
    """One PenSimPy batch: recipe warmup -> PAA-increment RL control, as MC-PILCO arrays."""

    def __init__(self, seed_offset=0, use_offline_measurements=False, pms_visc_delay=False,
                 action_mode="residual", fs_abs_min=FS_ABS_MIN, fs_abs_max=FS_ABS_MAX):
        self.seed_offset = seed_offset
        # How the normalised action maps to physical Fs -- see ACTION_MODES / fs_from_action
        # above for the two options and what "absolute" costs you. Instance-level rather than a
        # module constant on purpose: unlike T_SAMPLING / STATE_NAMES (which are snapshotted at
        # import time by other modules and so need the set_*() + deferred-import dance), this is
        # only ever read inside rollout(), so it carries no import-order hazard and two wrappers
        # with different modes can coexist in one process.
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        if fs_abs_max <= fs_abs_min:
            raise ValueError(f"fs_abs_max ({fs_abs_max}) must exceed fs_abs_min ({fs_abs_min})")
        self.action_mode = action_mode
        self.fs_abs_min = float(fs_abs_min)
        self.fs_abs_max = float(fs_abs_max)
        # When True, DELAYED_OFFLINE_NAMES channels (currently just Viscosity) in the
        # decision-level state (fed to the GP, cost, and policy alike -- see extract_state)
        # come from the delayed lab-assay proxy instead of the always-available online ODE
        # state -- uniformly, no true-vs-measured split. Default False preserves today's
        # behavior for every existing caller (setup_recipe_anchors, setup_high_feed_probes,
        # _measure_init_state_stats, ...).
        self.use_offline_measurements = use_offline_measurements
        # When True, wraps rollout()'s policy argument with _ZOHPolicyProxy so Viscosity is
        # zero-order-held from the most recently released 12h/4h lab sample -- MC-PILCO4PMS
        # style: only the policy's INPUT is degraded; `states`/GP training/cost still get the
        # true instantaneous Viscosity extract_state always returns. Default False preserves
        # today's behavior.
        self.pms_visc_delay = pms_visc_delay
        # These two mechanisms are mutually exclusive FOR THE SAME CHANNEL: combining them
        # would mean extract_state already returns held Viscosity (use_offline_measurements)
        # while _ZOHPolicyProxy tries to ALSO hold it on top (pms_visc_delay) against a
        # release table computed independently of what's actually in `states` -- silently
        # double-delayed, not a real protocol. Fail loudly instead.
        if use_offline_measurements and pms_visc_delay and "Viscosity" in DELAYED_OFFLINE_NAMES:
            raise ValueError(
                "use_offline_measurements (with Viscosity in DELAYED_OFFLINE_NAMES) and "
                "pms_visc_delay cannot both be True: pick ONE Viscosity-delay mechanism -- "
                "the simple uniform one (use_offline_measurements) or the MC-PILCO4PMS "
                "asymmetric one (pms_visc_delay), not both."
            )
        self._episode = 0
        self._recipe = self._build_default_recipe()
        self.monitor = []

    @staticmethod
    def _build_default_recipe():
        return RecipeCombo(recipe_dict={
            FS: Recipe(FS_DEFAULT_PROFILE, FS), FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
            FG: Recipe(FG_DEFAULT_PROFILE, FG), PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
            DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
            WATER: Recipe(WATER_DEFAULT_PROFILE, WATER), PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
        })

    def rollout(self, s0, policy, T, dt, noise, seed=None, pid_baseline=False,
                feed_delay_h=0.0):
        """feed_delay_h > 0 shifts the FEED_CHANNELS recipe profiles later by that many hours,
        leaving every other channel on the wall clock. Default 0.0 takes the original single
        -lookup path unchanged, so training and every existing caller are bit-identical.

        Recipe.get_value_at back-fills below its first setpoint, so t - feed_delay_h < 0 needs
        no guard: it returns the first value (Fs 8 L/h, Foil 22), i.e. the culture is held at
        the initial feed rate for the first feed_delay_h hours -- minimal but nonzero, delayed
        rather than starved. It also forward-fills past the last setpoint, so the tail of a
        shifted batch sits on the final plateau for feed_delay_h hours longer; log cumulative
        Fs if you need to confirm the disturbance is not quietly adding substrate."""
        env = PenSimEnv(recipe_combo=self._recipe, fast=True)
        if seed is not None:
            np.random.seed(seed)
            env.random_seed_ref = seed
        else:
            env.random_seed_ref = self._episode + self.seed_offset
        _, bx = env.reset()

        spd = STEPS_PER_DECISION
        k_warm = K_WARM
        n_decisions = int(T / dt)
        if self.pms_visc_delay:
            held = build_release_table(n_decisions + 1, dt, VISC_SAMPLE_INTERVAL_H, VISC_ANALYSIS_DELAY_H)
            policy = _ZOHPolicyProxy(policy, held, [VISC_IDX])
        states = np.zeros((n_decisions + 1, STATE_DIM))
        inputs = np.zeros((n_decisions + 1, ACTION_DIM))
        mon = {a: [] for a in ("t", "PAA", "Viscosity", "Wt", "P", "Fs", "Fpaa", "discharge", "yield_per_run")}

        a_fs = 0.0
        action_norm = np.zeros(ACTION_DIM)
        decision_idx = 0
        last_good = np.zeros(STATE_DIM)

        for k in range(1, NUM_STEPS + 1):
            v = self._recipe.get_values_dict_at(time=k * STEP_IN_HOURS)
            if feed_delay_h:
                # Second lookup, feed_delay_h earlier, and only FEED_CHANNELS are taken from it.
                # Guarded by the truthiness test so feed_delay_h=0.0 keeps the original single
                # -lookup path byte-for-byte rather than paying for a redundant lookup.
                vd = self._recipe.get_values_dict_at(time=k * STEP_IN_HOURS - feed_delay_h)
                v = {**v, **{c: vd[c] for c in FEED_CHANNELS}}

            fpaa_k = v[PAA]
            discharge_k = v[DISCHARGE]

            if k <= k_warm:
                fs_k = v[FS]
            else:
                local = k - k_warm - 1
                if not pid_baseline and local % spd == 0 and decision_idx < n_decisions:

                    if decision_idx == 0:
                        states[0] = np.clip(np.nan_to_num(
                            extract_state(bx, k_warm, self.use_offline_measurements)), -1.0, 1.0)
                        last_good = states[0]

                    raw = policy(states[decision_idx], decision_idx)
                    action_norm = np.clip(np.asarray(raw, dtype=float).ravel(), -1.0, 1.0)
                    a_fs = float(action_norm[0])
                    inputs[decision_idx] = action_norm

                fs_k = v[FS] if pid_baseline else fs_from_action(
                    a_fs, v[FS], self.action_mode, self.fs_abs_min, self.fs_abs_max)

            env.bypass_paa_pid = False

            _, bx, yield_per_run, done = env.step(
                k, bx, Fs=fs_k, Foil=v[FOIL], Fg=v[FG], pressure=v[PRES],
                discharge=discharge_k, Fw=v[WATER], Fpaa=fpaa_k,
            )

            i = k - 1
            mon["t"].append(k * STEP_IN_HOURS)
            mon["PAA"].append(_read(bx, "PAA", i))
            mon["Viscosity"].append(_read(bx, "Viscosity", i))
            mon["Wt"].append(_read(bx, "Wt", i))
            mon["P"].append(_read(bx, "P", i))
            mon["Fs"].append(fs_k)
            mon["Fpaa"].append(fpaa_k)
            mon["discharge"].append(discharge_k)
            mon["yield_per_run"].append(yield_per_run)

            if k > k_warm and (k - k_warm - 1) % spd == spd - 1:
                decision_idx += 1
                if decision_idx <= n_decisions:
                    s = extract_state(bx, k, self.use_offline_measurements)
                    s = np.clip(np.where(np.isfinite(s), s, last_good), -1.0, 1.0)
                    last_good = s
                    states[decision_idx] = s
                    inputs[decision_idx] = action_norm

        self.monitor.append({a: np.array(mon[a]) for a in mon})
        self._episode += 1

        # `noise` (std_meas_noise from config) previously had no effect: this always returned
        # `states.copy()` as the "noiseless" slot, meaning the GP was ALWAYS trained on the noiseless
        # trajectory and `noiseless_states_history` was a pure duplicate. Callers that feed the GP
        # (get_data_from_system, setup_recipe_anchors, setup_high_feed_probes) use the FIRST return
        # slot as training data, so measurement noise has to be injected there, with the true
        # simulator output kept in the second slot. Re-clip to [-1, 1] afterward: the GP is only ever
        # trained on [-1, 1] data (see Model_learning_RBF_det_time.state_clamp), so that invariant
        # must hold post-noise too.
        if noise is not None:
            noisy_states = states + np.random.normal(scale=noise, size=states.shape)
            noisy_states = np.clip(noisy_states, -1.0, 1.0)
        else:
            noisy_states = states
        return noisy_states, inputs, states.copy()


class PenSimMCPILCO(MCP.MC_PILCO):

    def __init__(self, pensim_wrapper, optim_horizon_steps=None, **kwargs):

        kwargs.setdefault("f_sim", lambda x, t, u: x)
        super().__init__(**kwargs)

        self.system = pensim_wrapper

        self.optim_horizon_steps = optim_horizon_steps
        self._anchor_states = None
        self._anchor_vars = None

    # n_seg segments split the ~230h CONTROL_H as evenly as possible (remainder decisions spread
    # one-extra-each across the FIRST segments via divmod, not all dumped into the last one --
    # that used to leave the last shuffled level held for a fraction of the others' duration: 5h
    # instead of 20h at T_SAMPLING=5, 8h instead of 20h at T_SAMPLING=2, verified). Works
    # automatically for any T_SAMPLING, no per-value tuning needed.
    #
    # Levels are drawn INDEPENDENTLY per segment (not a fixed set of n_seg values permuted across
    # segments), so a given magnitude can recur in both the growth and production halves of the
    # batch across different exploration episodes. The previous linspace+shuffle scheme made
    # level and time-position a strict bijection per episode -- a Monte Carlo check showed that
    # left only a ~47% chance every level got tried in BOTH halves across 5 exploration episodes.
    #
    # The name is residual-mode history: this emits levels in [-1, 1] and lets rollout()'s
    # action_mode decide what they mean, so it needs no change for "absolute" -- there a uniform
    # draw over [-1, 1] IS a uniform draw over [FS_ABS_MIN, FS_ABS_MAX] held for ~19h, which is
    # the intended free-feed exploration. Be aware that half that band overfeeds hard (sustained
    # 200 L/h adds ~61,000 kg over the batch against WT_OVERFLOW = 1.2e5) and the bottom end is
    # outright starvation, so a large share of exploration batches will collapse -- deliberate
    # here (the GP needs to see what "too much" does), and nothing screens them out since
    # FAILED_YIELD_KG / MAX_EXPLORATION_RETRIES are currently disabled above.
    def _recipe_exploration_policy(self, n_seg=12):
        n_decisions = int(CONTROL_H / T_SAMPLING)  # matches PenSimWrapper.rollout's own n_decisions
        base, extra = divmod(n_decisions, n_seg)
        seg_lens = [base + 1 if i < extra else base for i in range(n_seg)]
        boundaries = np.cumsum([0] + seg_lens)
        levels = np.random.uniform(-1.0, 1.0, n_seg)
        def pol(state, decision_idx):
            s = min(int(np.searchsorted(boundaries, decision_idx, side="right")) - 1, n_seg - 1)
            return np.array([levels[s]])
        return pol

    def get_data_from_system(self, initial_state, T_exploration,
                             trial_index, flg_exploration=False):
        # EXPLORATION batches are screened: one whose yield lands below FAILED_YIELD_KG has collapsed,
        # and feeding it to the GPs poisons the initial model. Re-draw the exploration policy and roll
        # again until a batch clears the bar, so the `num_explorations` batches that seed the model are
        # all non-failed. Only exploration is screened -- POLICY batches are always kept however they
        # turn out, or the learning loop would be blind to its own failures.
        # for attempt in range(1, MAX_EXPLORATION_RETRIES + 1):
        if flg_exploration:
            np_policy = self._recipe_exploration_policy()
        else:
            np_policy = self.control_policy.get_np_policy()

        # No `seed` here on purpose: rollout then draws its realisation from
        # `_episode + seed_offset`, so every episode gets fresh initial conditions and batch
        # parameters (alpha_kla, PAA_c, N_conc_paa). A fixed realisation made successive
        # on-policy batches near-duplicates once the policy converged -- the GP gained no new
        # information from them while its noise term kept shrinking. seed_offset = seed * 1000
        # (config_single_phase) keeps different config seeds from colliding.
        states, inputs, noiseless = self.system.rollout(
            initial_state, np_policy, T_exploration, self.T_sampling,
            self.std_meas_noise
        )

        self.state_samples_history.append(states)
        self.input_samples_history.append(inputs)
        self.noiseless_states_history.append(noiseless)
        self.num_data_collection += 1
        self.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)

    def setup_recipe_anchors(self, num_batches=2, num_anchors=12, anchor_var=0.01):
        """Build the fixed anchor set for multi-origin short rollouts. Call ONCE before reinforce().

        Rolls `num_batches` PURE-RECIPE (a=0) batches on a FRESH wrapper -- so the training run's own
        episode/seed sequence is untouched (clean A/B vs a no-anchor run) -- adds them to the GP
        training set (so every anchor is a state the GP has data around), then subsamples `num_anchors`
        real states spread across batch time as the launch points p(x0) for policy optimisation.
        """
        seed_offset = self.system.seed_offset
        anchor_policy = lambda state, decision_idx: np.array([0.0])
        # action_mode must MATCH self.system's: these trajectories go straight into the GP
        # training set, so a fresh wrapper left on the default "residual" would record a = 0
        # against recipe-fed states while an "absolute" agent reads that same column as
        # mid-range feed -- two action encodings in one training set. (No-op for residual runs,
        # where the default already matches.) NOTE that under "absolute", a = 0 is mid-range
        # feed, so these stop being PURE-RECIPE anchors and become constant-mid-feed ones.
        fresh = PenSimWrapper(seed_offset=seed_offset, action_mode=self.system.action_mode,
                              fs_abs_min=self.system.fs_abs_min,
                              fs_abs_max=self.system.fs_abs_max)

        np_state = np.random.get_state()
        batch_states = []
        for i in range(num_batches):
            states, inputs, _ = fresh.rollout(
                s0=initial_state_norm(), policy=anchor_policy,
                T=CONTROL_H, dt=self.T_sampling, noise=self.std_meas_noise,
                seed=seed_offset + i,
            )
            self.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)
            batch_states.append(states)
        np.random.set_state(np_state)

        T = batch_states[0].shape[0]
        time_idx = np.linspace(0, T - 1, num_anchors).round().astype(int)
        anchors = np.stack([batch_states[j % num_batches][ti]
                            for j, ti in enumerate(time_idx)])
        self._anchor_states = torch.tensor(anchors, dtype=self.dtype, device=self.device)
        self._anchor_vars = torch.full((num_anchors, STATE_DIM), float(anchor_var),
                                       dtype=self.dtype, device=self.device)
        # CHANGED_THIS added
        self._anchor_vars[:, TIME_IDX] = TIME_INIT_VAR

        print(f"[anchors] {num_anchors} launch states from {num_batches} recipe batches "
              f"(+{num_batches} into GP training set); optim_horizon_steps={self.optim_horizon_steps}")
        return self._anchor_states

    def setup_high_feed_probes(self, num_probes=4, levels=(0.6, 0.8, 1.0)):
        """Roll `num_probes` FIXED, TIME-VARYING feed-rate batches on a FRESH wrapper and add them
        straight to the GP training set, ON TOP OF the `num_explorations` exploration batches
        reinforce() rolls itself. Call ONCE before reinforce() (independent of, and combinable
        with, setup_recipe_anchors).

        Why this exists: the X and Viscosity GPs learn an action-lengthscale of 11.6-39.7 on the
        [-1, 1] action input in every reward-shaping config tried so far (see evaluations plan,
        Finding 5) -- i.e. the model has decided feed rate barely affects biomass growth or
        viscosity at all -- and the late-phase feed->P response comes out with the WRONG SIGN.
        Both are data problems, not model problems: `_recipe_exploration_policy` re-draws its level
        every ~19h, so it never HOLDS feed off-recipe long enough for a response that accumulates
        over the production window to show up at all. Averaged over a batch, its excitation looks
        like noise around the recipe, and the GP fits it as such.

        The probe set (PROBE_PLAN) attacks that directly: two balanced dithers at different
        timescales to separate the fast response from the slow one, then a sustained +/- step pair
        confined to the production window to observe feed->P on both sides of the sign reversal.
        See PROBE_PLAN for the per-probe rationale.

        Probes are deliberately unscreened -- collapsed batches are kept, since a batch that
        collapses under sustained overfeed is precisely the observation the GP is missing.
        """
        seed_offset = self.system.seed_offset
        # Same encoding-consistency requirement as setup_recipe_anchors -- see the comment there.
        # Under "absolute" the PROBE_PLAN shapes are no longer +/- excursions around the recipe
        # but around mid-range feed (and prod_step_down at level 1.0 is total starvation, Fs = 0).
        fresh = PenSimWrapper(seed_offset=seed_offset, action_mode=self.system.action_mode,
                              fs_abs_min=self.system.fs_abs_min,
                              fs_abs_max=self.system.fs_abs_max)
        n_decisions = int(CONTROL_H / self.T_sampling)

        np_state = np.random.get_state()
        used = []
        for i in range(num_probes):
            shape_name, level_idx = PROBE_PLAN[i % len(PROBE_PLAN)]
            level = levels[level_idx % len(levels)]
            profile = PROBE_SHAPES[shape_name](level, n_decisions)
            used.append((shape_name, level))
            probe_policy = lambda state, decision_idx, profile=profile: np.array(
                [profile[min(int(decision_idx), len(profile) - 1)]])
            # +900 keeps these seeds inside THIS seed_offset's own 1000-wide block (see
            # config_single_phase.wrapper_par), clear of both setup_recipe_anchors' seed_offset+i
            # range and the next seed's seed_offset.
            states, inputs, _ = fresh.rollout(
                s0=initial_state_norm(), policy=probe_policy,
                T=CONTROL_H, dt=self.T_sampling, noise=self.std_meas_noise,
                seed=seed_offset + 900 + i,
            )
            self.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)
        np.random.set_state(np_state)

        print(f"[high-feed probes] added {num_probes} time-varying batches {used} "
              f"to the GP training set")

    def reinforce_policy(self, *args, **kwargs):
        if self._anchor_states is not None:
            kwargs["particles_initial_state_mean"] = self._anchor_states
            kwargs["particles_initial_state_var"] = self._anchor_vars
            kwargs["flg_particles_init_multi_gauss"] = True
            kwargs["flg_particles_init_uniform"] = False
        return super().reinforce_policy(*args, **kwargs)

    def apply_policy(self, *args, **kwargs):
        if self.optim_horizon_steps is not None and "T_control" in kwargs:
            kwargs["T_control"] = min(int(kwargs["T_control"]), int(self.optim_horizon_steps))
        return super().apply_policy(*args, **kwargs)


class PenSimMCPILCODelayed(PenSimMCPILCO):
    """MC-PILCO4PMS-style split for Viscosity only: GP dynamics + cost still see the TRUE
    particle trajectory (states_sequence_list, per base MC_PILCO.apply_policy's contract,
    unchanged in MC_PILCO4PMS too -- MC-PILCO/policy_learning/MC_PILCO.py:906); only what
    control_policy is CALLED WITH has Viscosity zero-order-held from the most recently
    released 12h/4h lab-assay sample -- mirroring PenSimWrapper.rollout's real-system
    treatment (pms_visc_delay) so imagined rollouts match what the deployed policy actually
    observes.

    Swaps self.control_policy for a _ZOHPolicyProxy for the duration of a single
    apply_policy() call via object.__setattr__ (control_policy is a registered nn.Module
    submodule, so plain assignment would raise TypeError), then restores it. Safe because
    apply_policy touches self.control_policy only via __call__ (MC_PILCO.py:660/671), and
    the swap window never spans reinforce_policy's other self.control_policy accesses
    (.parameters()/.reinit(), MC_PILCO.py:454/468/558/577/605) -- those happen before/after
    this call, never during it.

    Gated on self.system.pms_visc_delay (the SAME flag PenSimWrapper.rollout checks for the
    real-system side) so this class is a single, permanently-safe on/off switch: passing
    pms_visc_delay=False into get_config()/PenSimWrapper turns Viscosity fully back online in
    BOTH the real and imagined rollouts without touching which agent class is instantiated --
    no need to fall back to PenSimMCPILCO/PenSimMCPILCOMultiPhase.
    """

    def apply_policy(self, *args, **kwargs):
        if not self.system.pms_visc_delay:
            return super().apply_policy(*args, **kwargs)
        held = build_release_table(int(kwargs["T_control"]), self.T_sampling,
                                    VISC_SAMPLE_INTERVAL_H, VISC_ANALYSIS_DELAY_H)
        real_policy = self.control_policy
        object.__setattr__(self, "control_policy", _ZOHPolicyProxy(real_policy, held, [VISC_IDX]))
        try:
            return super().apply_policy(*args, **kwargs)
        finally:
            object.__setattr__(self, "control_policy", real_policy)


class PenSimMCPILCOMultiPhase(PenSimMCPILCO):
    """Dual-phase variant: self.model_learning is a DualPhaseModelLearning (see
    model_learning_dual_phase.py) that BLENDS phase-1/phase-2 GP predictions via a sigmoid
    weight centered on PIVOT_HOURS, based on a decision-step counter. That counter must be
    reset to 0 at the start of every rollout (apply_policy during policy optimisation, and the
    diagnostic rollout() in MC_PILCO) -- both call get_next_state sequentially in decision
    order, so resetting once up front and letting the wrapper self-increment is enough to keep
    it aligned with absolute decision time.

    setup_recipe_anchors/optim_horizon_steps are unsupported here: they launch particles
    from arbitrary/relative batch times, which the phase router (keyed on ROLLOUT-RELATIVE
    step, assumed == absolute decision index) cannot interpret correctly -- an anchor state
    plucked from, say, hour 150 of a recipe batch would be handed to reinforce_policy as a
    particle-init point, then rolled out starting from step 0, so the router would treat
    hour-150 physics as if they were hour-0.

    setup_high_feed_probes IS supported (unlike the two above): it only ever calls
    model_learning.add_data() with FULL from-t=0 trajectories (see PenSimMCPILCO's own
    implementation, which this class inherits unchanged), and DualPhaseModelLearning.add_data
    splits purely on ABSOLUTE array index at pivot_step -- no rollout-relative step counter
    involved -- so a probe batch is routed to phase1/phase2 exactly like any other full
    training episode.
    """

    def apply_policy(self, *args, **kwargs):
        assert self.optim_horizon_steps is None, (
            "optim_horizon_steps is not supported with PenSimMCPILCOMultiPhase")
        self.model_learning.reset_step_counter()
        return super().apply_policy(*args, **kwargs)

    def rollout(self, *args, **kwargs):
        self.model_learning.reset_step_counter()
        return super().rollout(*args, **kwargs)

    def setup_recipe_anchors(self, *args, **kwargs):
        raise NotImplementedError(
            "setup_recipe_anchors is not supported with PenSimMCPILCOMultiPhase "
            "(anchor launch states don't carry the absolute decision time the phase "
            "router needs)")


class PenSimMCPILCOMultiPhaseDelayed(PenSimMCPILCOMultiPhase):
    """Dual-phase counterpart of PenSimMCPILCODelayed: same MC-PILCO4PMS-style Viscosity
    split (see that class's docstring), composed on top of PenSimMCPILCOMultiPhase rather
    than duplicated ad hoc. super().apply_policy(*args, **kwargs) here resolves to
    PenSimMCPILCOMultiPhase.apply_policy (reset_step_counter + optim_horizon_steps assert),
    which itself chains to PenSimMCPILCO.apply_policy (T_control clamp) and then the base
    MC_PILCO.apply_policy loop -- so the proxy swap composes correctly regardless of how
    many layers of apply_policy overrides sit in between.

    Gated on self.system.pms_visc_delay, same as PenSimMCPILCODelayed -- see that class's
    docstring.
    """

    def apply_policy(self, *args, **kwargs):
        if not self.system.pms_visc_delay:
            return super().apply_policy(*args, **kwargs)
        held = build_release_table(int(kwargs["T_control"]), self.T_sampling,
                                    VISC_SAMPLE_INTERVAL_H, VISC_ANALYSIS_DELAY_H)
        real_policy = self.control_policy
        object.__setattr__(self, "control_policy", _ZOHPolicyProxy(real_policy, held, [VISC_IDX]))
        try:
            return super().apply_policy(*args, **kwargs)
        finally:
            object.__setattr__(self, "control_policy", real_policy)

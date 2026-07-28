
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

# STATE_LOG_CHANNELS = {"S", "Wt", "X", "P"}
STATE_LOG_CHANNELS = {"Wt", "X", "P"}
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
    # "S":         (np.log(STATE_LOG_FLOOR), np.log(20.0)),
    # "S":         (0, 5),
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


def _read(batch_x, name, i):
    """Physical value of `name` at native index i. pH is stored as 10^(-pH)
    mid-batch, so invert it back to pH units here. `time` is not a batch channel;
    native index i maps to absolute batch time (i+1)*STEP_IN_HOURS hours."""
    if name == "time":
        return (i + 1) * STEP_IN_HOURS
    if name == "pH":
        return -np.log10(max(float(getattr(batch_x, "pH").y[i]), 1e-12))
    return float(getattr(batch_x, name).y[i])


def extract_state(batch_x, k):
    i = max(k - 1, 0)
    return np.array([
        _normalise(encode_state_value(n, _read(batch_x, n, i)), *STATE_RANGES[n])
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


PROBE_BLOCK_HOURS = 40.0


def _ramp_profile(level, n):
    """0 -> level -> 0 triangle over the batch: the action changes almost every decision."""
    half = n // 2
    up = np.linspace(0.0, level, half, endpoint=False)
    down = np.linspace(level, 0.0, n - half)
    return np.concatenate([up, down])


def _step_high_low_profile(level, n):
    """One switch: sustained +level for the first half, sustained -level for the second."""
    half = n // 2
    return np.concatenate([np.full(half, level), np.full(n - half, -level)])


def _alternating_blocks_profile(level, n, block_hours=PROBE_BLOCK_HOURS):
    """+level/-level square wave in `block_hours`-wide blocks: repeated switches spread through
    the whole batch, rather than _step_high_low_profile's single one."""
    block_decisions = max(1, int(round(block_hours / T_SAMPLING)))
    block = (np.arange(n) // block_decisions) % 2
    return np.where(block == 0, level, -level)


# Cycled round-robin across probes (see setup_high_feed_probes): each shape sustains high/low feed
# differently, giving the GP training set a range of sustained-feed patterns rather than just one.
PROBE_SHAPES = {
    "ramp": _ramp_profile,
    "step_high_low": _step_high_low_profile,
    "alternating_blocks": _alternating_blocks_profile,
}


class PenSimWrapper:
    """One PenSimPy batch: recipe warmup -> PAA-increment RL control, as MC-PILCO arrays."""

    def __init__(self, seed_offset=0):
        self.seed_offset = seed_offset
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

    def rollout(self, s0, policy, T, dt, noise, seed=None, pid_baseline=False):
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
        states = np.zeros((n_decisions + 1, STATE_DIM))
        inputs = np.zeros((n_decisions + 1, ACTION_DIM))
        mon = {a: [] for a in ("t", "PAA", "Viscosity", "Wt", "P", "Fs", "Fpaa", "discharge", "yield_per_run")}

        a_fs = 0.0
        action_norm = np.zeros(ACTION_DIM)
        decision_idx = 0
        last_good = np.zeros(STATE_DIM)

        for k in range(1, NUM_STEPS + 1):
            v = self._recipe.get_values_dict_at(time=k * STEP_IN_HOURS)

            fpaa_k = v[PAA]
            discharge_k = v[DISCHARGE]

            if k <= k_warm:
                fs_k = v[FS]
            else:
                local = k - k_warm - 1
                if not pid_baseline and local % spd == 0 and decision_idx < n_decisions:

                    if decision_idx == 0:
                        states[0] = np.clip(np.nan_to_num(extract_state(bx, k_warm)), -1.0, 1.0)
                        last_good = states[0]

                    raw = policy(states[decision_idx], decision_idx)
                    action_norm = np.clip(np.asarray(raw, dtype=float).ravel(), -1.0, 1.0)
                    a_fs = float(action_norm[0])
                    inputs[decision_idx] = action_norm

                fs_k = v[FS] if pid_baseline else v[FS] * (1.0 + FS_SCALE * a_fs)

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
                    s = extract_state(bx, k)
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

    # n_seg is distinct feed levels the episode uses, spread evenly across the [-1, 1] range.
    # This segments the action into n_seg segments, each with length 4 decisions.
    # for T_Sampling=5h, there are 46 decisions per batch.
    def _recipe_exploration_policy(self, seg_len=4, n_seg=12):
        levels = np.linspace(-1.0, 1.0, n_seg) + np.random.uniform(-0.08, 0.08, n_seg)
        levels = np.clip(levels, -1.0, 1.0)
        np.random.shuffle(levels)
        def pol(state, decision_idx):
            s = min(int(decision_idx) // seg_len, n_seg - 1)
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

        # if not flg_exploration:
        #     break
            # y = batch_yield_kg(self.system.monitor[-1])
            # if y >= FAILED_YIELD_KG:
            #     if attempt > 1:
            #         print(f"[exploration] accepted on attempt {attempt} (yield {y:.0f} kg)")
            #     break
            # if attempt == MAX_EXPLORATION_RETRIES:
            #     print(f"[exploration] WARNING: no batch cleared {FAILED_YIELD_KG:.0f} kg in "
            #         f"{MAX_EXPLORATION_RETRIES} attempts; keeping the last (yield {y:.0f} kg)")
            #     break
            # Rejected -> drop its monitor entry too, so `system.monitor` stays index-aligned with
            # the kept *_samples_history (diagnostics pair the two by episode index).
            # self.system.monitor.pop()
            # print(f"[exploration] rejected batch (yield {y:.0f} kg < {FAILED_YIELD_KG:.0f} kg), "
            #     f"attempt {attempt}/{MAX_EXPLORATION_RETRIES}; re-rolling")

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
        fresh = PenSimWrapper(seed_offset=seed_offset)

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

    def setup_high_feed_probes(self, num_probes=3, levels=(0.6, 0.8, 1.0)):
        """Roll `num_probes` FIXED, TIME-VARYING feed-rate batches on a FRESH wrapper and add them
        straight to the GP training set. Call ONCE before reinforce() (independent of, and
        combinable with, setup_recipe_anchors).

        Why this exists: the X and Viscosity GPs learn an action-lengthscale of 11.6-39.7 on the
        [-1, 1] action input in every reward-shaping config tried so far (see evaluations plan,
        Finding 5) -- i.e. the model has decided feed rate barely affects biomass growth or
        viscosity at all. That is plausibly because ordinary exploration
        (`_recipe_exploration_policy`) rarely SUSTAINS a high feed level long enough to reach the
        viscosity-collapse regime, and any exploration batch that does collapse is rejected and
        re-rolled by `get_data_from_system`'s FAILED_YIELD_KG screen -- so the GP training set is
        structurally starved of exactly the data that would teach it the action matters there.

        These probes are deliberately NOT screened by that yield threshold: the point is to give the
        GP real (state, high-action, viscosity-response) trajectories, collapse included, not to
        curate a "safe" dataset the way exploration episodes do.
        """
        seed_offset = self.system.seed_offset
        fresh = PenSimWrapper(seed_offset=seed_offset)
        n_decisions = int(CONTROL_H / self.T_sampling)
        shape_names = list(PROBE_SHAPES.keys())

        np_state = np.random.get_state()
        used = []
        for i in range(num_probes):
            level = levels[i % len(levels)]
            shape_name = shape_names[i % len(shape_names)]
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


class PenSimMCPILCOMultiPhase(PenSimMCPILCO):
    """Dual-phase variant: self.model_learning is a DualPhaseModelLearning (see
    model_learning_dual_phase.py) that BLENDS phase-1/phase-2 GP predictions via a sigmoid
    weight centered on PIVOT_HOURS, based on a decision-step counter. That counter must be
    reset to 0 at the start of every rollout (apply_policy during policy optimisation, and the
    diagnostic rollout() in MC_PILCO) -- both call get_next_state sequentially in decision
    order, so resetting once up front and letting the wrapper self-increment is enough to keep
    it aligned with absolute decision time.

    setup_recipe_anchors/setup_high_feed_probes/optim_horizon_steps are unsupported here:
    they launch particles from arbitrary/relative batch times, which the phase router (keyed
    on ROLLOUT-RELATIVE step, assumed == absolute decision index) cannot interpret correctly.
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

    def setup_high_feed_probes(self, *args, **kwargs):
        raise NotImplementedError(
            "setup_high_feed_probes is not supported with PenSimMCPILCOMultiPhase")

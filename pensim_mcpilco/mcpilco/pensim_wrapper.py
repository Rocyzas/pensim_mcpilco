
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
STATE_NAMES = ["Wt", "X", "P", "Viscosity", "time"]
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 1

# STATE_LOG_CHANNELS = {"S", "Wt", "X", "P"}
STATE_LOG_CHANNELS = {"Wt", "X", "P"}
STATE_LOG_FLOOR = 1e-6

WARMUP_H = 0
T_SAMPLING = 5.0
STEPS_PER_DECISION = int(round(T_SAMPLING / STEP_IN_HOURS))

K_WARM = max(1, int(round(WARMUP_H / STEP_IN_HOURS)))
WARMUP_H_EFF = K_WARM * STEP_IN_HOURS
CONTROL_H = 230.0 - WARMUP_H_EFF

FPAA_MIN, FPAA_MAX = 0.0, 15.0

# 1 = 100%
# making FS 50% because the initial explorations are 10%, so policy going outside the exploration is risky,
# as there is no data behind them.
# Also, it would be sensible to compare it with the BO baselines in this way.
FS_SCALE = 0.5


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
    "X":         (np.log(STATE_LOG_FLOOR), np.log(40.0)),
    "P":         (np.log(STATE_LOG_FLOOR), np.log(40.0)),
    # Measured over 619 logged batches: min 4.1, median 49.5, p99 142.6, max 188.8 cP. Linear, NOT
    # log-encoded -- the span is only ~46x (so a single RBF lengthscale copes), the decision-relevant
    # region is the 100-190 top end where linear gives the better resolution, and staying off the
    # exp() decode path keeps this channel out of the rollout blow-up mode that {Wt,X,P} needed
    # clamping for. Upper bound 200 leaves headroom above the worst observed batch.
    "Viscosity": (0.0,   200.0),
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


WT_SOFT = (7.0e4, 1.1e5)
WT_OVERFLOW = 1.2e5
P_CRASH = 55.0
PAA_BAND = (800.0, 1600.0)
VISC_MAX = 100.0

# Exploration screening: a batch whose total penicillin yield lands below FAILED_YIELD_KG has
# collapsed. Such batches are discarded and re-rolled rather than fed to the GPs, so the initial
# model is not built on failed batches (see PenSimMCPILCO.get_data_from_system).
FAILED_YIELD_KG = 2000.0
MAX_EXPLORATION_RETRIES = 20 


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


def _measure_init_state_norm(num_batches=6):
    """Normalised handover state (x0), MEASURED by rolling pure-recipe batches up to K_WARM.

    Derived from the live WARMUP_H / STATE_RANGES so it can never go stale. (INIT_STATE_PHYS below is
    a hard-coded snapshot taken for WARMUP_H=120; it silently became wrong the moment WARMUP_H changed,
    which made policy optimisation launch its imagined rollouts from a fully-grown reactor while the
    real batch started at inoculation.)

    NOT CACHED ON DISK, on purpose. A previous version wrote results/_init_state_cache/x0_kwarm{K}.npy
    keyed only by K_WARM -- so editing STATE_RANGES (x0 is stored NORMALISED) silently reused a stale
    x0. Nothing in the training path may depend on anything under results/: those artefacts carry the
    STATE_RANGES / WARMUP_H / code version of whenever they happened to be written. Measuring fresh
    costs `num_batches` recipe rollouts once per process, memoised below.
    """
    if K_WARM in _init_state_cache:
        return _init_state_cache[K_WARM]

    recipe_policy = lambda state, decision_idx: np.array([0.0])
    fresh = PenSimWrapper(seed_offset=0)
    np_state = np.random.get_state()
    x0 = np.mean([fresh.rollout(s0=None, policy=recipe_policy, T=CONTROL_H, dt=T_SAMPLING,
                                noise=None, seed=i)[0][0] for i in range(num_batches)], axis=0)
    np.random.set_state(np_state)

    _init_state_cache[K_WARM] = x0
    return x0


def initial_state_norm():
    """Normalised state at the WARMUP_H turn-on point (RL phase x0), measured from the recipe."""
    return _measure_init_state_norm()


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
        return states, inputs, states.copy()


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
        for attempt in range(1, MAX_EXPLORATION_RETRIES + 1):
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

            if not flg_exploration:
                break
            y = batch_yield_kg(self.system.monitor[-1])
            if y >= FAILED_YIELD_KG:
                if attempt > 1:
                    print(f"[exploration] accepted on attempt {attempt} (yield {y:.0f} kg)")
                break
            if attempt == MAX_EXPLORATION_RETRIES:
                print(f"[exploration] WARNING: no batch cleared {FAILED_YIELD_KG:.0f} kg in "
                      f"{MAX_EXPLORATION_RETRIES} attempts; keeping the last (yield {y:.0f} kg)")
                break
            # Rejected -> drop its monitor entry too, so `system.monitor` stays index-aligned with
            # the kept *_samples_history (diagnostics pair the two by episode index).
            self.system.monitor.pop()
            print(f"[exploration] rejected batch (yield {y:.0f} kg < {FAILED_YIELD_KG:.0f} kg), "
                  f"attempt {attempt}/{MAX_EXPLORATION_RETRIES}; re-rolling")

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

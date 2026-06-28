"""
Single-phase MC-PILCO baseline wrapper

Design decisions (baseline):
    1. Action  : PAA-flow setpoint; 
        the recipe drives every other channel
    2. Action type : ABSOLUTE setpoint — the policy outputs the Fpaa level directly
        (action in [-1,1] -> [FPAA_MIN, FPAA_MAX]). This keeps the feed level in the GP's
        action input so PAA dynamics are Markovian; an earlier INCREMENT design hid the
        level from the GP and let the policy ratchet Fpaa -> 0.
    3. Action freq : every 2 h
    4. Observation : 6 online sensors [T, DO2, O2-offgas, CO2-offgas, pH, weight]
                    + penicillin P (P is the reward - must be modelled state)
    5. Monitor PAA conc + viscosity
    6. Episode : default recipe runs first WARMUP_H hours open-loop; 
                the RL agent takes over after that (100h+)
    7. Reward  : penicillin concentration.
    8. Constraints : hard fixed penalty for vessel overflow / blow-up; 
                    soft proportional penalty otherwise
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MC-PILCO"))

import policy_learning.MC_PILCO as MCP

import numpy as np

from utils.peni_env_setup import PenSimEnv
from utils.recipe import Recipe, RecipeCombo
from utils.constants import NUM_STEPS, STEP_IN_HOURS
from PenSimPy.pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)

# LSODA integrator; must run before any PenSimEnv.step()
from utils.ode_patch import patch_fastodeint
patch_fastodeint()

# ADDING TIME
# Without time, one value at different points in the batch results in opposite outcomes
STATE_NAMES = ["T", "DO2", "O2", "CO2outgas", "pH", "Wt", "PAA", "P", "Culture_age"]
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 1

# WARMUP_H = 100.0 # recipe-only warmup; agent acts after
WARMUP_H = 0.2 # recipe-only warmup; agent acts after
T_SAMPLING = 2.0 # h between actions
STEPS_PER_DECISION = int(round(T_SAMPLING / STEP_IN_HOURS))
CONTROL_H = 230.0 - WARMUP_H
# PAA feed-rate setpoint bounds (L/h): action in [-1,1] maps linearly onto this range.
# Upper bound kept near the PID's working range (~4-12 L/h); the fpaa_sweep showed a
# sustained feed >~10 L/h accumulates PAA to toxic levels, so the top of the range is a
# soft-bad region the policy is expected to learn to avoid.
FPAA_MIN, FPAA_MAX = 0.0, 15.0

# physical (min, max) per state for [-1,1] squashing (default batches + headroom)
# From the IndPenSim analysis
STATE_RANGES = {
    # "T":         (296.0, 302.0),
    # "DO2":       (0.0,   25.0),
    # "O2":        (0.15,  0.25),
    # "CO2outgas": (0.0,   4.0),
    # "pH":        (5.5,   7.5),
    # "Wt":        (5.0e4, 1.3e5),

# MA Thesis adjusted
    "T":         (296.0, 302.0),
    "DO2":       (0.0,   30.0),
    "O2":        (0.0,   100),
    "CO2outgas": (0.0,   100),
    "pH":        (0.0,   14),
    "Wt":        (0,     111000),
    "PAA":       (600,   1800.0),
    "P":         (0.0,   40.0),
    "Culture_age": (0.0,   230.0),
}
# warmed-up physical state at t=WARMUP_H (default recipe);
# PAA conc is held ~1200 mg/L by the recipe PID through warmup.
# INIT_STATE_PHYS = {"T": 298.0, "DO2": 15.1, "O2": 0.19, "CO2outgas": 1.67,
                #    "pH": 6.51, "Wt": 1.014e5, "PAA": 1200.0, "P": 13.04}
INIT_STATE_PHYS = {"T": 297.65, "DO2": 14.74, "O2": 0.22, "CO2outgas": 0.09,
                   "pH": 6.44, "Wt": 61980.0, "PAA": 1422.0, "P": 0.01,
                   "Culture_age": 0.2}


# constraint thresholds (cost uses Wt/P/PAA; viscosity is monitor-only)
WT_SOFT = (7.0e4, 1.1e5) 
WT_OVERFLOW = 1.2e5 
P_CRASH = 40.0 
PAA_BAND = (600, 1800.0)
VISC_MAX = 100.0 


def _normalise(value, lo, hi):
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _read(batch_x, name, i):
    """Physical value of `name` at native index i. pH is stored as 10^(-pH)
    mid-batch, so invert it back to pH units here."""
    if name == "pH":
        return -np.log10(max(float(getattr(batch_x, "pH").y[i]), 1e-12))
    return float(getattr(batch_x, name).y[i])


# Normalise state vector [-1;1]
def extract_state(batch_x, k):
    i = max(k - 1, 0)
    return np.array([_normalise(_read(batch_x, n, i), *STATE_RANGES[n]) for n in STATE_NAMES])


def initial_state_norm():
    """Normalised state at the WARMUP_H turn-on point (RL phase x0)."""
    return np.array([_normalise(INIT_STATE_PHYS[n], *STATE_RANGES[n]) for n in STATE_NAMES])


# ------------------ SYSTEM WRAPPER ------------------
class PenSimWrapper:
    """One PenSimPy batch: recipe warmup -> PAA-increment RL control, as MC-PILCO arrays."""

    def __init__(self, seed_offset=0):
        self.seed_offset = seed_offset
        self._episode = 0
        self._recipe = self._build_default_recipe()
        self.monitor = [] # per-episode trajectories

    @staticmethod
    def _build_default_recipe():
        return RecipeCombo(recipe_dict={
            FS: Recipe(FS_DEFAULT_PROFILE, FS), FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
            FG: Recipe(FG_DEFAULT_PROFILE, FG), PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
            DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
            WATER: Recipe(WATER_DEFAULT_PROFILE, WATER), PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
        })

    def rollout(self, s0, policy, T, dt, noise):
        env = PenSimEnv(recipe_combo=self._recipe, fast=True)
        env.random_seed_ref = self._episode + self.seed_offset
        _, bx = env.reset()

        spd = STEPS_PER_DECISION
        k_warm = int(round(WARMUP_H / STEP_IN_HOURS)) # 500
        n_decisions = int(T / dt) # 65
        states = np.zeros((n_decisions + 1, STATE_DIM))
        inputs = np.zeros((n_decisions + 1, ACTION_DIM))
        mon = {a: [] for a in ("t", "PAA", "Viscosity", "Wt", "P", "Fpaa")}

        # start at recipe value
        fpaa = float(self._recipe.recipe_dict[PAA].get_value_at(WARMUP_H))
        action_norm = np.zeros(ACTION_DIM)
        decision_idx = 0
        last_good = None

        for k in range(1, NUM_STEPS + 1):
            v = self._recipe.get_values_dict_at(time=k * STEP_IN_HOURS)

            if k <= k_warm:
                # warmup: recipe drives PAA
                fpaa_k = v[PAA]
            else:
                local = k - k_warm - 1
                if local % spd == 0 and decision_idx < n_decisions:

                    # first controlled state = state at WARMUP_H
                    if decision_idx == 0:
                        # states[0] - first controlled 8-dim vector
                        # bx - complete batch record (to call at specific time specvific value: bx.P.y[i])
                        states[0] = np.clip(np.nan_to_num(extract_state(bx, k_warm)), -1.0, 1.0)
                        last_good = states[0]
                    raw = policy(states[decision_idx], decision_idx)
                    action_norm = np.clip(np.asarray(raw, dtype=float).ravel(), -1.0, 1.0)
                    # absolute Fpaa setpoint: the action IS the feed level, so the GP's
                    # action input becomes the actual driver of PAA dynamics (Markovian).
                    # (Previously an increment `fpaa += action*FPAA_DELTA`, which hid the
                    #  feed level from the GP and let the policy ratchet Fpaa -> 0.)
                    fpaa = float(FPAA_MIN + (action_norm[0] + 1.0) * 0.5 * (FPAA_MAX - FPAA_MIN))
                    inputs[decision_idx] = action_norm

                # held over the 2 h window
                fpaa_k = fpaa

            # warmup: keep PenSim's PAA PID
            # control phase: bypass it so the agent's Fpaa is applied open-loop.
            env.bypass_paa_pid = k > k_warm

            _, bx, _, done = env.step(
                k, bx, Fs=v[FS], Foil=v[FOIL], Fg=v[FG], pressure=v[PRES],
                discharge=v[DISCHARGE], Fw=v[WATER], Fpaa=fpaa_k,
            )

            i = k - 1
            mon["t"].append(k * STEP_IN_HOURS)
            mon["PAA"].append(_read(bx, "PAA", i))
            mon["Viscosity"].append(_read(bx, "Viscosity", i))
            mon["Wt"].append(_read(bx, "Wt", i))
            mon["P"].append(_read(bx, "P", i))
            mon["Fpaa"].append(fpaa_k)

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

    def __init__(self, pensim_wrapper, **kwargs):
    
        # mcpilco engine wants an ODE fn
        kwargs.setdefault("f_sim", lambda x, t, u: x)
        super().__init__(**kwargs)

        # swap ODE -> PenSimPy
        self.system = pensim_wrapper

    def _recipe_exploration_policy(self, noise_std=0.25):
        """Exploration = perturbed recipe Fpaa schedule (low early, ramp up),
        mapped to normalised actions. Gives the GP at least some P>0 batches."""
        recipe = self.system._recipe
        def pol(state, decision_idx):
            t = WARMUP_H + decision_idx * T_SAMPLING          # decision time (h)
            fpaa = float(recipe.recipe_dict[PAA].get_value_at(t))
            a = 2.0 * (fpaa - FPAA_MIN) / (FPAA_MAX - FPAA_MIN) - 1.0   # Fpaa -> [-1,1]
            a = a + noise_std * np.random.randn()
            return np.clip(np.array([a]), -1.0, 1.0)
        return pol

    def get_data_from_system(self, initial_state, T_exploration,
                             trial_index, flg_exploration=False):
        # policy = self.rand_exploration_policy if flg_exploration else self.control_policy
        # states, inputs, noiseless = self.system.rollout(
        #     initial_state, policy.get_np_policy(), T_exploration, self.T_sampling,
        #     self.std_meas_noise,
        # )
        if flg_exploration:
            np_policy = self._recipe_exploration_policy() # perturbed recipe schedule
        else:
            np_policy = self.control_policy.get_np_policy()
        states, inputs, noiseless = self.system.rollout(
            initial_state, np_policy, T_exploration, self.T_sampling,
            self.std_meas_noise,
        )
        self.state_samples_history.append(states)
        self.input_samples_history.append(inputs)
        self.noiseless_states_history.append(noiseless)
        self.num_data_collection += 1
        self.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)


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
# Without time, one value at different points in the batch results in opposite outcomes.
# We feed absolute batch time (h) directly rather than culture age as its proxy.
STATE_NAMES = ["T", "DO2", "O2", "CO2outgas", "pH", "Wt", "PAA", "X", "P", "time"]
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 1

# WARMUP_H = 100.0 # recipe-only warmup; agent acts after
WARMUP_H = 0.2 # recipe-only warmup; agent acts after
T_SAMPLING = 2.0 # h between actions
STEPS_PER_DECISION = int(round(T_SAMPLING / STEP_IN_HOURS))
CONTROL_H = 230.0 - WARMUP_H

FPAA_MIN, FPAA_MAX = 0.0, 15.0
# RL now drives Fs (sugar) as a RESIDUAL on the recipe: Fs = recipe_Fs(t) * (1 + FS_SCALE * a),
# a in [-1,1] (neutral a=0 == recipe). Discharge + Fpaa revert to recipe/PID; only Fs varies.
FS_SCALE = 0.5           # +/-50% correction band around the recipe Fs schedule
DO2_FLOOR = 5.0          # mg/L; smooth penalty if aggressive Fs outruns aeration and crashes DO2


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
    "O2":        (0.15,  0.25),
    "CO2outgas": (0.0,   4.0),
    "pH":        (5.5,   7.5),
    "Wt":        (5.0e4, 1.3e5),
    "PAA":       (600,   1800.0),
    "X":         (0.0,   25.0),    # biomass g/L (recipe reaches ~20); ground-truth state, not an online sensor
    "P":         (0.0,   60.0),
    "time":      (0.0,   230.0),
}
# warmed-up physical state at t=WARMUP_H (default recipe);
# PAA conc is held ~1200 mg/L by the recipe PID through warmup.
# INIT_STATE_PHYS = {"T": 298.0, "DO2": 15.1, "O2": 0.19, "CO2outgas": 1.67,
                #    "pH": 6.51, "Wt": 1.014e5, "PAA": 1200.0, "P": 13.04}
INIT_STATE_PHYS = {"T": 297.65, "DO2": 14.74, "O2": 0.22, "CO2outgas": 0.09,
                   "pH": 6.44, "Wt": 61980.0, "PAA": 1422.0, "X": 0.49, "P": 0.01,
                   "time": WARMUP_H}


# constraint thresholds (cost uses Wt/P/PAA; viscosity is monitor-only)
WT_SOFT = (7.0e4, 1.1e5) 
WT_OVERFLOW = 1.2e5
# above the physical max (~35 g/L) so the hard penalty stops firing inside the
# productive regime; P range widened to (0,60) to keep it below the clip boundary.
P_CRASH = 55.0
PAA_BAND = (600, 1800.0)
VISC_MAX = 100.0 


def _normalise(value, lo, hi):
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _read(batch_x, name, i):
    """Physical value of `name` at native index i. pH is stored as 10^(-pH)
    mid-batch, so invert it back to pH units here. `time` is not a batch channel;
    native index i maps to absolute batch time (i+1)*STEP_IN_HOURS hours."""
    if name == "time":
        return (i + 1) * STEP_IN_HOURS
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

    def rollout(self, s0, policy, T, dt, noise, seed=None, pid_baseline=False):
        # noise: unused -- real rollouts observe the true sim state (noiseless);
        # only the model/particle rollout is stochastic (via GP delta_var).
        # seed: if given, drives the batch RNG directly (explicit/reproducible eval),
        #   overriding the _episode+seed_offset counter; also reseeds numpy's global RNG
        #   so the per-step Raman/PRBS noise is reproducible for this rollout.
        # pid_baseline: if True, run PAA under PenSim's built-in PID (never bypassed) and
        #   ignore `policy` -- the matched baseline arm for the RL-vs-PID comparison.
        env = PenSimEnv(recipe_combo=self._recipe, fast=True)
        if seed is not None:
            np.random.seed(seed)
            env.random_seed_ref = seed
        else:
            env.random_seed_ref = self._episode + self.seed_offset
        _, bx = env.reset()

        spd = STEPS_PER_DECISION
        k_warm = int(round(WARMUP_H / STEP_IN_HOURS))
        n_decisions = int(T / dt) # =114
        states = np.zeros((n_decisions + 1, STATE_DIM))
        inputs = np.zeros((n_decisions + 1, ACTION_DIM))
        #  omits T/DO2/O2/CO2/pH because nothing downstream plots or constrains them
        mon = {a: [] for a in ("t", "PAA", "Viscosity", "Wt", "P", "Fs", "Fpaa", "discharge", "yield_per_run")}

        # RL drives Fs as a residual on the recipe; neutral correction a=0 -> recipe Fs.
        # Discharge + Fpaa follow the recipe/PID (identical to the recipe baseline).
        a_fs = 0.0
        action_norm = np.zeros(ACTION_DIM)
        decision_idx = 0
        last_good = np.zeros(STATE_DIM)  # safe default (PID arm never enters the decision block)

        for k in range(1, NUM_STEPS + 1):
            v = self._recipe.get_values_dict_at(time=k * STEP_IN_HOURS)

            # Everything except Fs follows the recipe/PID baseline:
            # Fpaa (recipe early, PAA PID -> ~1200 mg/L after t>=10h) and discharge (recipe pulses).
            fpaa_k = v[PAA]
            discharge_k = v[DISCHARGE]

            if k <= k_warm:
                # warmup: recipe drives Fs too
                fs_k = v[FS]
            else:
                local = k - k_warm - 1
                if not pid_baseline and local % spd == 0 and decision_idx < n_decisions:

                    # first controlled state = state at WARMUP_H
                    if decision_idx == 0:
                        # states[0] - first controlled state vector
                        # bx - complete batch record (to call at specific time specvific value: bx.P.y[i])
                        states[0] = np.clip(np.nan_to_num(extract_state(bx, k_warm)), -1.0, 1.0)
                        last_good = states[0]
                    raw = policy(states[decision_idx], decision_idx)
                    action_norm = np.clip(np.asarray(raw, dtype=float).ravel(), -1.0, 1.0)
                    # residual-on-recipe: the action is a correction factor held over the 2 h
                    # window; a=0 -> recipe Fs, a=+/-1 -> +/-FS_SCALE around recipe. The recipe's
                    # per-step shape is preserved (scaled), so the policy only learns corrections.
                    a_fs = float(action_norm[0])
                    inputs[decision_idx] = action_norm

                # RL: recipe Fs scaled by the held correction. PID/recipe baseline: recipe Fs.
                fs_k = v[FS] if pid_baseline else v[FS] * (1.0 + FS_SCALE * a_fs)

            # PAA PID stays in control throughout (never bypassed) -- exactly like the recipe.
            env.bypass_paa_pid = False

            # 3rd return is yield_per_run; summed over the batch it equals PenSimPy's
            # batch_yield (the discharge-aware metric the recipe/BO baselines report).
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

    def __init__(self, pensim_wrapper, **kwargs):
    
        # mcpilco engine wants an ODE fn
        kwargs.setdefault("f_sim", lambda x, t, u: x)
        super().__init__(**kwargs)

        # swap ODE -> PenSimPy
        self.system = pensim_wrapper

    def _recipe_exploration_policy(self, noise_std=0.4):
        """Exploration = random Fs corrections around the recipe. Neutral action a=0 is the
        recipe itself, so Gaussian noise about 0 brackets the recipe (Fs +/- FS_SCALE*noise),
        giving the GP varied biomass/DO2/Wt trajectories without wandering far from baseline."""
        def pol(state, decision_idx):
            a = noise_std * np.random.randn()
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


from logging import config
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MC-PILCO"))

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import model_learning.Model_learning as ML
import policy_learning.Policy as Policy

from mcpilco.penicillin_cost import PeniConcentrationCost, PeniConcentrationDenseCost, PeniMassChangeCost

# CHANGED_THIS
from mcpilco.model_learning_det_time import Model_learning_RBF_det_time
from mcpilco.pensim_wrapper import (STATE_DIM,
                                    ACTION_DIM,
                                    T_SAMPLING,
                                    CONTROL_H,
                                    TIME_IDX,
                                    TIME_INIT_VAR,
                                    initial_state_norm,
                                    initial_state_var_norm)


def get_config(seed=1, num_trials=10, fast=False, dtype=torch.float64, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               optim_horizon_steps=None, num_anchor_batches=0, num_anchors=12, anchor_var=0.01,
               risk_weight=0.0, visc_penalty=0.02, harvest_reward=True, constraint_strength=1.5,
               num_high_feed_probes=0, high_feed_levels=(0.6, 0.8, 1.0), pms_visc_delay=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_explorations = 5

    n_particles = 20 if fast else 400
    n_opt_steps = 150 if fast else 1000
    n_epoch = 100 if fast else 500

    n_list = num_trials + num_explorations

    gp_input_dim = STATE_DIM + ACTION_DIM
    num_gp = STATE_DIM

    init_dict_RBF = {
        "active_dims": np.arange(0, gp_input_dim),
        "lengthscales_init": np.ones(gp_input_dim),
        "flg_train_lengthscales": True,
        "lambda_init": np.ones(1),
        "flg_train_lambda": True,
        "sigma_n_init": 0.01 * np.ones(1),
        "flg_train_sigma_n": True,
        "sigma_n_num": 1e-3,
        "dtype": dtype,
        "device": device,
    }

    model_learning_par = {
        "num_gp": num_gp,
        "init_dict_list": [init_dict_RBF] * num_gp,
        "approximation_mode": "SOD",
        "approximation_dict": {"SOD_threshold_mode": "relative",
                               "SOD_threshold": 0.3,
                               "flg_SOD_permutation": False},
        # Rescales each GP's training targets by their own max(|delta|) before fitting (see
        # Model_learning.train_gp_likelihood), so sigma_n_init/lambda_init below are read against a
        # uniform ~O(1) target scale on every channel instead of each channel's raw delta magnitude
        # (which differ by orders of magnitude across {Wt, X, P, Viscosity, time}). Prediction mean
        # is unaffected (linear in Y, so raw-scale alpha falls out for free); predicted variance is
        # rescaled back explicitly in Model_learning.get_next_state.
        "flg_norm": False,
        "dtype": dtype, "device": device,
    }

    rand_exploration_policy_par = {
        "state_dim": STATE_DIM, "input_dim": ACTION_DIM,
        "u_max": 1.0, "dtype": dtype, "device": device,
    }

    num_basis = 100
    control_policy_par = {
        "state_dim": STATE_DIM, "input_dim": ACTION_DIM, "u_max": 1.0,
        "num_basis": num_basis,
        "centers_init": np.random.uniform(-1.0, 1.0, (num_basis, STATE_DIM)),
        "lengthscales_init": np.ones(STATE_DIM),
        "weight_init": 0.1 * (np.random.rand(ACTION_DIM, num_basis) - 0.5),
        "flg_squash": True, "flg_drop": True, "dtype": dtype, "device": device,
    }
    policy_reinit_dict = {"lenghtscales_par": np.ones(STATE_DIM),
                          "centers_par": np.ones(STATE_DIM),
                          "weight_par": 1.0}

# CHANGED_THIS added
    std_meas_noise = 0.01 * np.ones(STATE_DIM)
    std_meas_noise[TIME_IDX] = 0.0
    # Per-channel empirical spread at K_WARM, not a uniform guess -- see initial_state_var_norm's
    # docstring for why a uniform 0.01 was disproportionate for some channels.
    initial_state_var = initial_state_var_norm().copy()
    initial_state_var[TIME_IDX] = TIME_INIT_VAR

    mc_pilco_init = {
        "T_sampling": T_SAMPLING,
        "state_dim": STATE_DIM,
        "input_dim": ACTION_DIM,
        # CHANGED_THIS
        "f_model_learning": Model_learning_RBF_det_time,
        "model_learning_par": model_learning_par,
        "f_rand_exploration_policy": Policy.Random_exploration,
        "rand_exploration_policy_par": rand_exploration_policy_par,
        "f_control_policy": Policy.Sum_of_gaussians,
        "control_policy_par": control_policy_par,
        # "f_cost_function": PeniConcentrationDenseCost,
        'f_cost_function': PeniMassChangeCost,

        # Every penalty below is now priced in kg-of-penicillin-equivalent BEFORE its lambda is
        # applied (see mcpilco/penicillin_cost.py's module/class docstrings), so these numbers are
        # NOT comparable to the pre-refactor values -- 0.5 used to be an inert unit-conversion
        # accident for visc_penalty and is now ~50x over-priced (confirmed via
        # experiments/cost_term_report.py: charges ~15,000 kg for a batch that actually loses
        # ~2,900 kg). Defaults below are a starting point read off that script's sweep, not a
        # finished calibration -- re-run it (it prints the reward-vs-batch_yield_kg guard and the
        # per-term verdicts) before trusting a change here, and re-check against GP-PREDICTED
        # rollouts before treating it as final: viscosity is over-predicted ~4.5x in training
        # (evaluate_GPs.ipynb G.7), so a value calibrated on real trajectories fires harder there.
        #
        # soft_penalty (lambda_weight): tank-overflow constraint. Left at the old value -- it is
        # measured INERT on every reachable trajectory (Wt never nears WT_SOFT), so its magnitude
        # doesn't currently matter, but it's still scaled by constraint_strength if Wt behaviour
        # ever changes.
        # rate_penalty (lambda_rate): action-chatter SMOOTHNESS PREFERENCE, not a safety constraint
        # -- NOT scaled by constraint_strength (see penicillin_cost.py). 0.02 charges ~1,100 kg
        # (~31% of the good batch's own reward) on the worst-case every-step +/-1 alternation probe,
        # which no learned policy sits at continuously, so this is a soft nudge, not a hard limit.
        # visc_penalty (lambda_visc): viscosity-collapse constraint, scaled by constraint_strength.
        # risk_weight (lambda_risk): batch-OUTCOME spread (std across particles of the summed
        # trajectory cost -- a behavioural fix from the old per-timestep spread, see
        # PeniConcentrationCost.forward), also scaled by constraint_strength. 0.0 = risk-neutral.
        # The old "~25x the mean, use 0.005-0.02" guidance was calibrated against the OLD
        # per-timestep-summed std and does NOT carry over -- re-derive a working range against
        # this outcome-std formulation (e.g. via std_cost_trial_list) before relying on it.
        # constraint_strength: single global knob, multiplies soft_penalty, visc_penalty AND
        # risk_weight together ("how conservative overall"); 1.0 = exactly what those three specify.
        # visc_penalty/constraint_strength used to be hardcoded literals here (0.02/1.5),
        # silently ignoring whatever was passed into get_config()/the CLI --visc_penalty flag.
        # The literals happened to match this function's own defaults (see above), so default
        # behaviour is unchanged by wiring them as real parameters -- but --visc_penalty on
        # 02_mcpilco_single_phase.py previously had NO EFFECT at all; it does now.
        "cost_function_par": {"p_weight": 0.05, "soft_penalty": 0.05, "rate_penalty": 0.02,
                              "risk_weight": risk_weight, "visc_penalty": visc_penalty,
                              "harvest_reward": harvest_reward,
                              "constraint_strength": constraint_strength},
        # CHANGED_THIS
        "std_meas_noise": std_meas_noise,
        "log_path": f"results/single_phase/seed{seed}",
        "optim_horizon_steps": optim_horizon_steps,
        "dtype": dtype, "device": device,
    }

    model_opt_dict = {
        "f_optimizer": "lambda p : torch.optim.Adam(p, lr=0.01)",
        "criterion": Likelihood.Marginal_log_likelihood,
        "N_epoch": n_epoch, "N_epoch_print": 100,
    }
    model_optimization_opt_list = [model_opt_dict] * num_gp

    policy_optimization_dict = {
        "num_particles": n_particles,
        "opt_steps_list": [n_opt_steps] * n_list,
        # original 0.01
        "lr_list": [0.01] * n_list,
        "f_optimizer": "lambda p, lr : torch.optim.Adam(p, lr)",
        "num_step_print": 100,
        "p_dropout_list": [0.25] * n_list,
        "p_drop_reduction": 0.1,
        "alpha_diff_cost": 0.99,
        "min_diff_cost": 0.05,
        "num_min_diff_cost": 50,
        # CHANGED_THIS from 200
        "min_step": n_opt_steps // 3,
        "lr_min": 0.001, #"lr_min": 0.001,
        "policy_reinit_dict": policy_reinit_dict,
    }

    reinforce_par = {
        "initial_state": initial_state_norm(),
        # CHANGED_THIS
        "initial_state_var": initial_state_var,
        "T_exploration": CONTROL_H,
        "T_control": CONTROL_H,
        "num_trials": num_trials,
        "num_explorations": num_explorations,
        "model_optimization_opt_list": model_optimization_opt_list,
        "policy_optimization_dict": policy_optimization_dict,
    }

    # An explicit parameter (not a bare literal) so it round-trips through note.txt/eval's
    # config reconstruction (see eval_single_phase_lib.py's _GET_CONFIG_KEYS/_build_cfg_kwargs)
    # instead of every reconstruction silently assuming today's default regardless of what a
    # given saved run actually used.
    wrapper_par = {"seed_offset": seed * 1000, "pms_visc_delay": pms_visc_delay}

    anchor_par = {"num_batches": num_anchor_batches, "num_anchors": num_anchors, "anchor_var": anchor_var}

    probe_par = {"num_probes": num_high_feed_probes, "levels": high_feed_levels}

    return {"mc_pilco_init": mc_pilco_init, "reinforce_par": reinforce_par,
            "wrapper_par": wrapper_par, "anchor_par": anchor_par, "probe_par": probe_par}

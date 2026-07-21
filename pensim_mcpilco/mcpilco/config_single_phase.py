
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

from mcpilco.penicillin_cost import PeniConcentrationCost

# CHANGED_THIS
from mcpilco.model_learning_det_time import Model_learning_RBF_det_time
from mcpilco.pensim_wrapper import (STATE_DIM,
                                    ACTION_DIM,
                                    T_SAMPLING,
                                    CONTROL_H,
                                    TIME_IDX,
                                    TIME_INIT_VAR,
                                    initial_state_norm)


def get_config(seed=1, num_trials=10, fast=False, dtype=torch.float64, device=torch.device("cpu"),
               optim_horizon_steps=None, num_anchor_batches=0, num_anchors=12, anchor_var=0.01,
               risk_weight=0.0, visc_penalty=0.5, harvest_reward=True,
               num_high_feed_probes=0, high_feed_levels=(0.6, 0.8, 1.0)):
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
    initial_state_var = 0.01 * np.ones(STATE_DIM)
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
        "f_cost_function": PeniConcentrationCost,

        # risk_weight scales the across-particle std IN THE OPTIMISED OBJECTIVE (0.0 = stock
        # risk-neutral mean). The std runs ~25x the mean cost here, so useful values are small:
        # ~0.005-0.02 makes the penalty roughly 20% of the objective; >=0.1 swamps the yield signal.
        # visc_penalty guards the observed collapse mode (broth thickens -> O2 transfer fails ->
        # product degrades); harvest_reward credits penicillin removed by the discharge pulses, which
        # the in-tank-only reward discarded (~20% of batch_yield_kg).
        "cost_function_par": {"p_weight": 0.05, "soft_penalty": 0.5, "rate_penalty": 0.5,
                              "risk_weight": risk_weight, "visc_penalty": visc_penalty,
                              "harvest_reward": harvest_reward},
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
        "num_min_diff_cost": 100,
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

    wrapper_par = {"seed_offset": seed * 1000}

    anchor_par = {"num_batches": num_anchor_batches, "num_anchors": num_anchors, "anchor_var": anchor_var}

    probe_par = {"num_probes": num_high_feed_probes, "levels": high_feed_levels}

    return {"mc_pilco_init": mc_pilco_init, "reinforce_par": reinforce_par,
            "wrapper_par": wrapper_par, "anchor_par": anchor_par, "probe_par": probe_par}

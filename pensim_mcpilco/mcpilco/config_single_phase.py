
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
from mcpilco.pensim_wrapper import (STATE_DIM,
                                    ACTION_DIM,
                                    T_SAMPLING,
                                    CONTROL_H,
                                    initial_state_norm)


def get_config(seed=1, num_trials=10, fast=False, dtype=torch.float64, device=torch.device("cpu"),
               optim_horizon_steps=None, num_anchor_batches=0, num_anchors=12, anchor_var=0.01):
    # For Policy's initial centers/weioghts, and MCPILCO particle sampling/dropout
    torch.manual_seed(seed)
    np.random.seed(seed)

    # initial exploration perturbed-recipe batches before the trial loop
    num_explorations = 5

    # fast debug mode
    n_particles = 20 if fast else 100
    n_opt_steps = 150 if fast else 1000
    n_epoch = 100 if fast else 500

    n_list = num_trials + num_explorations

    gp_input_dim = STATE_DIM + ACTION_DIM
    num_gp = STATE_DIM # one GP per state dim

    # GP dynamics model (one RBF GP per state dim), ONE SHARED init dict for all of them.
    #
    # A per-channel init (lambda/sigma_n/jitter scaled to each channel's measured delta std) was
    # tried and REVERTED: it improved the model (held-out R^2 up on every channel) but COLLAPSED the
    # policy -- in full mode the controller saturated to a constant a=-1 (action std exactly 0), i.e.
    # it stopped being a state-feedback controller at all. Isolated against this shared init on the
    # same seed: shared -> 0% floored, action std 0.055; per-GP -> 100% floored, action std 0.000000.
    # (Fast mode's 150 opt-steps never reached tanh saturation, so an earlier fast A/B missed it.)
    # Do NOT reintroduce without: measuring scales on demand from FRESH recipe batches (never from
    # results/), changing ONE knob at a time, and A/B-ing in FULL mode.
    init_dict_RBF = {
        "active_dims": np.arange(0, gp_input_dim),
        "lengthscales_init": np.ones(gp_input_dim),
        "flg_train_lengthscales": True,
        "lambda_init": np.ones(1),
        "flg_train_lambda": True,
        "sigma_n_init": 0.01 * np.ones(1),
        "flg_train_sigma_n": True,
        # numerical jitter; keeps the predictive std from underflowing on near-constant channels
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

    mc_pilco_init = {
        "T_sampling": T_SAMPLING,
        "state_dim": STATE_DIM,
        "input_dim": ACTION_DIM,
        "f_model_learning": ML.Model_learning_RBF,
        "model_learning_par": model_learning_par,
        "f_rand_exploration_policy": Policy.Random_exploration,
        "rand_exploration_policy_par": rand_exploration_policy_par,
        "f_control_policy": Policy.Sum_of_gaussians,
        "control_policy_par": control_policy_par,
        "f_cost_function": PeniConcentrationCost,

        "cost_function_par": {"p_weight": 0.05, "soft_penalty": 0.5, "paa_penalty": 10, "do2_penalty": 5.0, "rate_penalty": 0.5},
        "std_meas_noise": 0.01 * np.ones(STATE_DIM),
        "log_path": f"results/single_phase/seed{seed}",
        # cap the imagined GP-rollout horizon during policy optimisation (None -> stock full horizon;
        # inert when None -- PenSimMCPILCO.apply_policy short-circuits on it)
        "optim_horizon_steps": optim_horizon_steps,
        "dtype": dtype, "device": device,
    }

    model_opt_dict = {
        "f_optimizer": "lambda p : torch.optim.Adam(p, lr=0.01)",
        "criterion": Likelihood.Marginal_log_likelihood,
        "N_epoch": n_epoch, "N_epoch_print": 100,
    }
    model_optimization_opt_list = [model_opt_dict] * num_gp

    # inherited from the upstream MC-PILCO cartpole reference config
    policy_optimization_dict = {
        "num_particles": n_particles, # 400 ->100
        # scaled down and flattened
        "opt_steps_list": [n_opt_steps] * n_list,
        "lr_list": [0.01] * n_list,
        "f_optimizer": "lambda p, lr : torch.optim.Adam(p, lr)",
        "num_step_print": 100,
        "p_dropout_list": [0.25] * n_list,
        "p_drop_reduction": 0.1, # 0.125 -> 0.1
        "alpha_diff_cost": 0.99,
        "min_diff_cost": 0.05, # 0.08 -> 0.05
        "num_min_diff_cost": 100, # 200 -> 100
        "min_step": 200,
        "lr_min": 0.001, # 0.0025 -> 0.001
        "policy_reinit_dict": policy_reinit_dict,
    }

    # RL phase starts at the warmed-up state (t=WARMUP_H); horizon = CONTROL_H hours.
    reinforce_par = {
        "initial_state": initial_state_norm(),
        "initial_state_var": 0.01 * np.ones(STATE_DIM),
        "T_exploration": CONTROL_H,
        "T_control": CONTROL_H,
        "num_trials": num_trials,
        "num_explorations": num_explorations,
        "model_optimization_opt_list": model_optimization_opt_list,
        "policy_optimization_dict": policy_optimization_dict,
    }

    wrapper_par = {"seed_offset": seed * 1000}

    # multi-origin short-rollout anchors (num_batches=0 -> disabled; setup uses the run's seed_offset)
    anchor_par = {"num_batches": num_anchor_batches, "num_anchors": num_anchors, "anchor_var": anchor_var}

    return {"mc_pilco_init": mc_pilco_init, "reinforce_par": reinforce_par,
            "wrapper_par": wrapper_par, "anchor_par": anchor_par}

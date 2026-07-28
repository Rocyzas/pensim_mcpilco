
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MC-PILCO"))

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import policy_learning.Policy as Policy

from mcpilco.penicillin_cost import PeniConcentrationCost, PeniConcentrationDenseCost, PeniMassChangeCost

from mcpilco.model_learning_dual_phase import DualPhaseModelLearning
from mcpilco.pensim_wrapper import (STATE_DIM,
                                    ACTION_DIM,
                                    T_SAMPLING,
                                    CONTROL_H,
                                    TIME_IDX,
                                    TIME_INIT_VAR,
                                    PIVOT_HOURS,
                                    BLEND_HALF_WIDTH_HOURS,
                                    initial_state_norm,
                                    initial_state_var_norm)


def get_config(seed=1, num_trials=10, fast=False, dtype=torch.float64, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               pivot_hours=PIVOT_HOURS, blend_half_width_hours=BLEND_HALF_WIDTH_HOURS,
               risk_weight=0.0, visc_penalty=0.02, harvest_reward=True, constraint_strength=1.5,
               pms_visc_delay=True):
    """Dual-phase config: same policy/cost/exploration as config_single_phase.get_config, but
    f_model_learning is DualPhaseModelLearning -- two independent Model_learning_RBF_det_time
    instances, each with the SAME per-channel init as the single-phase model, fit on disjoint
    slices of every batch (hard split at pivot_step, decision < pivot_step -> phase 1). Their
    predictions are BLENDED at rollout time via a sigmoid centered on pivot_hours (also the
    training-split point) with the given blend_half_width_hours -- see
    model_learning_dual_phase.py.

    Unsupported here (see PenSimMCPILCOMultiPhase docstring): recipe anchors, high-feed
    probes, optim_horizon_steps -- their launch states don't carry the absolute decision time
    the phase router needs, so those knobs are not exposed by this config at all.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pivot_step = int(round(pivot_hours / T_SAMPLING))

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

    # Both phases start from the SAME per-channel hyperparameter init and approximation
    # settings as the single-phase model -- they only ever diverge because each is fit on a
    # disjoint slice of every batch (see DualPhaseModelLearning.add_data).
    def _phase_model_learning_par():
        return {
            "num_gp": num_gp,
            "init_dict_list": [init_dict_RBF] * num_gp,
            "approximation_mode": "SOD",
            "approximation_dict": {"SOD_threshold_mode": "relative",
                                   "SOD_threshold": 0.3,
                                   "flg_SOD_permutation": False},
            "flg_norm": False,
            "dtype": dtype, "device": device,
        }

    model_learning_par = {
        "pivot_step": pivot_step,
        "pivot_hours": pivot_hours,
        "blend_half_width_hours": blend_half_width_hours,
        "phase1_par": _phase_model_learning_par(),
        "phase2_par": _phase_model_learning_par(),
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

    std_meas_noise = 0.01 * np.ones(STATE_DIM)
    std_meas_noise[TIME_IDX] = 0.0
    initial_state_var = initial_state_var_norm().copy()
    initial_state_var[TIME_IDX] = TIME_INIT_VAR

    mc_pilco_init = {
        "T_sampling": T_SAMPLING,
        "state_dim": STATE_DIM,
        "input_dim": ACTION_DIM,
        "f_model_learning": DualPhaseModelLearning,
        "model_learning_par": model_learning_par,
        "f_rand_exploration_policy": Policy.Random_exploration,
        "rand_exploration_policy_par": rand_exploration_policy_par,
        "f_control_policy": Policy.Sum_of_gaussians,
        "control_policy_par": control_policy_par,
        "f_cost_function": PeniMassChangeCost,
        # "f_cost_function": PeniConcentrationDenseCost,
        "cost_function_par": {"p_weight": 0.05, "soft_penalty": 0.05, "rate_penalty": 0.02,
                              "risk_weight": risk_weight, "visc_penalty": visc_penalty,
                              "harvest_reward": harvest_reward,
                              "constraint_strength": constraint_strength},
        "std_meas_noise": std_meas_noise,
        "log_path": f"results/dual_phase/seed{seed}",
        # PenSimMCPILCOMultiPhase.apply_policy asserts this is None -- not supported combined
        # with dual-phase (see its docstring).
        "optim_horizon_steps": None,
        "dtype": dtype, "device": device,
    }

    model_opt_dict = {
        "f_optimizer": "lambda p : torch.optim.Adam(p, lr=0.01)",
        "criterion": Likelihood.Marginal_log_likelihood,
        "N_epoch": n_epoch, "N_epoch_print": 100,
    }
    # 2*num_gp: DualPhaseModelLearning.reinforce_model splits this list at phase1.num_gp,
    # so it must cover both phases' GPs (see model_learning_dual_phase.py).
    model_optimization_opt_list = [model_opt_dict] * (2 * num_gp)

    policy_optimization_dict = {
        "num_particles": n_particles,
        "opt_steps_list": [n_opt_steps] * n_list,
        "lr_list": [0.01] * n_list,
        "f_optimizer": "lambda p, lr : torch.optim.Adam(p, lr)",
        "num_step_print": 100,
        "p_dropout_list": [0.25] * n_list,
        "p_drop_reduction": 0.1,
        "alpha_diff_cost": 0.99,
        "min_diff_cost": 0.05,
        "num_min_diff_cost": 50,
        "min_step": n_opt_steps // 3,
        "lr_min": 0.001,
        "policy_reinit_dict": policy_reinit_dict,
    }

    reinforce_par = {
        "initial_state": initial_state_norm(),
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

    return {"mc_pilco_init": mc_pilco_init, "reinforce_par": reinforce_par,
            "wrapper_par": wrapper_par}

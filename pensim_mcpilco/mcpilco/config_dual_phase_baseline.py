"""Plain-RBF ablation baseline for the DUAL-phase model: identical to config_dual_phase.get_config
in every respect except the two phase sub-models, which are swapped from Model_learning_RBF_det_time
(Wt mass-balance / Viscosity recipe-mean prior means) to Model_learning_RBF_baseline -- every channel
in BOTH phases falls through to a vanilla zero-mean RBF (see model_learning_baseline.py). This is the
exact dual-phase analog of config_single_phase_baseline (which does the same swap for the single-phase
model), and exists to isolate what those two prior means contribute in the dual-phase setup, as a
baseline to compare against config_dual_phase's runs.

Also drops `time` from every phase GP's own INPUT regressors (active_dims) -- not just its own channel
(already deterministic), but as a regressor for every other channel too -- mirroring
config_single_phase_baseline exactly. `time` still exists as a tracked state channel everywhere
(deterministic clock, policy feature); it just stops being fed to any GP as a regressor.

Done by post-processing DualPhaseModelLearning's model_learning_par AFTER calling
config_dual_phase.get_config: (1) inject phase_model_cls=Model_learning_RBF_baseline so
DualPhaseModelLearning builds plain-RBF phases instead of its default det_time phases (see
model_learning_dual_phase.py's __init__, which documents this as the intended override point), and
(2) drop TIME_IDX from each phase's init_dict_list active_dims/lengthscales_init. config_dual_phase.py
/ 03_mcpilco_dual_phase.py are not touched at all, and every other setting (GP hyperparameter inits,
SOD, pivot/blend, penalties, measurement flags) still comes from config_dual_phase unchanged.

Accepts the exact same kwargs as config_dual_phase.get_config (passed straight through), so it stays
in sync automatically as that function's parameter set evolves.
"""

import numpy as np

from mcpilco import config_dual_phase
from mcpilco.model_learning_baseline import Model_learning_RBF_baseline
from mcpilco.pensim_wrapper import TIME_IDX


def _drop_time_input(init_dict):
    """New dict, not a mutation: config_dual_phase.py's init_dict_RBF is the SAME object reused for
    every gp_index in BOTH phase1_par and phase2_par (see its init_dict_list = [init_dict_RBF] *
    num_gp), so mutating it in place here would corrupt every other GP's active_dims -- the same
    aliasing hazard config_single_phase_baseline._drop_time_input avoids."""
    active_dims = np.asarray(init_dict["active_dims"])
    keep = active_dims != TIME_IDX
    return dict(init_dict,
               active_dims=active_dims[keep],
               lengthscales_init=np.asarray(init_dict["lengthscales_init"])[keep])


def get_config(**kwargs):
    cfg = config_dual_phase.get_config(**kwargs)
    mlp = cfg["mc_pilco_init"]["model_learning_par"]
    # build plain-RBF baseline phases instead of DualPhaseModelLearning's default det_time phases
    mlp["phase_model_cls"] = Model_learning_RBF_baseline
    # drop `time` as a GP INPUT regressor in BOTH phases (mirrors config_single_phase_baseline)
    for phase_key in ("phase1_par", "phase2_par"):
        mlp[phase_key]["init_dict_list"] = [
            _drop_time_input(d) for d in mlp[phase_key]["init_dict_list"]]
    cfg["mc_pilco_init"]["log_path"] = f"results/dual_phase_baseline/seed{kwargs.get('seed', 1)}"
    return cfg

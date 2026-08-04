"""Plain-RBF ablation baseline: identical to config_single_phase.get_config in every respect
except f_model_learning, which is swapped for Model_learning_RBF_baseline -- same deterministic
`time` channel, state_clamp and SOD approximation, just WITHOUT the Wt mass-balance / Viscosity
recipe-mean prior means (see model_learning_baseline.py). Exists to isolate what those two prior
means contribute, as a baseline to compare against config_single_phase's runs.

Also drops `time` from every GP's own INPUT regressors (active_dims) -- not just its own channel
(already deterministic via DETERMINISTIC_CHANNELS, inherited unchanged), but as a regressor for
every other channel too. Mirrors the Viscosity-only exclusion model_learning_det_time.py tried
and later reverted (see its RECIPE_MEAN_CHANNELS comment -- current action-lengthscale
diagnostics no longer showed the outlier that motivated it there), applied uniformly here instead
since this is a deliberately stripped-down baseline, not the tuned model. `time` still exists as
a tracked state channel everywhere (deterministic clock, policy feature) -- it just stops being
fed to any GP as a regressor.

Done by post-processing init_dict_list AFTER calling config_single_phase.get_config, exactly like
the f_model_learning swap above -- config_single_phase.py / 02_mcpilco_single_phase.py are not
touched at all, and every other GP hyperparameter (lengthscale/lambda/sigma_n inits, SOD
settings) still comes from get_config unchanged.

Accepts the exact same kwargs as config_single_phase.get_config (passed straight through), so it
stays in sync automatically as that function's parameter set evolves.
"""

import numpy as np

from mcpilco import config_single_phase
from mcpilco.model_learning_baseline import Model_learning_RBF_baseline
from mcpilco.pensim_wrapper import TIME_IDX


def _drop_time_input(init_dict):
    """New dict, not a mutation: config_single_phase.py's init_dict_RBF is the SAME object
    reused for every gp_index's list entry (see its own init_dict_list = [init_dict_RBF] *
    num_gp), so mutating it in place here would corrupt every other GP's active_dims too --
    same aliasing hazard the old VISC_GP_IDX branch in model_learning_det_time.py had to avoid."""
    active_dims = np.asarray(init_dict["active_dims"])
    keep = active_dims != TIME_IDX
    return dict(init_dict,
               active_dims=active_dims[keep],
               lengthscales_init=np.asarray(init_dict["lengthscales_init"])[keep])


def get_config(**kwargs):
    cfg = config_single_phase.get_config(**kwargs)
    cfg["mc_pilco_init"]["f_model_learning"] = Model_learning_RBF_baseline
    model_learning_par = cfg["mc_pilco_init"]["model_learning_par"]
    model_learning_par["init_dict_list"] = [
        _drop_time_input(d) for d in model_learning_par["init_dict_list"]]
    cfg["mc_pilco_init"]["log_path"] = f"results/single_phase_baseline/seed{kwargs.get('seed', 1)}"
    return cfg

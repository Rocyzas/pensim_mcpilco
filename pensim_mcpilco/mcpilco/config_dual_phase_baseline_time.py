"""Plain-RBF DUAL-phase baseline WITH `time` kept as a GP regressor: identical to
config_dual_phase_baseline in every respect (same phase_model_cls=Model_learning_RBF_baseline
injection -- every channel in BOTH phases a vanilla zero-mean RBF, no Wt mass-balance / Viscosity
recipe-mean prior means) EXCEPT that it does NOT drop `time` from each phase GP's input regressors.
config_dual_phase_baseline strips TIME_IDX from both phases' active_dims via _drop_time_input; this
variant skips that, so `time` is fed to every GP in both phases as a regressor.

This is the exact dual-phase analog of config_single_phase_baseline_time. Purpose: isolate the
effect of the `time` regressor on the plain-RBF baseline's drift-channel underfitting, without
re-introducing the prior means. Compare this config's C2b/C4b against config_dual_phase_baseline's.

Logs to its own results/dual_phase_baseline_time/ tree so it never collides with the
config_dual_phase_baseline runs it is compared against. Accepts the exact same kwargs as
config_dual_phase.get_config (passed straight through), so it stays in sync automatically.
"""

from mcpilco import config_dual_phase
from mcpilco.model_learning_baseline import Model_learning_RBF_baseline


def get_config(**kwargs):
    cfg = config_dual_phase.get_config(**kwargs)
    mlp = cfg["mc_pilco_init"]["model_learning_par"]
    # build plain-RBF baseline phases instead of DualPhaseModelLearning's default det_time phases
    mlp["phase_model_cls"] = Model_learning_RBF_baseline
    # NOTE: intentionally NOT dropping `time` from either phase's active_dims -- `time` is kept as a
    # GP regressor in BOTH phases. That single omission is the only difference from
    # config_dual_phase_baseline.
    cfg["mc_pilco_init"]["log_path"] = f"results/dual_phase_baseline_time/seed{kwargs.get('seed', 1)}"
    return cfg

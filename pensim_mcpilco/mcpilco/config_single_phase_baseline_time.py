"""Plain-RBF baseline WITH `time` kept as a GP regressor: identical to
config_single_phase_baseline in every respect (same Model_learning_RBF_baseline swap -- every
channel a vanilla zero-mean RBF, no Wt mass-balance / Viscosity recipe-mean prior means) EXCEPT
that it does NOT drop `time` from each GP's input regressors. config_single_phase_baseline calls
_drop_time_input to strip TIME_IDX from every GP's active_dims; this variant simply skips that
step, so `time` is fed to every GP as a regressor (as in the un-ablated config_single_phase).

Purpose: isolate the effect of the `time` regressor alone. The plain-RBF baseline underfits the
monotonic drift channels (Wt, Viscosity, X, P) partly because, with a zero mean AND no `time`
input, the RBF cannot locate the batch phase. Re-adding `time` gives it the phase directly -- a
physics-free way to reduce that underfitting, without re-introducing the prior means. Compare this
config's C2b/C4b against config_single_phase_baseline's to measure what the `time` regressor buys.

Logs to its own results/single_phase_baseline_time/ tree so it never collides with the
config_single_phase_baseline runs it is meant to be compared against. Accepts the exact same kwargs
as config_single_phase.get_config (passed straight through), so it stays in sync automatically as
that function's parameter set evolves.
"""

from mcpilco import config_single_phase
from mcpilco.model_learning_baseline import Model_learning_RBF_baseline


def get_config(**kwargs):
    cfg = config_single_phase.get_config(**kwargs)
    cfg["mc_pilco_init"]["f_model_learning"] = Model_learning_RBF_baseline
    # NOTE: intentionally NOT calling _drop_time_input -- `time` is kept as a GP regressor here.
    # That single omission is the only difference from config_single_phase_baseline.
    cfg["mc_pilco_init"]["log_path"] = f"results/single_phase_baseline_time/seed{kwargs.get('seed', 1)}"
    return cfg

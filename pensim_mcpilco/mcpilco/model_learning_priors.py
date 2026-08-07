"""Plain-RBF dual-phase baseline + an empirical recipe-trajectory prior mean on EVERY learned
channel. Extends Model_learning_RBF_baseline (same plain zero-mean RBF kernels, same dropped-`time`
INPUT regressors), but get_gp wraps each non-deterministic channel's GP with a recipe-trajectory
prior mean measured from NUM_PRIOR_RECIPE_BATCHES pure-recipe (a=0) simulator batches. So each GP's
prior mean becomes m(x,u) = "this channel changes like the recipe at this batch time" instead of
zero, and the RBF learns only the action-induced residual on top -- the semiparametric decomposition
that mirrors how the action itself is parameterised (a residual on the recipe feed).

This is the generalisation, to ALL channels, of the Viscosity-only RBF_RecipeMean already used by
Model_learning_RBF_det_time (see model_learning_det_time.py / recipe_trajectory_mean.py). It reuses
that existing, tested measurement (RecipeTrajectoryMean) and the same get_mean-additive pattern, so
no existing file is changed; this class only exists to (a) apply the recipe mean to every channel
and (b) measure it from NUM_PRIOR_RECIPE_BATCHES batches (RecipeTrajectoryMean's own default is 4).

Note on the dropped `time`: the recipe mean is indexed by batch time (RecipeTrajectoryMean.delta_norm
reads X[:, TIME_IDX]). That still works under the baseline's dropped-time kernel because get_mean is
called on the FULL, unsliced gp input (GP_prior.get_estimate_from_alpha -> get_mean(X_test)); active_dims
only slices the KERNEL (Stationary_GP.get_covariance), so the `time` column is present in X for the mean
lookup even though the kernel never sees it. `time` itself stays deterministic (inherited from
det_time) and gets a plain RBF (no prior mean -- its GP is ignored anyway).
"""

import gpr_lib.GP_prior.Stationary_GP as SGP

from mcpilco.model_learning_baseline import Model_learning_RBF_baseline
from mcpilco.model_learning_det_time import DETERMINISTIC_CHANNELS
from mcpilco.recipe_trajectory_mean import RecipeTrajectoryMean
from mcpilco.pensim_wrapper import STATE_NAMES

# The recipe prior mean is measured from this many pure-recipe batches (the user's ask: 10).
# RecipeTrajectoryMean memoises the measurement per (num_batches, seed_offset), so the 10 recipe
# rollouts are paid for ONCE per process no matter how many channels/phases reference them.
NUM_PRIOR_RECIPE_BATCHES = 10


class RBF_RecipeMeanN(SGP.RBF):
    """RBF kernel whose PRIOR MEAN is the measured pure-recipe delta for one channel -- identical to
    model_learning_det_time.RBF_RecipeMean, but with a configurable recipe batch count (default 10).
    Semiparametric: mean = super().get_mean(X) + recipe-delta(X); parameter set is identical to plain
    RBF so saved state_dicts stay load-compatible."""

    def __init__(self, channel=None, num_batches=NUM_PRIOR_RECIPE_BATCHES, **init_dict):
        super().__init__(**init_dict)
        self._recipe_mean = RecipeTrajectoryMean(channel, num_batches=num_batches,
                                                 dtype=self.dtype, device=self.device)

    def get_mean(self, X):
        return super().get_mean(X) + self._recipe_mean.delta_norm(X)


class Model_learning_RBF_recipe_priors(Model_learning_RBF_baseline):
    """Model_learning_RBF_baseline with a recipe-trajectory prior mean on every LEARNED channel.

    Wired as a phase_model_cls for DualPhaseModelLearning by config_dual_phase_baseline_priors -- both
    phases get recipe-mean GPs. Each phase's GPs are queried at their own batch-time range, and the
    recipe mean is indexed by that time, so one shared (memoised) recipe measurement serves both phases
    correctly.
    """

    def get_gp(self, gp_index, init_dict):
        if gp_index in DETERMINISTIC_CHANNELS:
            # `time` is deterministic (handled by det_time's get_next_state_from_gp_output) -> plain
            # RBF, no prior mean needed (its GP output is ignored during rollout).
            return super().get_gp(gp_index, init_dict)
        return RBF_RecipeMeanN(channel=STATE_NAMES[gp_index], **init_dict)

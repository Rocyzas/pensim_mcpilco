"""GP dynamics model that treats `time` as a deterministic clock rather than a learned state.

`time` stays in the state vector (so every GP still sees it as an INPUT regressor, and the
policy still uses it as a gain-scheduling feature), but its own imagined-rollout update is
forced to a constant, noiseless tick. This removes a pointless GP over a constant target and,
more importantly, stops the imagined clock from random-walking over the ~114 rollout steps --
a drifting clock would corrupt every other channel, since they all regress on time.
See pensim_wrapper.TIME_IDX / TIME_DELTA_NORM.
"""

import torch

import model_learning.Model_learning as ML
import gpr_lib.GP_prior.Stationary_GP as SGP

from mcpilco.pensim_wrapper import STATE_NAMES, STATE_DIM, TIME_IDX, TIME_DELTA_NORM, TIME_INIT_VAR

WT_GP_IDX = STATE_NAMES.index("Wt")
X_GP_IDX = STATE_NAMES.index("X")
VISC_GP_IDX = STATE_NAMES.index("Viscosity")
# gp_input = concat(state, action) (Model_learning.data_to_gp_input); action is the only, last
# input column, so its position in the lengthscales vector is always STATE_DIM.
ACTION_INPUT_IDX = STATE_DIM

# Channels whose GP is forced to keep resolving the feed-rate action rather than optimise it away.
# Measured directly off trained checkpoints (log.pkl["parameters_gp_14"], seed12_{0,2,4}, see
# evaluations plan Finding 5): the X and Viscosity GPs converge to an action-lengthscale of
# 11.6-39.7 on this [-1, 1] input in every one of 5 reward-shaping configs -- so large the kernel
# has decided feed rate has ~no effect on biomass growth or viscosity. That means a viscosity
# penalty in the cost function has no learned gradient path back to the policy, regardless of its
# weight. RBF_BoundedActionLengthscale caps that one lengthscale so the kernel cannot optimise the
# action-dependence away.
BOUNDED_ACTION_GP_IDX = {X_GP_IDX, VISC_GP_IDX}
MAX_ACTION_LENGTHSCALE = 2.0

# Channels given the empirical recipe-trajectory prior mean (see recipe_trajectory_mean.py).
# `Wt` is deliberately NOT here -- it has the exact analytic mass balance, which is strictly better.
# Start with {"P"} alone: it is the channel the cost depends on, so its effect is measurable in
# isolation via the R.1b particle-spread and C.2b coverage diagnostics. Add "X" only once P is shown
# to help; changing both at once makes the result uninterpretable.
RECIPE_MEAN_CHANNELS = {"P"}
RECIPE_MEAN_GP_IDX = {STATE_NAMES.index(c) for c in RECIPE_MEAN_CHANNELS}

_wt_mb_cache = {}
_recipe_mean_cache = {}


def _get_wt_mass_balance(dtype, device):
    """Lazily build (and memoise) the Wt mass-balance prior mean.

    Lazy because Model_learning.__init__ calls init_gp_models() -> get_gp() BEFORE any subclass
    __init__ body runs, so the object cannot be stored on self ahead of time.
    """
    key = (dtype, str(device))
    if key not in _wt_mb_cache:
        from mcpilco.wt_mass_balance import WtMassBalance
        _wt_mb_cache[key] = WtMassBalance(dtype=dtype, device=device)
    return _wt_mb_cache[key]


def _get_recipe_mean(channel, dtype, device):
    """Lazily build (and memoise) a channel's recipe-trajectory prior mean.

    Lazy for the same reason as above, and memoised per (channel, dtype, device) so the recipe
    rollouts that measure it are paid for once per process rather than once per GP re-fit -- note
    reinforce_model() calls init_gp_models() again every trial.
    """
    key = (channel, dtype, str(device))
    if key not in _recipe_mean_cache:
        from mcpilco.recipe_trajectory_mean import RecipeTrajectoryMean
        _recipe_mean_cache[key] = RecipeTrajectoryMean(channel, dtype=dtype, device=device)
    return _recipe_mean_cache[key]


class RBF_WtMassBalance(SGP.RBF):
    """RBF kernel whose PRIOR MEAN is the known recipe mass balance for `Wt`.

    Semiparametric: the physics mean carries the deterministic ~7200 kg discharge pulses (a
    one-decision-wide discontinuity an RBF cannot represent and was booking as noise -- held-out
    R^2 0.13, sigma_n ~25x the other channels), and the RBF models only the smooth residual
    (evaporation, acid/base). Parameter set is identical to plain RBF, so saved state_dicts stay
    load-compatible. Measured on held-out data the mean alone reaches R^2 ~0.95.
    """

    def __init__(self, **init_dict):
        super(RBF_WtMassBalance, self).__init__(**init_dict)
        self._wt_mb = _get_wt_mass_balance(self.dtype, self.device)

    def get_mean(self, X):
        return super(RBF_WtMassBalance, self).get_mean(X) + self._wt_mb.delta_norm(X)


class RBF_RecipeMean(SGP.RBF):
    """RBF kernel whose PRIOR MEAN is the measured pure-recipe delta for one channel.

    Semiparametric, like RBF_WtMassBalance, but empirical rather than analytic: P and X have no
    closed-form mean available from this 4-channel state (their ODE terms need S and DO2), so the
    mean is measured from a = 0 recipe batches and indexed by batch time. The RBF then models only
    the action-induced residual -- which mirrors how the action itself is parameterised, as a
    residual on the recipe feed.

    The point is the OFF-DATA behaviour. A zero-mean RBF decays to "delta = 0", freezing an
    accumulating channel; this decays to "grows like the recipe", which is both physically
    plausible and the correct a = 0 limit. Parameter set is identical to plain RBF, so saved
    state_dicts stay load-compatible.
    """

    def __init__(self, channel=None, **init_dict):
        super(RBF_RecipeMean, self).__init__(**init_dict)
        self._channel = channel
        self._recipe_mean = _get_recipe_mean(channel, self.dtype, self.device)

    def get_mean(self, X):
        return super(RBF_RecipeMean, self).get_mean(X) + self._recipe_mean.delta_norm(X)


class RBF_BoundedActionLengthscale(SGP.RBF):
    """Plain RBF, except the lengthscale on the action input is capped at MAX_ACTION_LENGTHSCALE.

    Mirrors `Stationary_GP.get_weigted_distances` line for line except for the clamp on one entry
    of `lengthscales` -- duplicated rather than patched in place because the clamp has to apply to
    the value used in THIS forward pass, not to the stored parameter: clamping the parameter itself
    would zero its gradient permanently, while clamping the derived value lets Adam keep pushing on
    it every step while the effective lengthscale used in the kernel stays bounded (a soft ceiling,
    not a hard freeze).
    """

    def get_weigted_distances(self, X1, X2):
        if self.flg_ARD:
            lengthscales = torch.exp(self.log_lengthscales_par)
        else:
            lengthscales = torch.exp(
                self.log_lengthscales_par * torch.ones(self.num_features, dtype=self.dtype, device=self.device)
            )
        lengthscales = torch.cat([
            lengthscales[:ACTION_INPUT_IDX],
            torch.clamp(lengthscales[ACTION_INPUT_IDX:ACTION_INPUT_IDX + 1], max=MAX_ACTION_LENGTHSCALE),
            lengthscales[ACTION_INPUT_IDX + 1:],
        ])

        X1_sliced = X1[:, self.active_dims] / lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        if X2 is None:
            dist = (
                X1_squared
                + X1_squared.transpose(dim0=0, dim1=1)
                - 2 * torch.matmul(X1_sliced, X1_sliced.transpose(dim0=0, dim1=1))
            )
        else:
            X2_sliced = X2[:, self.active_dims] / lengthscales
            X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
            dist = (
                X1_squared
                + X2_squared.transpose(dim0=0, dim1=1)
                - 2 * torch.matmul(X1_sliced, X2_sliced.transpose(dim0=0, dim1=1))
            )
        return dist


class Model_learning_RBF_det_time(ML.Model_learning_RBF):
    """RBF-GP dynamics with a deterministic `time` channel and prior means on the integrating
    channels: an analytic mass balance for `Wt`, measured recipe trajectories for
    RECIPE_MEAN_CHANNELS."""

    def get_gp(self, gp_index, init_dict):
        """Wt gets the mass-balance prior mean, RECIPE_MEAN_CHANNELS get the measured recipe
        trajectory, X/Viscosity get a bounded action-lengthscale; every other channel stays a
        plain RBF."""
        if gp_index == WT_GP_IDX:
            return RBF_WtMassBalance(**init_dict)
        if gp_index in RECIPE_MEAN_GP_IDX:
            return RBF_RecipeMean(channel=STATE_NAMES[gp_index], **init_dict)
        if gp_index in BOUNDED_ACTION_GP_IDX:
            return RBF_BoundedActionLengthscale(**init_dict)
        return super(Model_learning_RBF_det_time, self).get_gp(gp_index, init_dict)

    # Clamp every imagined next-state to the normalised physical box each rollout step. The real
    # data pipeline stores states clipped to [-1, 1] (PenSimWrapper.rollout / extract_state), so the
    # GP was only ever trained on [-1, 1]; without this, a particle that drifts past +1 is fed back
    # as a GP query far outside the training domain and hallucinates. Worse, the log-encoded
    # channels {Wt, X, P} are decoded through exp(): a normalised value > 1 decodes ABOVE the
    # physical ceiling and exp() turns a small over-shoot into 1e6..1e21, which is what blew the
    # imagined cost (and the training objective) up to ~1e21. Clamping to [-1, 1] keeps particles
    # in-distribution and bounds the decoded quantities (P<=40 g/L, Wt<=1.3e5 L). Set to None to
    # restore the original unbounded rollout (e.g. for an ablation).
    state_clamp = (-1.0, 1.0)

    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list,
                                      particle_pred=True):
        # Force the time channel's predicted delta to the exact clock tick. Its variance must be a
        # tiny positive jitter, not 0: the base builds one Normal over the whole state vector and
        # torch requires scale > 0. The jitter's sample is then discarded below, so time advances
        # deterministically and identically across all particles (no random walk over the rollout).
        # The time GP still trains/predicts and is simply ignored for this channel.
        gp_output_mean_list[TIME_IDX] = torch.full_like(
            gp_output_mean_list[TIME_IDX], TIME_DELTA_NORM)
        gp_output_var_list[TIME_IDX] = torch.full_like(
            gp_output_var_list[TIME_IDX], TIME_INIT_VAR)
        next_states, delta_mean, delta_var = super().get_next_state_from_gp_output(
            current_state, current_input, gp_output_mean_list, gp_output_var_list,
            particle_pred)
        # Replace the sampled time with the exact deterministic tick (autograd-safe, no in-place).
        det_time = current_state[:, TIME_IDX:TIME_IDX + 1] + TIME_DELTA_NORM
        next_states = torch.cat(
            [next_states[:, :TIME_IDX], det_time, next_states[:, TIME_IDX + 1:]], dim=1)
        # Keep particles inside the normalised physical box (see `state_clamp` above). clamp is
        # out-of-place (autograd-safe) and has zero gradient at the bound, so a particle pinned at
        # the ceiling stops contributing runaway gradients to the policy optimiser.
        if self.state_clamp is not None:
            next_states = torch.clamp(next_states, self.state_clamp[0], self.state_clamp[1])
        return next_states, delta_mean, delta_var

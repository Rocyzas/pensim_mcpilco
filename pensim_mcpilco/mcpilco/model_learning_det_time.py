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

from mcpilco.pensim_wrapper import TIME_IDX, TIME_DELTA_NORM, TIME_INIT_VAR


class Model_learning_RBF_det_time(ML.Model_learning_RBF):
    """RBF-GP dynamics with a deterministic `time` channel."""

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

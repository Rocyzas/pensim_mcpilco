"""Composite model-learning object for dual-phase MC-PILCO: BLENDS the predictions of two
independent Model_learning_RBF_det_time instances (phase 1: trained on data before the pivot,
phase 2: after) via a sigmoid weight w(t) centered on the pivot, rather than learning a single
set of GPs over the whole batch. See pensim_wrapper.PenSimMCPILCOMultiPhase for the agent that
drives this.

Training data is still HARD-split at pivot_step (see add_data) -- each phase's GPs are fit only
on data strictly before/after the pivot, exactly as before. What changed is how their two
predictions are COMBINED at rollout/evaluation time: instead of routing every decision to
exactly one phase (a discontinuity at the pivot), get_next_state now computes
    y(t) = (1 - w(t)) * y1(t) + w(t) * y2(t)
with w(t) a logistic sigmoid centered at `pivot_hours`, rising from ~0.01 to ~0.99 across
`pivot_hours -+ blend_half_width_hours` (see _blend_weight). Both phases still reuse
Model_learning_RBF_det_time unmodified (Wt mass-balance prior, Viscosity recipe-mean prior,
deterministic time channel, SOD approximation, state clamp -- see model_learning_det_time.py),
so all of that per-channel logic stays defined in exactly one place. `num_gp`/`gp_list`/
`gp_inputs`/`gp_output_list` are exposed as concatenations of the two sub-models' own
(phase1 first, phase2 second) so that MC_PILCO.reinforce()'s generic log/bookkeeping code
(which indexes `model_learning.gp_list[k]` for k in range(model_learning.num_gp), etc.) keeps
working unmodified against a `2*STATE_DIM`-GP composite.
"""

import math

import torch

from mcpilco.model_learning_det_time import Model_learning_RBF_det_time
from mcpilco.pensim_wrapper import T_SAMPLING, BLEND_HALF_WIDTH_HOURS

# Blend weight is ~0 (or ~1) far enough from the pivot that evaluating the far-side phase would
# only ever change the blended output by a numerically negligible amount -- skip it rather than
# paying for a second GP forward pass on every single decision step. Deliberately matches the
# SAME "1%/99%" convention _blend_weight's k is derived from (not an arbitrarily tighter
# tolerance): w(pivot_hours - blend_half_width_hours) == _BLEND_SKIP_EPS exactly, so "both
# phases evaluated" only ever happens strictly inside the +-blend_half_width_hours window, not
# some wider region -- with the defaults (95.5+-45.3h, ~46 decisions/batch, T_SAMPLING=5h) that's
# ~18 of 46 steps/rollout paying for both phases instead of all 46. (A much tighter EPS, e.g.
# 1e-4, would roughly double that window instead -- measured, not hypothetical: it roughly
# doubled the extra compute cost before this was caught.)
_BLEND_SKIP_EPS = 0.01


class DualPhaseModelLearning(torch.nn.Module):

    def __init__(self, pivot_step, pivot_hours, phase1_par, phase2_par,
                 blend_half_width_hours=BLEND_HALF_WIDTH_HOURS,
                 phase_model_cls=Model_learning_RBF_det_time,
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.pivot_step = pivot_step
        self.pivot_hours = pivot_hours
        self.blend_half_width_hours = blend_half_width_hours
        self.dtype = dtype
        self.device = device
        # phase_model_cls defaults to the deployed per-channel-prior-mean model, so every existing
        # config_dual_phase.py call site is unaffected; config_dual_phase_baseline.py is the only
        # caller that overrides it (to Model_learning_RBF_baseline), for the same plain-RBF
        # ablation config_single_phase_baseline.py already runs for single-phase -- see
        # model_learning_baseline.py's docstring for what this removes and why.
        self.phase1 = phase_model_cls(**phase1_par)
        self.phase2 = phase_model_cls(**phase2_par)
        self._t = 0

    @property
    def num_gp(self):
        return self.phase1.num_gp + self.phase2.num_gp

    @property
    def gp_list(self):
        return list(self.phase1.gp_list) + list(self.phase2.gp_list)

    @property
    def gp_inputs(self):
        return torch.cat([self.phase1.gp_inputs, self.phase2.gp_inputs], 0)

    @property
    def gp_output_list(self):
        return list(self.phase1.gp_output_list) + list(self.phase2.gp_output_list)

    @property
    def norm_list(self):
        # Read directly by MC_PILCO.get_model_learning_performance (self.model_learning.norm_list[i]).
        return list(self.phase1.norm_list) + list(self.phase2.norm_list)

    def reset_step_counter(self, start_t=0):
        """Reset the decision-step counter used by get_next_state. Must be called before
        every rollout (apply_policy / diagnostic rollout()) that will call get_next_state
        sequentially in decision order -- see PenSimMCPILCOMultiPhase."""
        self._t = start_t

    def _blend_weight(self, t_step):
        """Sigmoid blend weight at decision step t_step: ~0 well before pivot_hours (phase1
        dominates), ~1 well after (phase2 dominates), 0.5 exactly at pivot_hours.

        k is set so w is ~0.01 at (pivot_hours - blend_half_width_hours) and ~0.99 at
        (pivot_hours + blend_half_width_hours) -- the standard "1%/99%" convention for a
        logistic's effective width, giving e.g. pivot_hours=90/blend_half_width_hours=40 =>
        w~0.01 at 50h, w=0.5 at 90h, w~0.99 at 130h."""
        t_hours = t_step * T_SAMPLING
        k = math.log(99.0) / self.blend_half_width_hours
        return 1.0 / (1.0 + math.exp(-k * (t_hours - self.pivot_hours)))

    def add_data(self, new_state_samples, new_input_samples):
        """Split one trajectory at pivot_step and route each segment to its phase's own
        add_data. states[0..pivot_step] (inclusive) go to phase 1, so the t=pivot_step state
        is phase 1's last input row AND phase 2's first input row -- every transition
        (t, t+1) is owned by exactly one phase, none dropped or duplicated. Training stays
        HARD-split even though predictions now blend smoothly -- see module docstring."""
        p = self.pivot_step
        self.phase1.add_data(new_state_samples[:p + 1], new_input_samples[:p + 1])
        self.phase2.add_data(new_state_samples[p:], new_input_samples[p:])

    def reinforce_model(self, optimization_opt_list=None):
        n1 = self.phase1.num_gp
        self.phase1.reinforce_model(optimization_opt_list=optimization_opt_list[:n1])
        self.phase2.reinforce_model(optimization_opt_list=optimization_opt_list[n1:])

    def set_eval_mode(self):
        self.phase1.set_eval_mode()
        self.phase2.set_eval_mode()

    def set_training_mode(self):
        self.phase1.set_training_mode()
        self.phase2.set_training_mode()

    def get_next_state(self, current_state, current_input, particle_pred=True):
        """Blends phase1/phase2 predictions via the sigmoid weight at the current decision
        step (see _blend_weight), based on an internal decision-step counter incremented once
        per call. Callers (MC_PILCO.apply_policy, MC_PILCO.rollout) invoke this exactly once
        per decision step, strictly in increasing order, within one rollout -- so the counter
        (reset via reset_step_counter before each rollout) tracks absolute decision time
        correctly.

        Skips evaluating whichever phase the weight has collapsed onto (see _BLEND_SKIP_EPS)
        rather than always paying for both. When both are evaluated, next_states/delta_mean
        blend as a plain weighted sum (matching y=(1-w)y1+w*y2 applied every step); delta_var
        uses the MIXTURE (law-of-total-variance) formula
        Var = (1-w)*Var1 + w*Var2 + (1-w)*w*(mean1-mean2)^2 -- correct for "one true regime
        governs, we just don't know exactly when it switched for this rollout", which is what
        w(t) actually represents here. The previously-used weighted-SUM-of-independent-Gaussians
        formula, (1-w)^2*Var1 + w^2*Var2, silently halves the reported variance right at w=0.5
        (where phase1/phase2 disagree the most) and omits the disagreement term entirely --
        confirmed as the source of measured one-step calibration under-coverage, not just a
        theoretical concern. That formula would only be correct if both phases' predictions were
        literally, simultaneously summed -- they aren't; exactly one phase is ever "true" at a
        given step, we're blending our belief about which.

        No special-casing needed for the deterministic `time` channel (both phases compute the
        identical exact-delta formula, so blending two identical values is a no-op) or for
        state_clamp (each phase already clamps its own output to [-1,1]; a convex combination
        of two already-clamped tensors stays in [-1,1] automatically)."""
        t = self._t
        self._t += 1
        w = self._blend_weight(t)

        if w <= _BLEND_SKIP_EPS:
            return self.phase1.get_next_state(current_state, current_input, particle_pred=particle_pred)
        if w >= 1.0 - _BLEND_SKIP_EPS:
            return self.phase2.get_next_state(current_state, current_input, particle_pred=particle_pred)

        next1, mean1, var1 = self.phase1.get_next_state(current_state, current_input, particle_pred=particle_pred)
        next2, mean2, var2 = self.phase2.get_next_state(current_state, current_input, particle_pred=particle_pred)
        next_states = (1.0 - w) * next1 + w * next2
        delta_mean = (1.0 - w) * mean1 + w * mean2
        delta_var = ((1.0 - w) * var1 + w * var2
                     + (1.0 - w) * w * (mean1 - mean2) ** 2)
        return next_states, delta_mean, delta_var

    def get_gp_estimate_from_data(self, states, inputs, flg_pretrain=False, gp_index_list=None,
                                  flg_onestep=False):
        """Diagnostic passthrough (used by get_model_learning_performance): split the
        trajectory at pivot_step exactly like add_data, evaluate each phase's own GPs on its
        own segment, and concatenate the 4 returned lists in [phase1 x num_gp, phase2 x
        num_gp] order -- consistent with gp_list's ordering. Deliberately NOT blended: this
        scores each phase's own fit to its own (still hard-split) training data, independent
        of the rollout-time blend."""
        p = self.pivot_step
        gp_inputs_1, gp_out_1, mean_1, var_1 = self.phase1.get_gp_estimate_from_data(
            states=states[:p + 1], inputs=inputs[:p + 1],
            flg_pretrain=flg_pretrain, flg_onestep=flg_onestep)
        gp_inputs_2, gp_out_2, mean_2, var_2 = self.phase2.get_gp_estimate_from_data(
            states=states[p:], inputs=inputs[p:],
            flg_pretrain=flg_pretrain, flg_onestep=flg_onestep)
        gp_inputs = torch.cat([gp_inputs_1, gp_inputs_2], 0)
        gp_outputs_target_list = None if gp_out_1 is None else list(gp_out_1) + list(gp_out_2)
        gp_output_mean_list = list(mean_1) + list(mean_2)
        gp_output_var_list = list(var_1) + list(var_2)
        return gp_inputs, gp_outputs_target_list, gp_output_mean_list, gp_output_var_list

    def to(self, device):
        super().to(device)
        self.device = device
        self.phase1.to(device)
        self.phase2.to(device)
        return self

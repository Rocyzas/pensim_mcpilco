"""Composite model-learning object for dual-phase MC-PILCO: routes to one of two independent
Model_learning_RBF_det_time instances (phase 1: t < pivot_step, phase 2: t >= pivot_step)
instead of learning a single set of GPs over the whole batch. See
pensim_wrapper.PenSimMCPILCOMultiPhase for the agent that drives this.

This is a thin router, not a new GP implementation: both phases reuse
Model_learning_RBF_det_time unmodified (Wt mass-balance prior, Viscosity recipe-mean prior,
deterministic time channel, SOD approximation, state clamp -- see model_learning_det_time.py),
so all of that per-channel logic stays defined in exactly one place. `num_gp`/`gp_list`/
`gp_inputs`/`gp_output_list` are exposed as concatenations of the two sub-models' own
(phase1 first, phase2 second) so that MC_PILCO.reinforce()'s generic log/bookkeeping code
(which indexes `model_learning.gp_list[k]` for k in range(model_learning.num_gp), etc.) keeps
working unmodified against a `2*STATE_DIM`-GP composite.
"""

import torch

from mcpilco.model_learning_det_time import Model_learning_RBF_det_time


class DualPhaseModelLearning(torch.nn.Module):

    def __init__(self, pivot_step, phase1_par, phase2_par, dtype=torch.float64,
                 device=torch.device("cpu")):
        super().__init__()
        self.pivot_step = pivot_step
        self.dtype = dtype
        self.device = device
        self.phase1 = Model_learning_RBF_det_time(**phase1_par)
        self.phase2 = Model_learning_RBF_det_time(**phase2_par)
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

    def add_data(self, new_state_samples, new_input_samples):
        """Split one trajectory at pivot_step and route each segment to its phase's own
        add_data. states[0..pivot_step] (inclusive) go to phase 1, so the t=pivot_step state
        is phase 1's last input row AND phase 2's first input row -- every transition
        (t, t+1) is owned by exactly one phase, none dropped or duplicated."""
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
        """Routes to phase1/phase2 based on an internal decision-step counter, incremented
        once per call. Callers (MC_PILCO.apply_policy, MC_PILCO.rollout) invoke this exactly
        once per decision step, strictly in increasing order, within one rollout -- so the
        counter (reset via reset_step_counter before each rollout) tracks absolute decision
        time correctly."""
        t = self._t
        self._t += 1
        model = self.phase1 if t < self.pivot_step else self.phase2
        return model.get_next_state(current_state, current_input, particle_pred=particle_pred)

    def get_gp_estimate_from_data(self, states, inputs, flg_pretrain=False, gp_index_list=None,
                                  flg_onestep=False):
        """Diagnostic passthrough (used by get_model_learning_performance): split the
        trajectory at pivot_step exactly like add_data, evaluate each phase's own GPs on its
        own segment, and concatenate the 4 returned lists in [phase1 x num_gp, phase2 x
        num_gp] order -- consistent with gp_list's ordering."""
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

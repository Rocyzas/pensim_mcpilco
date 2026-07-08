'''
Dense reward implemented

Todo: reward shaping
'''

import torch
import policy_learning.Cost_function as CF

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES,
                                    WT_SOFT, PAA_BAND, DO2_FLOOR)

P_IDX = STATE_NAMES.index("P")
WT_IDX = STATE_NAMES.index("Wt")
PAA_IDX = STATE_NAMES.index("PAA")
DO2_IDX = STATE_NAMES.index("DO2")


class PeniConcentrationCost(CF.Expected_cost):
    def __init__(self, p_weight=0.05, soft_penalty=0.5, paa_penalty=100.0,
                 do2_penalty=5.0, rate_penalty=0.5):
        self.p_weight = p_weight          # scales P (g/L) reward to ~O(1) per step
        self.soft_penalty = soft_penalty  # Wt outside WT_SOFT band
        self.paa_penalty = paa_penalty    # PAA outside PAA_BAND
        self.do2_penalty = do2_penalty    # DO2 below DO2_FLOOR (Fs overfeed crashes O2)
        self.rate_penalty = rate_penalty  # penalise fast action changes
        super().__init__(cost_function=self._cost)

    def _dn(self, x_norm, lo, hi):
        """Denormalise a [-1, 1] state channel back to physical units."""
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    @staticmethod
    def _outside(x, lo, hi, scale):
        """Smooth squared penalty for x out of bounds."""
        return torch.relu((lo - x) / scale) ** 2 + torch.relu((x - hi) / scale) ** 2

    def _cost(self, states_sequence, inputs_sequence, trial_index=None):
        # states_sequence: [T, num_particles, state_dim]
        P = self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"])       # g/L
        Wt = self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"])    # kg
        PAA = self._dn(states_sequence[:, :, PAA_IDX], *STATE_RANGES["PAA"]) # mg/L
        DO2 = self._dn(states_sequence[:, :, DO2_IDX], *STATE_RANGES["DO2"]) # mg/L

        reward = self.p_weight * P

        # Wt guardrail: OVERFLOW side only
        soft = self.soft_penalty * torch.relu((Wt - WT_SOFT[1]) / 1e4) ** 2
        paa_soft = self.paa_penalty * self._outside(PAA, *PAA_BAND, 1e3)
        do2_soft = self.do2_penalty * torch.relu((DO2_FLOOR - DO2) / DO2_FLOOR) ** 2

        # penalise how fast the action moves (first step has no predecessor -> 0)
        u = inputs_sequence[:, :, 0]
        action_rate = torch.zeros_like(u)
        action_rate[1:] = self.rate_penalty * (u[1:] - u[:-1]) ** 2

        return -reward + soft + paa_soft + do2_soft + action_rate  # minimise
'''
Dense reward implemented

Todo: reward shaping
'''

import torch
import policy_learning.Cost_function as CF

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES,
                                    WT_SOFT, PAA_BAND, DO2_FLOOR,
                                    decode_state_value)

P_IDX = STATE_NAMES.index("P")
WT_IDX = STATE_NAMES.index("Wt")
PAA_IDX = STATE_NAMES.index("PAA")
DO2_IDX = STATE_NAMES.index("DO2")


class PeniConcentrationCost(CF.Expected_cost):
    def __init__(self, p_weight=0.05, soft_penalty=0.5, paa_penalty=100.0,
                 do2_penalty=5.0, rate_penalty=0.5):
        self.p_weight = p_weight
        self.soft_penalty = soft_penalty
        self.paa_penalty = paa_penalty
        self.do2_penalty = do2_penalty
        self.rate_penalty = rate_penalty
        super().__init__(cost_function=self._cost)


    def _dn(self, x_norm, lo, hi):
        """Denormalise a [-1, 1] state channel back to physical units."""
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    @staticmethod
    def _outside(x, lo, hi, scale):
        """Smooth squared penalty for x out of bounds."""
        return torch.relu((lo - x) / scale) ** 2 + torch.relu((x - hi) / scale) ** 2

    def _cost(self, states_sequence, inputs_sequence, trial_index=None):
        P = decode_state_value("P", self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]))
        Wt = decode_state_value("Wt", self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]))
        PAA = self._dn(states_sequence[:, :, PAA_IDX], *STATE_RANGES["PAA"])
        DO2 = self._dn(states_sequence[:, :, DO2_IDX], *STATE_RANGES["DO2"])

        # ORIGINAL
        # reward = self.p_weight * P

        # SEED2_6 run
        # mass = P * Wt / 1000.0          # ≈ kg penicillin in the tank (P g/L × Wt L)
        # reward = self.p_weight * mass    # was: self.p_weight * P

        # SEED2_7 run TODO this run because it got stuck
        mass = P * Wt / 1000.0                    # [T, particles]
        dmass = torch.zeros_like(mass)
        dmass[1:] = mass[1:] - mass[:-1]
        reward = self.p_weight * dmass

        soft = self.soft_penalty * torch.relu((Wt - WT_SOFT[1]) / 1e4) ** 2
        paa_soft = self.paa_penalty * self._outside(PAA, *PAA_BAND, 1e3)
        do2_soft = self.do2_penalty * torch.relu((DO2_FLOOR - DO2) / DO2_FLOOR) ** 2

        u = inputs_sequence[:, :, 0]
        action_rate = torch.zeros_like(u)
        action_rate[1:] = self.rate_penalty * (u[1:] - u[:-1]) ** 2

        # return -reward + soft + paa_soft + do2_soft + action_rate
        return -reward + soft + do2_soft + action_rate
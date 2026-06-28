"""
Reward / cost for the single-phase MC-PILCO baseline

Reward = penicillin concentration P (maximise)
Cost (minimised) = -p_weight * P (per step)
"""
import torch
import policy_learning.Cost_function as CF

from mcpilco.pensim_wrapper import (STATE_NAMES,
                                    STATE_RANGES,
                                    WT_SOFT,
                                    WT_OVERFLOW,
                                    P_CRASH,
                                    PAA_BAND)

P_IDX = STATE_NAMES.index("P")
WT_IDX = STATE_NAMES.index("Wt")
PAA_IDX = STATE_NAMES.index("PAA")


class PeniConcentrationCost(CF.Expected_cost):
    def __init__(self, p_weight=0.05, hard_penalty=50.0, soft_penalty=0.5, paa_penalty=100, rate_penalty=0.5):
        self.p_weight = p_weight            # scales P (g/L) reward to O(1) per step
        self.hard_penalty = hard_penalty    # fixed
        self.soft_penalty = soft_penalty
        self.paa_penalty = paa_penalty
        self.rate_penalty = rate_penalty
        super().__init__(cost_function=self._cost)

    def _dn(self, x_norm, lo, hi):
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    def _cost(self, states_sequence, inputs_sequence, trial_index=None):
        # states_sequence: [T, num_particles, state_dim]
        P = self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]) # g/L
        Wt = self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]) # kg
        PAA = self._dn(states_sequence[:, :, PAA_IDX], *STATE_RANGES["PAA"]) # mg/L

        # maximise P
        reward = self.p_weight * P

        # HARD fixed penalties: vessel overflow or penicillin blow-up.
        hard = self.hard_penalty * ((Wt > WT_OVERFLOW).to(P.dtype) + (P > P_CRASH).to(P.dtype))
        # SOFT proportional penalty: outside the IndPenSim weight band.
        lo, hi = WT_SOFT
        # penalizes where the state is;
        soft = self.soft_penalty * (torch.relu((lo - Wt) / 1e4) + torch.relu((Wt - hi) / 1e4))
        # SOFT proportional penalty: outside the PAA concentration band.
        paa_lo, paa_hi = PAA_BAND

        paa_dev_lo = torch.relu((paa_lo - PAA) / 1e3)
        paa_dev_hi = torch.relu((PAA - paa_hi) / 1e3)
        paa_soft = self.paa_penalty * (paa_dev_lo**2 + paa_dev_hi**2)
        # paa_soft = self.paa_penalty * (paa_dev_lo + paa_dev_hi)
        # paa_soft = self.paa_penalty * (torch.relu((paa_lo - PAA) / 1e3) + torch.relu((PAA - paa_hi) / 1e3))

        #penalizes how fast the action moves
        u = inputs_sequence[:, :, 0]
        du = u[1:] - u[:-1]
        action_rate = torch.zeros_like(u)
        action_rate[1:] = self.rate_penalty * du**2  # first step has no predecessor -> 0

        # Minimise
        return -reward + hard + soft + paa_soft + action_rate
        
        # return -reward

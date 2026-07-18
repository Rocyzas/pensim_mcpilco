'''
Dense reward implemented

Todo: reward shaping
'''

import torch
import policy_learning.Cost_function as CF

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES,
                                    WT_SOFT, VISC_MAX, T_SAMPLING, K_WARM,
                                    decode_state_value)
from utils.constants import STEP_IN_HOURS

P_IDX = STATE_NAMES.index("P")
WT_IDX = STATE_NAMES.index("Wt")
VISC_IDX = STATE_NAMES.index("Viscosity")
TIME_IDX = STATE_NAMES.index("time")

# Viscosity soft penalty. Onset at VISC_MAX (100 cP) with a 50 cP scale, mirroring the WT_SOFT
# construction: quadratic, ~0 just above the threshold and growing toward the collapse region.
# Calibration from real batches: healthy ones peak 79-111 cP, collapsed ones 158-184 cP, so at the
# observed failure point (~160 cP) the term is (60/50)^2 = 1.4x its weight per decision.
VISC_SOFT_SCALE = 50.0

_disch_int_cache = {}


def _decision_discharge_L(dtype, device):
    """Litres of broth discharged in each decision window, indexed by decision.

    Reuses wt_mass_balance's integrator, which walks the SAME integer sim-step grid as
    PenSimWrapper.rollout. That matters: the discharge pulses are ONE sim-step wide, so sampling the
    recipe on a decision-time grid misses them entirely. Imported lazily -- building it rolls the
    recipe, and this module is imported during config construction.
    """
    key = (dtype, str(device))
    if key not in _disch_int_cache:
        from mcpilco.wt_mass_balance import _DISCH_INT
        _disch_int_cache[key] = torch.tensor(_DISCH_INT, dtype=dtype, device=device)
    return _disch_int_cache[key]

# Physical ceilings for the reward-relevant decoded channels, taken from the model's own declared
# domain (STATE_RANGES upper bound, decoded out of log space). The rollout is already clamped to
# [-1, 1] in Model_learning_RBF_det_time, which bounds these; clamping again here is defence in
# depth so the cost can never read an impossible P/Wt (and thus never explode) even if a caller
# rolls an unclamped model. P: exp(log 40) = 40 g/L, Wt: exp(log 1.3e5) = 1.3e5 L.
P_MAX = float(decode_state_value("P", STATE_RANGES["P"][1]))
WT_MAX = float(decode_state_value("Wt", STATE_RANGES["Wt"][1]))


class PeniConcentrationCost(CF.Expected_cost):
    def __init__(self, p_weight=None, soft_penalty=None, rate_penalty=None, risk_weight=0.0,
                 visc_penalty=0.5, harvest_reward=True):
        self.p_weight = p_weight
        self.soft_penalty = soft_penalty
        self.rate_penalty = rate_penalty
        # Quadratic penalty weight on broth viscosity above VISC_MAX -- the observed collapse mode.
        self.visc_penalty = visc_penalty
        # Credit penicillin removed by the recipe's discharge pulses. Defaults ON because it makes
        # the objective match batch_yield_kg; set False to reproduce the old in-tank-only reward.
        self.harvest_reward = harvest_reward
        # Risk aversion: weight on the across-particle spread in the OPTIMISED objective.
        # 0.0 reproduces the stock risk-neutral expectation exactly. See `forward` below.
        self.risk_weight = risk_weight # THIS rewards small std
        super().__init__(cost_function=self._cost)

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """Risk-sensitive objective: sum_t mean_p(cost) + risk_weight * sum_t std_p(cost).

        WHY OVERRIDE THIS
        -----------------
        The stock `Expected_cost.forward` minimises the mean over particles. Measured on this
        system the across-particle std runs 19-29x the mean cost (R.1b in evaluations/Rollouts.ipynb),
        so the mean is a weak summary of the imagined outcome: a policy can look good on average
        while a large share of particles predict a collapsed batch -- and collapsed batches do show
        up in the real episodes. Penalising the spread makes the optimiser prefer policies whose
        outcome the model can actually forecast.

        THE DETACH, WHICH IS THE WHOLE POINT
        ------------------------------------
        The base class computes `torch.std(costs.detach(), 1)`, so its std is a logging quantity
        with NO gradient path. Reusing it here would move the reported cost while leaving the policy
        gradient untouched -- a silent no-op. The penalty term below is therefore computed from the
        NON-detached costs. The second return value stays detached, so `std_cost_trial_list` and the
        R.1b diagnostic keep exactly their previous meaning.
        """
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)
        objective = torch.sum(mean_costs)
        if self.risk_weight:
            objective = objective + self.risk_weight * torch.sum(torch.std(costs, 1))
        return objective, torch.sum(torch.std(costs.detach(), 1))


    def _dn(self, x_norm, lo, hi):
        """Denormalise a [-1, 1] state channel back to physical units."""
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    def _cost(self, states_sequence, inputs_sequence, trial_index=None):
        P = decode_state_value("P", self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]))
        Wt = decode_state_value("Wt", self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]))
        # Cap the decoded quantities at their physical ceiling so an off-distribution GP prediction
        # cannot inflate the mass reward (P * Wt) to an impossible magnitude. No-op in the normal
        # operating range (P ~ 0-40 g/L); only bites the hallucinated tail.
        P = torch.clamp(P, min=0.0, max=P_MAX)
        Wt = torch.clamp(Wt, min=0.0, max=WT_MAX)

        # ORIGINAL
        # reward = self.p_weight * P

        # SEED2_6 run
        # mass = P * Wt / 1000.0          # ≈ kg penicillin in the tank (P g/L × Wt L)
        # reward = self.p_weight * mass    # was: self.p_weight * P

        # SEED2_7 run TODO this run because it got stuck
        mass = P * Wt / 1000.0                    # [T, particles]
        dmass = torch.zeros_like(mass)
        dmass[1:] = mass[1:] - mass[:-1]

        # HARVESTED product. `mass` is only what is left IN THE TANK, so summing dmass telescopes to
        # (final - initial) in-tank mass and silently discards everything the recipe's discharge
        # pulses drew off. Measured on real episodes that gap is ~800 kg -- about 20% of the yield
        # actually reported by batch_yield_kg. Worse, a discharge DROPS Wt, so dmass goes negative
        # and the optimiser was being PENALISED for the six harvest events, i.e. for the moments the
        # process collects product. Crediting P * (litres discharged) realigns the objective with the
        # number every baseline is scored on.
        if self.harvest_reward:
            t_h = self._dn(states_sequence[:, :, TIME_IDX], *STATE_RANGES["time"])
            disch_L = _decision_discharge_L(P.dtype, P.device)
            # Decision index from the CLOCK, not the sequence position: with setup_recipe_anchors
            # particles launch from different batch times, so position is an offset, not a decision.
            idx = torch.round((t_h - K_WARM * STEP_IN_HOURS) / T_SAMPLING).long()
            idx = idx.clamp(0, disch_L.shape[0] - 1)
            dmass = dmass + P * disch_L[idx] / 1000.0

        reward = self.p_weight * dmass

        soft = self.soft_penalty * torch.relu((Wt - WT_SOFT[1]) / 1e4) ** 2

        # Viscosity guardrail. Adding Viscosity to the state lets the GP SEE the collapse mechanism;
        # this term gives the optimiser a reason to stay away from it rather than discovering the
        # cost only after the broth has already thickened and product has started degrading.
        visc = self._dn(states_sequence[:, :, VISC_IDX], *STATE_RANGES["Viscosity"])
        visc_soft = self.visc_penalty * torch.relu((visc - VISC_MAX) / VISC_SOFT_SCALE) ** 2

        u = inputs_sequence[:, :, 0]
        action_rate = torch.zeros_like(u)
        action_rate[1:] = self.rate_penalty * (u[1:] - u[:-1]) ** 2

        return -reward + soft + visc_soft + action_rate
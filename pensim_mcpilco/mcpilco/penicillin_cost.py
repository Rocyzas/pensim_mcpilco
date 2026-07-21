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

P_MAX = float(decode_state_value("P", STATE_RANGES["P"][1]))
WT_MAX = float(decode_state_value("Wt", STATE_RANGES["Wt"][1]))


def _mass_change_reward(cost, states_sequence, P, Wt):
    """dmass = first difference of mass=P*Wt/1000 (dmass[0] = 0), plus a discharge/harvest credit
    whenever cost.harvest_reward is True.

    THE single implementation of this formula. Both `PeniConcentrationCost` (the base/deployed
    class, see its `_terms`) and `PeniMassChangeCost` below call this SAME function rather than
    each having their own copy -- that is what makes "deployed formula" and "tested formula"
    structurally impossible to drift apart, instead of relying on someone noticing a comment went
    stale (which is exactly how the base class ended up silently computing this formula WITHOUT
    the discharge credit for a while: the credit's code existed only as a commented-out block that
    a doc comment claimed was inert, and both statements quietly stopped being true).
    """
    mass = P * Wt / 1000.0
    dmass = torch.zeros_like(mass)
    dmass[1:] = mass[1:] - mass[:-1]

    if cost.harvest_reward:
        t_h = cost._dn(states_sequence[:, :, TIME_IDX], *STATE_RANGES["time"])
        disch_L = _decision_discharge_L(P.dtype, P.device)
        # Decision index from the CLOCK, not the sequence position: with setup_recipe_anchors
        # particles launch from different batch times, so position is an offset, not a decision.
        idx = torch.round((t_h - K_WARM * STEP_IN_HOURS) / T_SAMPLING).long()
        idx = idx.clamp(0, disch_L.shape[0] - 1)
        dmass = dmass + P * disch_L[idx] / 1000.0

    return dmass


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
        """Total per-(step, particle) cost. Thin sum over `_terms` -- keep the arithmetic there so
        a diagnostic that inspects the terms individually reads the SAME code the optimiser runs.
        (A diagnostic that re-implements these formulas drifts silently the moment one is edited.)"""
        t = self._terms(states_sequence, inputs_sequence, trial_index)
        return -t["reward"] + t["soft"] + t["visc_soft"] + t["action_rate"]

    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        """The cost's individual terms, unsummed, each shaped [T, particles].

        Signs are as NAMED, not as combined: `reward` is a reward (higher is better) and the three
        penalties are costs. `_cost` applies the signs. Divide any term by `p_weight` to read it in
        kilograms of penicillin, which is what makes the weights comparable -- see
        experiments/cost_term_report.py.
        """
        P = decode_state_value("P", self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]))
        Wt = decode_state_value("Wt", self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]))

        P = torch.clamp(P, min=0.0, max=P_MAX)
        Wt = torch.clamp(Wt, min=0.0, max=WT_MAX)

        # DEPLOYED reward: dense mass-change, with the discharge/harvest credit applied whenever
        # harvest_reward=True (the default). Delegates to `_mass_change_reward` -- the SAME
        # function `PeniMassChangeCost` below calls -- so this base class's default is always
        # identical to that already-validated candidate (experiments/cost_reward_hacking_bo.py's
        # "mass_change_discharge" arm), never a fifth, untested formula. See that function's
        # docstring for why this indirection exists instead of the formula living inline here.
        #
        # To deploy a DIFFERENT one of the 4 validated candidates instead, point
        # mcpilco/config_single_phase.py's f_cost_function at PeniConcentrationDenseCost /
        # PeniConcentrationChangeCost / PeniMassTerminalCost directly, rather than editing this
        # method -- that keeps "what's deployed" and "what's tested" the same class by
        # construction, instead of by manually toggling which formula is commented out here.
        reward = self.p_weight * _mass_change_reward(self, states_sequence, P, Wt)

        # DISABLED penalties. All three are returned as zeros so `_cost` and
        # experiments/cost_term_report.py keep their four-term contract -- restore any one by
        # commenting the zeros line and uncommenting the formula beneath it.

        # soft: measured inert. Max Wt is 106,535 at full overfeed (a=+1, seed 700000), below
        # the WT_SOFT[1]=1.1e5 threshold, so this never fired on any reachable trajectory.
        # soft = torch.zeros_like(P)
        soft = self.soft_penalty * torch.relu((Wt - WT_SOFT[1]) / 1e4) ** 2

        # visc_soft: RESTORE THIS FIRST if collapsed batches return. The seed11-14 ablation put
        # the collapse rate at 9/44 (20%) with this off vs 5/44 (11%) with visc_penalty=0.5.
        # That difference is not significant at n=4 seeds (Fisher p~0.4), which is why it is
        # being tested -- but it is the only penalty with evidence behind it.
        # visc_soft = torch.zeros_like(P)
        visc = self._dn(states_sequence[:, :, VISC_IDX], *STATE_RANGES["Viscosity"])
        visc_soft = self.visc_penalty * torch.relu((visc - VISC_MAX) / VISC_SOFT_SCALE) ** 2

        # action_rate: never measured. cost_term_report.py reports it INERT, but that is an
        # artifact of its constant-action probes, not evidence. Dropping it permits chattering Fs.
        u = inputs_sequence[:, :, 0]
        action_rate = torch.zeros_like(u)
        # action_rate[1:] = self.rate_penalty * (u[1:] - u[:-1]) ** 2

        return {"reward": reward, "soft": soft, "visc_soft": visc_soft, "action_rate": action_rate}


def _decode_P_Wt(cost, states_sequence):
    """Shared P/Wt decode+clamp, factored out so the 4 variants below don't each hand-copy it.
    Same lines `_terms` above and `MassCost` (experiments/cost_offmanifold_probe.py) already use."""
    P = decode_state_value("P", cost._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]))
    Wt = decode_state_value("Wt", cost._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]))
    return torch.clamp(P, min=0.0, max=P_MAX), torch.clamp(Wt, min=0.0, max=WT_MAX)




# OK
class PeniConcentrationDenseCost(PeniConcentrationCost):
    """reward = p_weight * P, every step. Restores the currently-disabled line-117 formula in
    the base class as its own always-instantiable candidate, so it can be compared against the
    other 3 in the same run instead of needing a comment/uncomment swap in the shared file."""
    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        t = super()._terms(states_sequence, inputs_sequence, trial_index)
        P, _ = _decode_P_Wt(self, states_sequence)
        return {**t, "reward": self.p_weight * P}


# THIS DID NOT WORK
class PeniConcentrationChangeCost(PeniConcentrationCost):
    """reward_t = p_weight * (P_t - P_{t-1}), reward[0] = 0. Dense, like PeniConcentrationDenseCost,
    but rewards the RATE of concentration increase rather than its level. No discharge correction:
    discharge removes broth volume, not concentration directly, so it isn't a conserved quantity
    the way mass is -- there's no equivalent "lost on harvest" credit to add back here."""
    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        t = super()._terms(states_sequence, inputs_sequence, trial_index)
        P, _ = _decode_P_Wt(self, states_sequence)
        dP = torch.zeros_like(P)
        dP[1:] = P[1:] - P[:-1]
        return {**t, "reward": self.p_weight * dP}


# THIS DID NOT WORK
class PeniMassTerminalCost(PeniConcentrationCost):
    """reward = 0 everywhere except the last step, where reward = p_weight * P_T * Wt_T / 1000.
    Deliberately NAIVE (no discharge credit) -- this is the textbook sparse terminal-mass formula
    that ignores product already harvested via discharge pulses, kept naive on purpose so it can
    demonstrate that exact failure mode as a contrast against PeniMassChangeCost(harvest_reward=True)
    below, which is the harvest-aware fix for it."""
    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        t = super()._terms(states_sequence, inputs_sequence, trial_index)
        P, Wt = _decode_P_Wt(self, states_sequence)
        mass = P * Wt / 1000.0
        reward = torch.zeros_like(mass)
        reward[-1] = mass[-1]
        return {**t, "reward": self.p_weight * reward}

# OK
class PeniMassChangeCost(PeniConcentrationCost):
    """reward_t = p_weight * (dmass_t + discharge_credit_t if harvest_reward else dmass_t), where
    mass = P*Wt/1000 and dmass is its first difference (dmass[0] = 0). With harvest_reward=True
    (the default), every discharge pulse's removed penicillin mass is credited back, so dmass no
    longer treats harvesting product as a loss -- this is "dense mass change accounting for
    discharge." With harvest_reward=False it reduces to plain dmass.

    Calls the SAME `_mass_change_reward` the base class's own `_terms` delegates to (single
    implementation -- see that function's docstring). This class exists as a STABLE, explicitly-
    named reference to that formula, independent of whatever the base class is later repointed at:
    experiments/cost_reward_hacking_bo.py imports THIS class by name for its "mass_change_discharge"
    arm precisely so that arm can't silently drift out from under the test the way the base class's
    default once did."""
    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        t = super()._terms(states_sequence, inputs_sequence, trial_index)
        P, Wt = _decode_P_Wt(self, states_sequence)
        return {**t, "reward": self.p_weight * _mass_change_reward(self, states_sequence, P, Wt)}
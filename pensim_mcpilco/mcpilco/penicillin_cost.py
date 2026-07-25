'''
Dense reward implemented

Todo: reward shaping
'''

import torch
import policy_learning.Cost_function as CF

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES,
                                    WT_SOFT, WT_OVERFLOW, VISC_MAX,
                                    T_SAMPLING, K_WARM,
                                    decode_state_value)
from utils.constants import STEP_IN_HOURS

P_IDX = STATE_NAMES.index("P")
WT_IDX = STATE_NAMES.index("Wt")
VISC_IDX = STATE_NAMES.index("Viscosity")
TIME_IDX = STATE_NAMES.index("time")

VISC_SOFT_SCALE = 50.0
# Penalty starts ramping BEFORE the hard VISC_MAX=100 envelope ceiling, so the optimiser gets
# gradient signal while a policy is still approaching the collapse regime rather than only once
# it has already crossed it.
VISC_SOFT_START = 80.0

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


def _mass_kg(P, Wt):
    """Penicillin mass held in the tank, kg. P is g/L, Wt is broth volume in L.

    THE single conversion from (concentration, volume) to a mass-in-kg currency. Used both for the
    dense reward (`_mass_change_reward` below) and, since the cost refactor, to PRICE every
    constraint penalty in the same currency: a weight-overflow or viscosity-collapse event is
    priced as the fraction of this in-tank mass that event puts at risk (spilled on overflow, or
    lost to a batch abort on collapse) -- see `_ramp_severity` and `PeniConcentrationCost._terms`.
    """
    return P * Wt / 1000.0


def _ramp_severity(value, soft_start, hard_limit):
    """Dimensionless violation severity: 0 at/below `soft_start`, growing quadratically to 1 exactly
    at `hard_limit`, unbounded above it. This is deliberately just the "how bad is it" ramp, kept
    separate from `_mass_kg` (the "how much is at stake" pricing) so the two can be recombined
    differently later -- e.g. a chance-constraint formulation would replace this smooth ramp with
    the fraction of particles past `hard_limit`, while still multiplying by the SAME `_mass_kg`
    pricing, without touching how the kg currency itself is defined.
    """
    span = hard_limit - soft_start
    return torch.relu((value - soft_start) / span) ** 2


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
    mass = _mass_kg(P, Wt)
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
                 visc_penalty=0.5, harvest_reward=True, constraint_strength=1.0):
        """
        Every penalty below is priced in kg-of-penicillin-equivalent (via `_mass_kg` /
        `_ramp_severity`) BEFORE its lambda is applied, so a lambda of 1 means "trade 1 kg of yield
        to avoid 1 kg-equivalent of this violation" -- see `_terms` for the arithmetic. The
        parameter names are kept from the pre-refactor API for backward compatibility (existing
        configs construct this class by keyword); only their MEANING changed, from raw unit-
        conversion factors to interpretable per-term lambdas:

          p_weight      -- kg -> cost-unit conversion for the reward AND every penalty (shared,
                            so all terms stay in one currency; see `_terms`).
          soft_penalty  -- lambda_weight: kg traded per kg-equivalent of tank-overflow risk.
          visc_penalty  -- lambda_visc:   kg traded per kg-equivalent of viscosity-collapse risk.
          rate_penalty  -- lambda_rate:   kg traded per kg-equivalent of action chatter. A
                            smoothness preference, not a safety constraint, so it is NOT scaled by
                            `constraint_strength` (see that parameter below).
          risk_weight   -- lambda_risk:   kg traded per kg-equivalent of batch-OUTCOME spread
                            (std across particles of the summed per-trajectory cost). Previously
                            this was added outside the r-lambda*c scheme and measured PER-TIMESTEP
                            particle spread rather than outcome spread -- see `forward`, and treat
                            that change as a behavioural fix, not a relabelling.
        """
        self.p_weight = p_weight
        self.soft_penalty = soft_penalty
        self.rate_penalty = rate_penalty
        # Quadratic penalty weight on broth viscosity above VISC_SOFT_START -- the observed collapse mode.
        self.visc_penalty = visc_penalty
        # Credit penicillin removed by the recipe's discharge pulses. Defaults ON because it makes
        # the objective match batch_yield_kg; set False to reproduce the old in-tank-only reward.
        self.harvest_reward = harvest_reward
        # Risk aversion: weight on the batch-OUTCOME spread in the OPTIMISED objective.
        # 0.0 reproduces the stock risk-neutral expectation exactly. See `forward` below.
        self.risk_weight = risk_weight # THIS rewards small std
        # Single global trade-off knob: scales every CONSTRAINT penalty's effective lambda
        # (weight overflow, viscosity collapse, batch-outcome risk) together, "how conservative
        # overall", while leaving each term's relative weighting (soft_penalty vs visc_penalty vs
        # risk_weight) intact. action_rate is excluded -- it is a smoothness preference, not a
        # safety constraint, and stays controlled by rate_penalty alone. Default 1.0 means "exactly
        # what the per-term lambdas alone specify", so existing configs (which don't pass this
        # argument) are unaffected by its introduction.
        self.constraint_strength = constraint_strength
        super().__init__(cost_function=self._cost)

    @property
    def _lambda_weight(self):
        return self.soft_penalty * self.constraint_strength

    @property
    def _lambda_visc(self):
        return self.visc_penalty * self.constraint_strength

    @property
    def _lambda_risk(self):
        return self.risk_weight * self.constraint_strength

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """Risk-sensitive objective: sum_t mean_p(cost) + lambda_risk * std_p(sum_t cost).

        WHY OVERRIDE THIS
        -----------------
        The stock `Expected_cost.forward` minimises the mean over particles. Measured on this
        system the across-particle std runs 19-29x the mean cost (R.1b in evaluations/Rollouts.ipynb),
        so the mean is a weak summary of the imagined outcome: a policy can look good on average
        while a large share of particles predict a collapsed batch -- and collapsed batches do show
        up in the real episodes. Penalising the spread makes the optimiser prefer policies whose
        outcome the model can actually forecast.

        BEHAVIOURAL CHANGE: OUTCOME VARIANCE, NOT PER-TIMESTEP VARIANCE
        -----------------------------------------------------------
        The previous version summed the across-particle std AT EACH TIMESTEP: `sum_t std_p(cost)`.
        That measures how much particles disagree step-by-step, not whether the BATCH outcome is
        risky -- a policy whose particles wobble around a common trajectory but converge to the same
        total cost would be penalised the same as one whose particles diverge into "collapsed" vs
        "fine" batches. The stated objective is reducing batch-to-batch (outcome) variance, so this
        now takes the std ACROSS PARTICLES of each particle's SUMMED trajectory cost:
        `std_p(sum_t cost)`, computed once per rollout rather than once per timestep. This is a
        deliberate behavioural change (not a silent refactor) -- risk_weight values tuned against
        the old per-timestep formulation are not directly comparable to this one.

        THE DETACH, WHICH IS THE WHOLE POINT
        ------------------------------------
        The base class computes `torch.std(costs.detach(), 1)`, so its std is a logging quantity
        with NO gradient path. Reusing it here would move the reported cost while leaving the policy
        gradient untouched -- a silent no-op. The penalty term below is therefore computed from the
        NON-detached costs. The second return value stays detached: it is the SAME outcome-std
        quantity the risk penalty uses (previously it was the per-timestep-summed std), so
        `std_cost_trial_list` and the R.1b diagnostic now report outcome spread, not step spread.
        """
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)
        objective = torch.sum(mean_costs)
        trajectory_cost = torch.sum(costs, 0)  # [particles]: summed per-trajectory OUTCOME cost
        outcome_std = torch.std(trajectory_cost)
        if self.risk_weight:
            objective = objective + self._lambda_risk * outcome_std
        return objective, torch.std(trajectory_cost.detach())


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
        """The cost's individual terms, unsummed, each shaped [T, particles], ALL in one currency:
        p_weight * kg-of-penicillin-equivalent. `reward` is that currency's reference (p_weight *
        kg of actual product); every penalty is `p_weight * lambda_term * kg_violation`, where
        `kg_violation` is a severity ramp (`_ramp_severity`, 0..1 from soft threshold to hard limit)
        times the mass currently at risk (`_mass_kg(P, Wt)`, i.e. what an overflow spill or a
        collapse-driven batch abort would put at risk). A lambda of 1 therefore means "trade 1 kg of
        yield to avoid a full-severity (at-the-hard-limit) violation of that constraint" -- see
        experiments/cost_term_report.py, which divides every term by p_weight to read it in kg.

        Signs are as NAMED, not as combined: `reward` is a reward (higher is better) and the three
        penalties are costs. `_cost` applies the signs.
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

        # Mass currently held in the tank -- what a weight-overflow spill or a viscosity-driven
        # batch abort would put at risk. The SAME currency prices both constraints (and, in
        # `forward`, the risk term), so their lambdas are directly comparable to each other and to
        # p_weight's kg reference.
        mass_at_risk_kg = _mass_kg(P, Wt)

        # soft (weight-overflow constraint): severity ramps 0 at WT_SOFT[1] to 1 at WT_OVERFLOW,
        # multiplied by the mass an overflow would spill. `constraint_strength` scales this term.
        wt_severity = _ramp_severity(Wt, WT_SOFT[1], WT_OVERFLOW)
        soft = self.p_weight * self._lambda_weight * wt_severity * mass_at_risk_kg

        # visc_soft (viscosity-collapse constraint): severity ramps 0 at VISC_SOFT_START to 1 at
        # VISC_MAX, multiplied by the mass a collapse-driven abort would lose. Reaching full
        # severity exactly AT the hard limit (VISC_MAX) is a deliberate change from the previous
        # formula, which used a fixed VISC_SOFT_SCALE=50 denominator unrelated to VISC_MAX=100 and
        # so did not reach full severity until 130 cP, well past the stated hard cap.
        # `constraint_strength` scales this term.
        visc = self._dn(states_sequence[:, :, VISC_IDX], *STATE_RANGES["Viscosity"])
        visc_severity = _ramp_severity(visc, VISC_SOFT_START, VISC_MAX)
        visc_soft = self.p_weight * self._lambda_visc * visc_severity * mass_at_risk_kg

        # action_rate (smoothness preference, NOT a safety constraint -- excluded from
        # constraint_strength): severity is the squared action step as a fraction of the largest
        # possible step (+1 to -1, i.e. divide by 2 before squaring so a full reversal = severity
        # 1), priced against the same mass-at-risk currency so rate_penalty is also an
        # interpretable kg-per-kg-equivalent lambda rather than a raw unit-conversion factor.
        u = inputs_sequence[:, :, 0]
        rate_violation_kg = torch.zeros_like(u)
        rate_violation_kg[1:] = ((u[1:] - u[:-1]) / 2.0) ** 2 * mass_at_risk_kg[1:]
        action_rate = self.p_weight * self.rate_penalty * rate_violation_kg

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
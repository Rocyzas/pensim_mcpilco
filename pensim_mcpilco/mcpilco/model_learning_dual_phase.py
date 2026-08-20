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

import numpy as np

from mcpilco.model_learning_det_time import Model_learning_RBF_det_time
from mcpilco.pensim_wrapper import (T_SAMPLING, BLEND_HALF_WIDTH_HOURS, STATE_NAMES,
                                    STATE_RANGES, decode_state_value)

# ---------------------------------------------------------------------------------------------
# Biomass-progress training split (pivot_mode="biomass") -- STAGE 1 of moving the phase boundary
# off the wall clock. See evaluations/ryu_mu_check for the measurements behind every number here.
#
# WHY: the production regime is not a clock event. Over 40 diagnostic batches the production-rate
# peak lands anywhere in 39-185 h (CV 0.41 in time) but at a far more repeatable biomass level
# (CV 0.21). A fixed pivot_hours therefore splits different batches at genuinely different
# metabolic stages -- the current 100 h pivot sits at ~89% of median peak biomass, a level roughly
# a quarter of batches never reach at all.
#
# WHAT THE PIVOT VARIABLE IS: CER is the best-scoring ONLINE phase coordinate measured there
# (residual across-batch variance of production rate 0.52, vs biomass X 0.61, time 0.82), and the
# simulator builds it as CER = (a0+a1)*q_co2*V -- active biomass x volume. It is not a state
# channel here, and threading the real CER trajectory down to add_data would mean editing
# MC_PILCO.py at four call sites. But X and Wt ARE state columns, and corr(CER, X*Wt) = 0.9910,
# so the same coordinate is recoverable in place from the samples add_data already receives.
# These are the REAL logged trajectory states, not GP predictions, so nothing is approximated
# except CER -> X*Wt itself.
#
# BM = X[g/L] * Wt[kg] / 1000. Measured per-batch peak over the 40 batches: min 1433, p25 1864,
# median 2249, max 3037.
BM_PIVOT_DEFAULT = 1349.0   # ~60% of the median per-batch peak; all 40/40 batches cross it,
                            # inducing a pivot at median 58 h (range 39-91 h, CV 0.19 vs 0.41
                            # for the events a fixed clock is trying to track). At 70% three
                            # batches never cross, at 80% nine never do -- hence the fallback.

# The raw signal is NOT monotone: X*Wt peaks around 134 h and declines, and 13 of 26 batches fall
# back below a fixed threshold after first crossing it. A crossing test on the raw signal would
# therefore be ambiguous late in the batch. Taking a causal running max (np.maximum.accumulate)
# first makes the crossing well-defined without using any future information.


def _bm_from_states(state_samples):
    """Biomass-progress signal BM = X*Wt/1000 (physical units) from NORMALISED state rows.

    Inverts _normalise then the log encoding, per channel, using the same STATE_RANGES /
    decode_state_value the cost function uses -- so this stays correct if either is retuned.

    Accepts a numpy array (add_data's real logged trajectories) OR a torch tensor
    (get_next_state's particles under --onEachRollout). decode_state_value is already
    torch-aware and the affine de-normalisation is dtype-agnostic, so the only thing that has to
    branch is not coercing a tensor through np.asarray -- which would detach it, and raises
    outright once it carries grad.
    """
    is_t = torch.is_tensor(state_samples)
    out = []
    for name in ("X", "Wt"):
        lo, hi = STATE_RANGES[name]
        src = state_samples if is_t else np.asarray(state_samples)
        col = src[:, STATE_NAMES.index(name)]
        out.append(decode_state_value(name, (col + 1.0) / 2.0 * (hi - lo) + lo))
    return out[0] * out[1] / 1000.0


# Half-width of the biomass blend sigmoid, in BM units (NOT hours -- deliberately not derived
# from blend_half_width_hours, since the two live in different spaces and a unit mix-up would be
# silent). Same "1%/99%" convention as the time sigmoid: w ~= 0.01 at pivot_bm - this, ~= 0.99 at
# pivot_bm + this. Calibrated against evaluations/pivot_point_biomass/pivot_points_biomass.csv:
# pivot_bm defaults to 1349 and the median per-batch BM peak is ~2249, so 250 puts the ~99% point
# at ~1600 -- comfortably below every observed peak, so every rollout does reach phase 2.
BLEND_HALF_WIDTH_BM_DEFAULT = 250.0

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
                 pivot_mode="time", pivot_bm=BM_PIVOT_DEFAULT, min_phase_steps=3,
                 on_each_rollout=False, blend_half_width_bm=BLEND_HALF_WIDTH_BM_DEFAULT,
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.pivot_step = pivot_step
        self.pivot_hours = pivot_hours
        self.blend_half_width_hours = blend_half_width_hours
        # pivot_mode selects ONLY how the TRAINING data is hard-split (see add_data). Rollout
        # blending is unchanged in both modes -- still the time sigmoid centred on pivot_hours.
        # That asymmetry is deliberate: the split runs on real logged trajectories where the
        # biomass signal is directly available, whereas the blend runs inside imagined rollouts
        # where the weight would have to become a per-particle stochastic quantity. Keeping the
        # blend on the clock isolates the training-split effect with no change to the rollout
        # maths (mixture variance, gradients, _BLEND_SKIP_EPS accounting all stay as-is).
        # Default "time" reproduces the previous behaviour EXACTLY -- every stored run under
        # results/full/ still regenerates bit-identically (see
        # evaluations/config_regression/check_dual_phase_notes.py).
        if pivot_mode not in ("time", "biomass"):
            raise ValueError(f"pivot_mode must be 'time' or 'biomass', got {pivot_mode!r}")
        self.pivot_mode = pivot_mode
        self.pivot_bm = pivot_bm
        # Never hand a phase a degenerate slice: a trajectory that crosses at step 0, or not
        # until the final step, would otherwise starve one phase of that batch entirely. Clamping
        # into [min_phase_steps, n-1-min_phase_steps] keeps both phases fed from every batch.
        self.min_phase_steps = min_phase_steps
        self._split_log = []   # (step, hours, crossed?) per added trajectory, for diagnostics

        # --onEachRollout: also move the ROLLOUT BLEND onto the biomass coordinate, so the blend
        # tracks each rollout's own progress instead of the wall clock. Off by default, and when
        # off _blend_weight/get_next_state run their pre-existing code verbatim -- see those two
        # methods. Rejected without pivot_mode="biomass": the blend needs the same coordinate the
        # split uses, and silently ignoring the flag would leave a run indistinguishable from a
        # stage-1 one in its own note.txt.
        if on_each_rollout and pivot_mode != "biomass":
            raise ValueError(
                "on_each_rollout=True requires pivot_mode='biomass' (got "
                f"{pivot_mode!r}): the per-rollout blend is computed from the SAME X*Wt "
                "coordinate the training split uses, so enabling it while the split is still "
                "on the clock would blend and split on two different axes.")
        self.on_each_rollout = bool(on_each_rollout)
        self.blend_half_width_bm = blend_half_width_bm
        # Per-rollout running max of the biomass signal, reset by reset_step_counter. Only read
        # under on_each_rollout; see _blend_weight for why the running max is required at all.
        self._bm_max = None
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

    def reset_step_counter(self, start_t=0, bm_max0=None):
        """Reset the decision-step counter used by get_next_state. Must be called before
        every rollout (apply_policy / diagnostic rollout()) that will call get_next_state
        sequentially in decision order -- see PenSimMCPILCOMultiPhase.

        Also clears the on_each_rollout running max. That reset is NOT optional when the flag is
        on: _bm_max holds tensors from the rollout just finished, so carrying it into the next
        one both routes the new rollout off stale biomass and keeps the previous autograd graph
        alive (a second backward through it raises).

        bm_max0 seeds the running max for callers that JUMP to a mid-batch decision index rather
        than walking from 0 (start_t > 0). Under on_each_rollout the weight depends on biomass
        accumulated since the start of the batch, which a jump has not accumulated -- so those
        call sites must pass the real trajectory's running-max biomass at start_t. Leaving it
        None there is rejected in _blend_weight rather than silently under-weighting phase 2."""
        self._t = start_t
        self._bm_max = bm_max0

    def _blend_weight(self, t_step, current_state=None):
        """Sigmoid blend weight at decision step t_step: ~0 well before pivot_hours (phase1
        dominates), ~1 well after (phase2 dominates), 0.5 exactly at pivot_hours.

        k is set so w is ~0.01 at (pivot_hours - blend_half_width_hours) and ~0.99 at
        (pivot_hours + blend_half_width_hours) -- the standard "1%/99%" convention for a
        logistic's effective width, giving e.g. pivot_hours=90/blend_half_width_hours=40 =>
        w~0.01 at 50h, w=0.5 at 90h, w~0.99 at 130h.

        Under on_each_rollout this instead returns a PER-PARTICLE weight on the biomass
        coordinate: same 1%/99% sigmoid, but centred on pivot_bm with blend_half_width_bm, and
        evaluated on the running max of BM = X*Wt/1000 decoded from current_state. The running
        max is required for the same reason add_data needs it -- raw BM peaks around 134h and
        then declines, so a bare threshold would let the weight slide back toward phase 1 near
        harvest (13 of 26 diagnostic batches dip back below a fixed threshold after crossing).

        The result is DETACHED on purpose. current_state carries grad inside
        MC_PILCO.compute_particles_trj, so an attached weight would add a gradient path with
        d(next_states)/dw = (next2 - next1) -- largest exactly where the two phases disagree
        most, i.e. where the model is least trustworthy. That is an invitation for the policy to
        steer biomass so the more favourable phase keeps the weight instead of making
        penicillin, which is a failure mode MC-PILCO is already prone to. Detaching keeps
        per-rollout routing with no new gradient path; a differentiable variant belongs behind
        its own flag, and only once this one is shown to help."""
        if not self.on_each_rollout:
            t_hours = t_step * T_SAMPLING
            k = math.log(99.0) / self.blend_half_width_hours
            return 1.0 / (1.0 + math.exp(-k * (t_hours - self.pivot_hours)))

        if current_state is None:
            raise ValueError("on_each_rollout=True needs current_state to compute the blend "
                             "weight; _blend_weight was called without it.")
        if self._bm_max is None and t_step != 0:
            raise ValueError(
                f"on_each_rollout=True: _blend_weight reached decision {t_step} with no "
                "accumulated biomass. This caller jumped to a mid-batch decision via "
                "reset_step_counter(start_t) without passing bm_max0, so the running max is "
                "empty and the weight would under-weight phase 2. Pass "
                "reset_step_counter(start_t, bm_max0=<running-max BM at start_t>).")

        bm = _bm_from_states(current_state)
        self._bm_max = bm if self._bm_max is None else torch.maximum(self._bm_max, bm)
        k = math.log(99.0) / self.blend_half_width_bm
        return torch.sigmoid(k * (self._bm_max - self.pivot_bm)).detach()

    def _biomass_pivot_step(self, state_samples, log=True):
        """Decision step at which THIS trajectory's biomass-progress signal first crosses
        pivot_bm, from its causal running max (see the module-level notes on why the raw signal
        cannot be used). Falls back to the fixed pivot_step for a trajectory that never crosses,
        so phase 2 is never starved by a batch that simply never grew.

        log=False suppresses the _split_log entry, for callers that need the split point WITHOUT
        claiming a training split happened. _split_log is dumped to split_log.pkl by the
        03_mcpilco_dual_phase_baseline driver and read back by check_split_distribution, so an
        extra entry there would be indistinguishable from a real training split and would skew
        that diagnostic. add_data (the only true training split) keeps the default."""
        n = len(state_samples)
        bm = _bm_from_states(state_samples)
        # add_data hands this numpy; get_gp_estimate_from_data hands it a torch tensor (and on
        # GPU runs, a CUDA one, where the numpy ufuncs below would raise). Coerce once here
        # rather than making every caller remember.
        if torch.is_tensor(bm):
            bm = bm.detach().cpu().numpy()
        bm = np.maximum.accumulate(bm)
        hit = np.flatnonzero(bm >= self.pivot_bm)
        crossed = hit.size > 0
        p = int(hit[0]) if crossed else self.pivot_step
        lo, hi = self.min_phase_steps, n - 1 - self.min_phase_steps
        p = int(np.clip(p, lo, max(lo, hi)))
        if log:
            self._split_log.append((p, p * T_SAMPLING, crossed))
        return p

    def add_data(self, new_state_samples, new_input_samples):
        """Split one trajectory and route each segment to its phase's own add_data.
        states[0..p] (inclusive) go to phase 1, so the t=p state is phase 1's last input row
        AND phase 2's first input row -- every transition (t, t+1) is owned by exactly one
        phase, none dropped or duplicated. Training stays HARD-split even though predictions
        blend smoothly -- see module docstring.

        The split point p is the fixed pivot_step under pivot_mode="time", or THIS trajectory's
        own biomass-progress crossing under pivot_mode="biomass" -- so every batch is cut at the
        same metabolic stage rather than at the same wall-clock hour. Only the split moves;
        get_next_state still blends on the time sigmoid in both modes."""
        p = (self._biomass_pivot_step(new_state_samples) if self.pivot_mode == "biomass"
             else self.pivot_step)
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

        if not self.on_each_rollout:
            # ---- unchanged scalar path (every run predating --onEachRollout takes this) ----
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

        # ---- per-particle biomass path (--onEachRollout) ----
        # Deliberately a separate branch rather than a generalisation of the above: keeping the
        # scalar code textually identical is what makes "nothing changed when the flag is off"
        # checkable by reading the diff instead of by reasoning about broadcasting.
        w = self._blend_weight(t, current_state)          # (n_particles,), detached

        # The skip shortcut now needs unanimity: with a per-particle weight, a single straggling
        # particle still inside the transition means the far-side phase genuinely contributes for
        # that particle, so it cannot be skipped for the batch. Expect more steps evaluating both
        # phases than the scalar path's ~18 of 46 -- that cost is the price of per-rollout routing.
        if bool((w <= _BLEND_SKIP_EPS).all()):
            return self.phase1.get_next_state(current_state, current_input, particle_pred=particle_pred)
        if bool((w >= 1.0 - _BLEND_SKIP_EPS).all()):
            return self.phase2.get_next_state(current_state, current_input, particle_pred=particle_pred)

        next1, mean1, var1 = self.phase1.get_next_state(current_state, current_input, particle_pred=particle_pred)
        next2, mean2, var2 = self.phase2.get_next_state(current_state, current_input, particle_pred=particle_pred)
        # (n_particles,) -> (n_particles, 1) so it broadcasts across the state dimension.
        wc = w.unsqueeze(-1)
        next_states = (1.0 - wc) * next1 + wc * next2
        delta_mean = (1.0 - wc) * mean1 + wc * mean2
        # Same law-of-total-variance mixture formula as the scalar path -- w is still "our belief
        # about which regime governs", it is just now per-particle rather than shared.
        delta_var = ((1.0 - wc) * var1 + wc * var2
                     + (1.0 - wc) * wc * (mean1 - mean2) ** 2)
        return next_states, delta_mean, delta_var

    def get_gp_estimate_from_data(self, states, inputs, flg_pretrain=False, gp_index_list=None,
                                  flg_onestep=False):
        """Diagnostic passthrough (used by get_model_learning_performance): split the
        trajectory exactly like add_data, evaluate each phase's own GPs on its own segment, and
        concatenate the 4 returned lists in [phase1 x num_gp, phase2 x num_gp] order --
        consistent with gp_list's ordering. Deliberately NOT blended: this scores each phase's
        own fit to its own (still hard-split) training data, independent of the rollout-time
        blend.

        The split must follow pivot_mode, exactly as add_data does. It previously always used
        pivot_step, which is only add_data's split under pivot_mode="time"; under "biomass" the
        two diverged (for seed4_13, trained on [0, ~14] but scored on [0, 20]), so each phase was
        graded on a segment it was never fitted to and the reported per-channel MSE / one_step_fit
        R^2 described the wrong data. Callers that consume this: MC_PILCO.get_model_learning_
        performance (prints per-GP MSE during training; return values discarded, so no trained
        policy was ever affected) and eval_multi_phase_lib.one_step_fit (section C.1/C.3/C.4).
        Numbers for pivot_mode="time" runs are unchanged."""
        p = (self._biomass_pivot_step(states, log=False) if self.pivot_mode == "biomass"
             else self.pivot_step)
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

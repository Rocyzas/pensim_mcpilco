"""Recipe-trajectory prior mean for the integrating channels (`P`, and optionally `X`).

WHY THIS EXISTS
---------------
Every GP predicts a DELTA and the rollout integrates it (`Model_learning.get_next_state_from_gp_output`:
`next_states = current_state + delta_sample`). A plain RBF has a ZERO prior mean, so wherever the GP
has no nearby data its predicted delta decays to 0 and the channel FREEZES IN PLACE.

For an equilibrium channel that is harmless. For the accumulating channels {Wt, X, P} it is wrong in
one direction: "no data here" produces "stopped growing", a systematic underestimate rather than a
symmetric uncertainty. Integrated over the ~46-decision horizon those errors compound -- which is why
a one-step R^2 of ~0.99 coexists with a multi-step particle spread ~25x the mean cost (see the R.1b
diagnostic in evaluations/Rollouts.ipynb). The one-step metric never sees the integration.

`wt_mass_balance.WtMassBalance` already fixed this for `Wt` by supplying the KNOWN recipe mass balance
as a prior mean, leaving the RBF to model only the residual. `P` and `X` cannot get the same treatment
from first principles: their true ODE terms are Monod kinetics in substrate and dissolved oxygen, and
neither S nor DO2 is in this 4-channel state.

So this module supplies an EMPIRICAL prior mean instead: the mean per-decision delta measured from
pure-recipe (a = 0) batches, indexed by batch time. The decomposition mirrors the way the ACTION is
parameterised -- the action is a residual on the recipe feed, so the natural prior mean is the recipe
trajectory and the natural GP target is how the action perturbs it. Off-data, predictions then revert
to "grows like the recipe" rather than "stops growing".

WHY THE PRIOR MEAN CARRIES NO ACTION TERM
-----------------------------------------
Unlike the Wt mass balance (where Fs enters the physics directly and analytically), this mean is the
a = 0 baseline by construction. All action sensitivity is left to the RBF residual, which is exactly
the quantity that has to be learned from data. A prior mean that guessed at the action response would
bias the very effect the agent is trying to exploit.

WHY IT IS EVALUATED IN NORMALISED-ENCODED SPACE
----------------------------------------------
The stored channels for {Wt, X, P} are normalise(log(.)), so a delta in this space is a LOG-RATIO --
"grew by 4% this window" -- not an absolute increment. Adding the recipe's log-delta to a particle
whose biomass is already above the recipe therefore means "grow by the same proportion", which
extrapolates far better than transplanting an absolute increment.

NOT CACHED ON DISK, on purpose -- same rule as `_measure_init_state_norm`: the values depend on
STATE_RANGES / WARMUP_H / the encoding, and a stale cache keyed on anything less would silently
poison the model. Measured fresh once per process, memoised below.
"""

import numpy as np
import torch

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES, T_SAMPLING, K_WARM,
                                    CONTROL_H, STATE_LOG_CHANNELS)
from utils.constants import STEP_IN_HOURS

TIME_IDX = STATE_NAMES.index("time")
_T_LO, _T_HI = STATE_RANGES["time"]

# Number of pure-recipe batches averaged to build the mean. Enough to average out batch-to-batch
# variability (which is now real: episodes draw fresh realisations) without costing many rollouts.
DEFAULT_NUM_BATCHES = 4

_traj_cache = {}


def _measure_recipe_deltas(num_batches=DEFAULT_NUM_BATCHES, seed_offset=0):
    """Mean per-decision delta of every state channel under the pure recipe.

    Returns an array [n_decisions, STATE_DIM] of deltas in NORMALISED (encoded) units -- the same
    space the GPs predict in.

    Rolled on a FRESH wrapper with the global NumPy state saved/restored, so a training run's own
    episode/seed sequence is untouched and building the prior mean cannot perturb the experiment.
    """
    key = (num_batches, seed_offset)
    if key in _traj_cache:
        return _traj_cache[key]

    from mcpilco.pensim_wrapper import PenSimWrapper

    recipe_policy = lambda state, decision_idx: np.array([0.0])
    fresh = PenSimWrapper(seed_offset=seed_offset)

    np_state = np.random.get_state()
    deltas = []
    for i in range(num_batches):
        states, _, _ = fresh.rollout(s0=None, policy=recipe_policy, T=CONTROL_H,
                                     dt=T_SAMPLING, noise=None, seed=seed_offset + i)
        deltas.append(np.diff(states, axis=0))
    np.random.set_state(np_state)

    mean_delta = np.mean(np.stack(deltas), axis=0)
    _traj_cache[key] = mean_delta
    return mean_delta


class RecipeTrajectoryMean:
    """Torch-side lookup of the recipe prior mean for ONE channel (batched, autograd-safe)."""

    def __init__(self, channel, num_batches=DEFAULT_NUM_BATCHES, seed_offset=0,
                 dtype=torch.float64, device=torch.device("cpu")):
        if channel not in STATE_NAMES:
            raise ValueError(f"{channel!r} is not a state channel: {STATE_NAMES}")
        self.channel = channel
        self.channel_idx = STATE_NAMES.index(channel)
        self.dtype, self.device = dtype, device

        mean_delta = _measure_recipe_deltas(num_batches, seed_offset)
        self.delta_table = torch.tensor(mean_delta[:, self.channel_idx],
                                        dtype=dtype, device=device)
        self.n_decisions = self.delta_table.shape[0]

        space = "normalised-log" if channel in STATE_LOG_CHANNELS else "normalised"
        print(f"[recipe-mean] {channel}: {self.n_decisions} decisions from {num_batches} recipe "
              f"batches, {space} delta range "
              f"[{self.delta_table.min():.4f}, {self.delta_table.max():.4f}]")

    def delta_norm(self, X):
        """Prior-mean delta for gp inputs X [N, state_dim+1] -> [N, 1].

        The lookup is a hard index on `time`, which the deterministic-clock model advances exactly
        (see Model_learning_RBF_det_time), so no gradient is needed through it. The returned mean is
        constant w.r.t. state and action by construction -- see the module docstring.
        """
        t_norm = X[:, TIME_IDX:TIME_IDX + 1]
        t_h = _T_LO + (t_norm + 1.0) * (_T_HI - _T_LO) / 2.0
        idx = torch.round((t_h - K_WARM * STEP_IN_HOURS) / T_SAMPLING).long()
        idx = idx.clamp(0, self.n_decisions - 1).squeeze(-1)
        return self.delta_table[idx].unsqueeze(-1)

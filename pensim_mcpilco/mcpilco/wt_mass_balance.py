"""Known-physics mass balance for the `Wt` (broth weight) channel.

WHY THIS EXISTS
---------------
`Wt` was the worst GP channel: held-out one-step R^2 = 0.13 and a fitted sigma_n ~25x larger than
the other channels. The cause is NOT noise -- it is a deterministic, *discontinuous* term the RBF
kernel cannot represent: the recipe dumps a ~7200 kg discharge pulse on a handful of single decision
steps late in the batch (decisions 20, 26, 30, 34, 38, 42; plus a 800 kg precursor step). An RBF with
one lengthscale over normalised `time` cannot resolve a one-decision-wide pulse, so it smooths the
pulse away and books the +-7200 kg swing as observation noise -- inflating sigma_n and destroying the
mean fit.

Every term of that pulse is *known*: discharge, Foil, Fw and Fpaa are pure functions of batch time
(the recipe), and Fs is the same recipe profile scaled by the agent's action. So instead of asking the
GP to learn it, we supply it as a PRIOR MEAN and let the RBF model only the residual (evaporation,
acid/base, density drift) -- the standard semiparametric decomposition. The GP library already
subtracts the prior mean before fitting (`GP_prior.get_alpha`: alpha = K^-1 (Y - m_X)) and adds it
back at prediction (`get_estimate_from_alpha`: Y_hat = m_X_test + K alpha), including under the SOD
approximation and inside the imagined rollout, so nothing else needs to change.

THE EQUATION (PenSimPy `indpensim_ode_py.py:323`)
-------------------------------------------------
    dWt = Fs*pho_feed/1000 + pho_oil/1000*Foil + Fb + Fa + Fw + F_discharge - F_evp + Fpaa*pho_paa/1000

We reproduce the terms that are exactly knowable from the recipe + action (Fs, Foil, Fw, Fpaa,
F_discharge) and deliberately OMIT F_evp and the acid/base flows Fa, Fb: those depend on volume and
temperature, which this 4-channel state does not carry. They are smooth and comparatively small, so
they are exactly what the RBF residual should absorb. The prior mean does not need to be exact -- it
needs to carry the discontinuity.
"""

import numpy as np
import torch

from utils.constants import STEP_IN_HOURS, NUM_STEPS
from PenSimPy.pensimpy.data.constants import FS, FOIL, WATER, PAA, DISCHARGE

from mcpilco.pensim_wrapper import (STATE_NAMES, STATE_RANGES, STATE_LOG_FLOOR, T_SAMPLING,
                                    K_WARM, STEPS_PER_DECISION, FS_SCALE, CONTROL_H)

# Densities from indpensim_ode_py (c_s=600, pho_g=1540, pho_w=1000 -> pho_feed = 1324).
PHO_FEED = 600.0 / 1000.0 * 1540.0 + (1.0 - 600.0 / 1000.0) * 1000.0
PHO_OIL = 900.0
PHO_PAA = 1000.0

WT_IDX = STATE_NAMES.index("Wt")
TIME_IDX = STATE_NAMES.index("time")
ACTION_COL = len(STATE_NAMES)  # gp input = [state..., action]

_WT_LO, _WT_HI = STATE_RANGES["Wt"]
_T_LO, _T_HI = STATE_RANGES["time"]


def _build_decision_integrals():
    """Integrate each recipe stream over every decision window, using the SAME integer sim-step
    indexing as PenSimWrapper.rollout (k = K_WARM + 1 + i*STEPS_PER_DECISION + j).

    Sampling the recipe on a decision-time grid instead of these exact k values silently misses the
    discharge pulses entirely (they are one sim-step wide), which is what made this term look like
    noise in the first place. Returns kg accumulated per decision window.
    """
    from mcpilco.pensim_wrapper import PenSimWrapper

    combo = PenSimWrapper._build_default_recipe()
    n_dec = int(CONTROL_H / T_SAMPLING)
    fs_int = np.zeros(n_dec)
    other_int = np.zeros(n_dec)   # Foil, Fw, Fpaa (density-weighted), action-independent
    disch_int = np.zeros(n_dec)
    for i in range(n_dec):
        for j in range(STEPS_PER_DECISION):
            k = K_WARM + 1 + i * STEPS_PER_DECISION + j
            if k > NUM_STEPS:
                break
            v = combo.get_values_dict_at(time=k * STEP_IN_HOURS)
            fs_int[i] += v[FS] * PHO_FEED / 1000.0 * STEP_IN_HOURS
            other_int[i] += (PHO_OIL / 1000.0 * v[FOIL] + v[WATER]
                             + v[PAA] * PHO_PAA / 1000.0) * STEP_IN_HOURS
            disch_int[i] += v[DISCHARGE] * STEP_IN_HOURS
    return fs_int, other_int, disch_int


_FS_INT, _OTHER_INT, _DISCH_INT = _build_decision_integrals()
N_DECISIONS = len(_FS_INT)


class WtMassBalance:
    """Torch-side lookup + evaluation of the Wt prior mean (batched, autograd-safe)."""

    def __init__(self, dtype=torch.float64, device=torch.device("cpu")):
        self.dtype, self.device = dtype, device
        self.fs_int = torch.tensor(_FS_INT, dtype=dtype, device=device)
        self.other_int = torch.tensor(_OTHER_INT, dtype=dtype, device=device)
        self.disch_int = torch.tensor(_DISCH_INT, dtype=dtype, device=device)

    def delta_norm(self, X):
        """Predicted NORMALISED-log delta of Wt for gp inputs X [N, state_dim+1] -> [N, 1].

        Gradients flow through the action (Fs term) and through the current Wt (log ratio); the
        decision-index lookup is a hard index on `time`, which is a deterministic clock, so no
        gradient is needed there.
        """
        wt_norm = X[:, WT_IDX:WT_IDX + 1]
        t_norm = X[:, TIME_IDX:TIME_IDX + 1]
        a = X[:, ACTION_COL:ACTION_COL + 1]

        # decode current physical weight and batch time
        wt_phys = torch.exp(_WT_LO + (wt_norm + 1.0) * (_WT_HI - _WT_LO) / 2.0)
        t_h = _T_LO + (t_norm + 1.0) * (_T_HI - _T_LO) / 2.0

        idx = torch.round((t_h - K_WARM * STEP_IN_HOURS) / T_SAMPLING).long().clamp(0, N_DECISIONS - 1)
        idx = idx.squeeze(-1)
        fs_i = self.fs_int[idx].unsqueeze(-1)
        other_i = self.other_int[idx].unsqueeze(-1)
        disch_i = self.disch_int[idx].unsqueeze(-1)

        d_wt = fs_i * (1.0 + FS_SCALE * a) + other_i - disch_i
        wt_next = torch.clamp(wt_phys + d_wt, min=STATE_LOG_FLOOR)
        # stored channel is normalise(log(Wt)), so its delta is 2*log(Wt_next/Wt)/(hi-lo)
        return 2.0 * (torch.log(wt_next) - torch.log(wt_phys)) / (_WT_HI - _WT_LO)

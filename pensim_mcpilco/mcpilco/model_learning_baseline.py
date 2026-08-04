"""Plain-RBF ablation baseline for Model_learning_RBF_det_time: same deterministic `time`
channel, state clamp and SOD approximation, but WITHOUT the Wt mass-balance / Viscosity
recipe-mean prior means -- every channel gets a vanilla zero-mean RBF. Exists to isolate what
those two prior means actually contribute (see model_learning_det_time.py's RBF_WtMassBalance /
RBF_RecipeMean docstrings for what this removes and why each one was added).
"""

from mcpilco.model_learning_det_time import Model_learning_RBF_det_time


class Model_learning_RBF_baseline(Model_learning_RBF_det_time):
    """Model_learning_RBF_det_time with get_gp's Wt/Viscosity prior-mean branches skipped, so
    every channel falls through to the grandparent's plain SGP.RBF(**init_dict). Bypasses
    Model_learning_RBF_det_time.get_gp entirely (same super(Model_learning_RBF_det_time, self)
    idiom that method itself uses to reach ITS parent) rather than reimplementing its dispatch,
    so this can never drift out of sync with which channels that method special-cases.
    """

    def get_gp(self, gp_index, init_dict):
        return super(Model_learning_RBF_det_time, self).get_gp(gp_index, init_dict)

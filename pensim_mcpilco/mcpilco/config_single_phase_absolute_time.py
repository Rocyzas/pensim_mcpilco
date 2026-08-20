"""Absolute-action variant of the single-phase plain-RBF baseline WITH `time` kept as a GP
regressor: config_single_phase_baseline_time (same Model_learning_RBF_baseline swap, no prior
means, `time` NOT dropped from any GP's active_dims) plus the absolute action encoding.

This is the `time`-carrying half of the pair described in config_single_phase_absolute's
docstring -- it shares that module's `_absolutise` helper, so the two differ ONLY in which
baseline config they wrap, i.e. only in the GP input set. That is what makes the 2x2 clean:

                       | `time` dropped from GP inputs | `time` kept
    residual action    | config_single_phase_baseline   | config_single_phase_baseline_time
    absolute action    | config_single_phase_absolute   | config_single_phase_absolute_time (here)

and the absolute-vs-residual and time-vs-no-time comparisons stay independent of each other.

Logs to its own results/single_phase_absolute_time/ tree so it never collides with the other
three arms. Accepts the exact same kwargs as config_single_phase_absolute (passed straight
through), so it stays in sync automatically.
"""

from mcpilco import config_single_phase_baseline_time
from mcpilco.config_single_phase_absolute import _absolutise
from mcpilco.pensim_wrapper import FS_ABS_MIN, FS_ABS_MAX


def get_config(fs_abs_min=FS_ABS_MIN, fs_abs_max=FS_ABS_MAX, action_mode="absolute", **kwargs):
    cfg = config_single_phase_baseline_time.get_config(**kwargs)
    _absolutise(cfg, fs_abs_min, fs_abs_max, action_mode, "config_single_phase_absolute_time")
    cfg["mc_pilco_init"]["log_path"] = (
        f"results/single_phase_absolute_time/seed{kwargs.get('seed', 1)}")
    return cfg

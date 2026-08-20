"""Absolute-action variant of the single-phase PLAIN-RBF baseline: identical to
config_single_phase_baseline.get_config in every respect except `wrapper_par`, which switches
PenSimWrapper to action_mode="absolute" -- Fs is commanded directly over
[fs_abs_min, fs_abs_max] (default 0-200 L/h) instead of as a +/-FS_SCALE residual on the recipe
Fs profile. See pensim_wrapper.ACTION_MODES / fs_from_action for the map and its consequences.

The action ENCODING and the GP's INPUT set are orthogonal knobs: this one lives in `wrapper_par`,
the `time`-regressor choice lives in `model_learning_par["init_dict_list"]`. So the same
`_absolutise` helper below serves both members of the pair, exactly mirroring the existing
baseline / baseline_time split:

    config_single_phase_baseline       -> config_single_phase_absolute       (this module, no `time`)
    config_single_phase_baseline_time  -> config_single_phase_absolute_time  (`time` kept)

WHY IT WRAPS A *BASELINE* CONFIG AND NOT config_single_phase
------------------------------------------------------------
config_single_phase's model is Model_learning_RBF_det_time, whose `Wt` GP carries the analytic
mass-balance prior mean -- and wt_mass_balance.py hardcodes the RESIDUAL formula
(`d_wt = fs_i * (1 + FS_SCALE * a)`, with fs_i the recipe's integrated feed over the window).
Under an absolute action that prior predicts broth weight from a feed the simulator never
applied, which is wrong in the one channel that has an exact physics prior. Both baseline configs
use Model_learning_RBF_baseline (vanilla zero-mean RBF everywhere, see model_learning_baseline.py),
which makes that prior and the Viscosity recipe-trajectory mean inert, so neither can silently
disagree with the action encoding. `_absolutise`'s check is what keeps that true if either
baseline config is ever repointed.

Making the Wt prior absolute-aware is a real option (integrate hours per decision window and use
`fs_from_action(a, ...) * PHO_FEED/1000 * hours_i`), just deliberately out of scope here.

Everything else -- GP hyperparameter inits, SOD settings, cost terms, penalties, exploration
count, measurement-delay flags -- comes from the wrapped baseline config unchanged, and kwargs
pass straight through, so this stays in sync as that function's parameter set evolves.
config_single_phase.py / config_single_phase_baseline*.py / their drivers are not touched at all.

EVALUATING RUNS FROM THIS CONFIG
--------------------------------
Use evaluations/evaluations_single_phase_absolute.py, which passes this get_config and the
results/single_phase_absolute/ root to eval_single_phase_lib.load_run. Loading such a run with a
RESIDUAL driver now fails loudly (its get_config has no action_mode kwarg) rather than silently
rebuilding the wrapper with the wrong encoding -- see eval_single_phase_lib._GET_CONFIG_KEYS.
"""

from mcpilco import config_single_phase_baseline
from mcpilco.model_learning_baseline import Model_learning_RBF_baseline
from mcpilco.pensim_wrapper import FS_ABS_MIN, FS_ABS_MAX


def _absolutise(cfg, fs_abs_min, fs_abs_max, action_mode, log_tree):
    """Switch a plain-RBF single-phase cfg to the absolute action encoding, in place.

    Shared by this module and config_single_phase_absolute_time so the two can never drift in
    what "absolute" means -- only in which GP inputs they keep.
    """
    if action_mode != "absolute":
        raise ValueError(
            f"{log_tree} is the absolute-action config; action_mode must be 'absolute', got "
            f"{action_mode!r}. For the residual encoding use the config_single_phase_baseline"
            "[_time] module this one wraps.")

    # The whole point of building on a baseline config (see module docstring). If this fires, the
    # prior means are back in play and the Wt mass balance is silently computing the broth-weight
    # delta from the recipe-residual feed formula while the simulator applies an absolute one.
    if cfg["mc_pilco_init"]["f_model_learning"] is not Model_learning_RBF_baseline:
        raise AssertionError(
            f"{log_tree} requires the plain-RBF baseline model "
            f"({Model_learning_RBF_baseline.__name__}), got "
            f"{cfg['mc_pilco_init']['f_model_learning'].__name__} -- the Wt mass-balance prior "
            "mean (wt_mass_balance.py) hardcodes the residual action formula and is invalid "
            "under action_mode='absolute'.")

    cfg["wrapper_par"].update(action_mode=action_mode,
                              fs_abs_min=fs_abs_min, fs_abs_max=fs_abs_max)
    return cfg


def get_config(fs_abs_min=FS_ABS_MIN, fs_abs_max=FS_ABS_MAX, action_mode="absolute", **kwargs):
    cfg = config_single_phase_baseline.get_config(**kwargs)
    _absolutise(cfg, fs_abs_min, fs_abs_max, action_mode, "config_single_phase_absolute")
    cfg["mc_pilco_init"]["log_path"] = f"results/single_phase_absolute/seed{kwargs.get('seed', 1)}"
    return cfg

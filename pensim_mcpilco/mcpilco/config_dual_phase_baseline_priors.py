"""Dual-phase plain-RBF baseline WITH an empirical recipe-trajectory prior mean on every learned
channel: identical to config_dual_phase_baseline (same plain-RBF phases, same `time` dropped from the
GP kernels, same penalties / GP inits / pivot-blend / measurement flags) EXCEPT the phase model class
is Model_learning_RBF_recipe_priors instead of Model_learning_RBF_baseline. So both phases' GPs get a
non-zero prior mean m(x,u) -- the mean per-decision delta measured from NUM_PRIOR_RECIPE_BATCHES (=10)
pure-recipe simulator batches, indexed by batch time -- and each RBF learns only the action-induced
residual (see model_learning_priors.py).

Built by post-processing config_dual_phase_baseline.get_config: it already injects the plain-RBF phase
class and drops `time` from the kernels; here we simply RE-POINT phase_model_cls at the recipe-priors
subclass (which extends Model_learning_RBF_baseline, so the kernels stay identical -- only the prior
mean is added) and redirect log_path. config_dual_phase_baseline.py / config_dual_phase.py and their
drivers are not touched, so the `*_baseline` behaviour is unchanged and the two are directly comparable.

Accepts the exact same kwargs as config_dual_phase_baseline.get_config (passed straight through).
"""

from mcpilco import config_dual_phase_baseline
from mcpilco.model_learning_priors import Model_learning_RBF_recipe_priors


def get_config(**kwargs):
    cfg = config_dual_phase_baseline.get_config(**kwargs)
    # keep the baseline's plain-RBF, time-dropped kernels; only swap in the recipe-mean phase class
    cfg["mc_pilco_init"]["model_learning_par"]["phase_model_cls"] = Model_learning_RBF_recipe_priors
    cfg["mc_pilco_init"]["log_path"] = f"results/dual_phase_baseline_priors/seed{kwargs.get('seed', 1)}"
    return cfg

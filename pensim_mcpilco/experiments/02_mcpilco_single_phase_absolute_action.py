"""
 PYTHONPATH=.. python -m experiments.02_mcpilco_single_phase_absolute_action --seed 1 --num_trials 5 --fast

ABSOLUTE-ACTION experiment: identical driver to 02_mcpilco_single_phase_baseline.py, but built on
config_single_phase_absolute, which switches PenSimWrapper to action_mode="absolute". The policy
commands Fs DIRECTLY over [--fs_abs_min, --fs_abs_max] (default 0-200 L/h) instead of emitting a
+/-FS_SCALE multiplicative residual on the recipe Fs profile, and the initial exploration episodes
use the SAME encoding -- `_recipe_exploration_policy` draws U(-1, 1) per ~19h segment, which under
this mode is a uniform draw over the full 0-200 L/h band (see its comment in pensim_wrapper.py).
So exploration and every on-policy trial share one action encoding; there is no residual-vs-
absolute split anywhere in the training set.

Logs to its own results/single_phase_absolute/ tree so it never collides with the residual runs.

WHAT TO WATCH, versus the residual runs
---------------------------------------
1. Under the residual parameterisation the action BOUND confined the policy to 0.5-1.5x recipe, so
   the reachable set was (roughly) the data support. Here it is not: the policy can command any
   Fs in the band while the GP only has data where exploration happened to land. Expect the
   optimiser to probe GP extrapolation; state_clamp and the cost penalties are the only pushback.
2. A large share of exploration batches will collapse (sustained 200 L/h adds ~61,000 kg of feed
   over the batch against WT_OVERFLOW = 1.2e5; the bottom of the band is starvation). That is
   intended -- the GP has never had data on what "too much feed, held" does -- and nothing screens
   them out (FAILED_YIELD_KG / MAX_EXPLORATION_RETRIES are disabled in pensim_wrapper.py).
3. `a = 0` is now mid-range feed (~100 L/h), not the recipe. The policy's near-zero init therefore
   starts the optimiser at a constant ~100 L/h from inoculation, against a recipe that feeds
   8-30 L/h before hour 24 -- i.e. the first policy-optimisation steps run on GP extrapolation.
   The policy is optimised for the full opt_steps before it is ever rolled out, so this affects
   where the optimiser starts, not what gets deployed.
4. No action sequence reproduces the recipe here: fs_k is constant across a whole T_SAMPLING
   window, and the recipe profile has 4h resolution before hour 24. The recipe REFERENCE batches
   in eval are unaffected -- they go through rollout()'s pid_baseline branch at native resolution.

Anchors and high-feed probes default OFF and are best left off for this experiment: both are
defined relative to "a = 0 == the recipe" (setup_recipe_anchors' pure-recipe launch states,
PROBE_PLAN's +/- excursions). They are kept ENCODING-consistent if enabled -- the fresh wrappers
they build now inherit this run's action_mode -- but their semantics shift to "constant mid-range
feed" and "excursions around mid-range feed". See their comments in pensim_wrapper.py.
"""
import argparse
import datetime
import pickle
import pprint
from pathlib import Path

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCODelayed

_RESULTS_ROOT = Path(_ROOT) / "results" / "single_phase_absolute"

# Kept as a plain literal (not imported from mcpilco.config_single_phase.COST_FUNCTIONS) so
# argparse's --cost_function choices= can be built at module level, BEFORE the deferred
# get_config import below runs -- see main()'s t_sampling handling for why that import can't
# happen at module level any more. get_config still does the actual name->class lookup (and
# will KeyError on a real mismatch), so this list is just for CLI help/validation.
_COST_FUNCTION_NAMES = ("PeniConcentrationCost", "PeniConcentrationDenseCost",
                        "PeniConcentrationChangeCost", "PeniMassTerminalCost", "PeniMassChangeCost")

"""Auto-incrementing default log dir - not to overwrite:
seed{seed}_1, seed{seed}_2,etc"""
def _next_run_dir(seed):
    n = 1
    while (_RESULTS_ROOT / f"seed{seed}_{n}").exists():
        n += 1
    return str(_RESULTS_ROOT / f"seed{seed}_{n}")


def _write_note(log_path, run_params, cfg):
    """Dump the parameters used for this run to a note.txt."""
    lines = [
        f"run timestamp : {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "== run parameters ==",
    ]
    lines += [f"{k} = {v}" for k, v in run_params.items()]
    lines += ["", "== resolved config =="]
    lines.append(pprint.pformat(cfg, width=100, sort_dicts=False))
    (Path(log_path) / "note.txt").write_text("\n".join(lines) + "\n")


def main(seed=1, num_trials=10, fast=False, out_dir=None,
         optim_horizon=None, num_anchor_batches=0, num_anchors=None, anchor_var=0.01,
         risk_weight=0.0, visc_penalty=0.02, constraint_strength=0.75, harvest_reward=True,
         num_high_feed_probes=0, pms_visc_delay=False, use_offline_measurements=False,
         t_sampling=None, cost_function=None, num_explorations=None,
         fs_abs_min=None, fs_abs_max=None):
    # T_sampling is a module-level pensim_wrapper.py constant baked into several OTHER mcpilco
    # modules at THEIR OWN import time (model_learning_det_time's TIME_DELTA_NORM,
    # penicillin_cost.py, PenSimWrapper.rollout's own STEPS_PER_DECISION) via
    # `from mcpilco.pensim_wrapper import T_SAMPLING`-style statements, which snapshot the value
    # at THAT moment. So this override must run, via pensim_wrapper.set_t_sampling(), before
    # config_single_phase_absolute (and everything it transitively imports) is EVER imported in
    # this process -- hence get_config is imported here, deferred, instead of at module level
    # like every other name in this file. See set_t_sampling's own docstring for the full
    # rationale and its one-process-per-value caveat.
    #
    # action_mode needs none of this dance: it is a PenSimWrapper INSTANCE attribute read only
    # inside rollout(), never snapshotted at import by anything.
    if t_sampling is not None:
        import mcpilco.pensim_wrapper as _pw
        _pw.set_t_sampling(t_sampling)
    from mcpilco.config_single_phase_absolute import get_config

    # Only forward the band when overridden, so the config's own defaults stay the single source
    # of truth for what "0-200" means.
    band_kwargs = {}
    if fs_abs_min is not None:
        band_kwargs["fs_abs_min"] = fs_abs_min
    if fs_abs_max is not None:
        band_kwargs["fs_abs_max"] = fs_abs_max

    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast,
                     optim_horizon_steps=optim_horizon, num_anchor_batches=num_anchor_batches,
                     num_anchors=num_anchors, anchor_var=anchor_var, risk_weight=risk_weight,
                     visc_penalty=visc_penalty, constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward,
                     num_high_feed_probes=num_high_feed_probes, pms_visc_delay=pms_visc_delay,
                     use_offline_measurements=use_offline_measurements,
                     cost_function=cost_function, num_explorations=num_explorations,
                     **band_kwargs)
    log_path = out_dir if out_dir is not None else _next_run_dir(seed)
    cfg["mc_pilco_init"]["log_path"] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)

    run_params = {"seed": seed, "num_trials": num_trials, "fast": fast, "out_dir": out_dir,
                  "optim_horizon": optim_horizon, "num_anchor_batches": num_anchor_batches,
                  "num_anchors": num_anchors, "anchor_var": anchor_var,
                  "risk_weight": risk_weight, "visc_penalty": visc_penalty,
                  "constraint_strength": constraint_strength,
                  "harvest_reward": harvest_reward, "num_high_feed_probes": num_high_feed_probes,
                  # Read back from cfg (not re-hardcoded here) so note.txt can never drift from
                  # what the wrapper actually used -- see eval_single_phase_lib.py's
                  # _GET_CONFIG_KEYS/_build_cfg_kwargs, which reads this back at eval time.
                  "pms_visc_delay": cfg["wrapper_par"]["pms_visc_delay"],
                  "use_offline_measurements": cfg["wrapper_par"]["use_offline_measurements"],
                  # Same rule: the action encoding this run actually trained under. Without these
                  # three lines a saved run is indistinguishable from a residual one.
                  "action_mode": cfg["wrapper_par"]["action_mode"],
                  "fs_abs_min": cfg["wrapper_par"]["fs_abs_min"],
                  "fs_abs_max": cfg["wrapper_par"]["fs_abs_max"],
                  "t_sampling": cfg["mc_pilco_init"]["T_sampling"],
                  "cost_function": cfg["mc_pilco_init"]["f_cost_function"].__name__,
                  "num_explorations": cfg["reinforce_par"]["num_explorations"]}
    _write_note(log_path, run_params, cfg)

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    print(f"[action] mode={wrapper.action_mode}: Fs commanded directly over "
          f"[{wrapper.fs_abs_min:g}, {wrapper.fs_abs_max:g}] L/h "
          f"(a=-1 -> {wrapper.fs_abs_min:g}, a=0 -> "
          f"{(wrapper.fs_abs_min + wrapper.fs_abs_max) / 2:g}, a=+1 -> {wrapper.fs_abs_max:g})")
    agent = PenSimMCPILCODelayed(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    # multi-origin short rollouts: build the fixed anchor set once, before training (no-op if
    # disabled). See this module's docstring for why these are best left off here.
    if num_anchor_batches > 0:
        agent.setup_recipe_anchors(**cfg["anchor_par"])
    # probe batches: excite the feed->response channels the exploration policy structurally
    # cannot (sustained, off-recipe, held across the production window), before training starts
    # (no-op if disabled). See PROBE_PLAN / setup_high_feed_probes docstring.
    if num_high_feed_probes > 0:
        agent.setup_high_feed_probes(**cfg["probe_par"])
    agent.reinforce(**cfg["reinforce_par"])

    # constraint plots
    pickle.dump(wrapper.monitor, open(Path(log_path) / "monitor.pkl", "wb"))
    print(f"Saved monitor.pkl ({len(wrapper.monitor)} episodes) to {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_trials", type=int, default=10)
    p.add_argument("--fast", action="store_true", help="small particles/steps/epochs for quick debugging")
    p.add_argument("--out_dir", type=str, default=None, help="override log_path")
    # THE point of this driver: the absolute feed band the action spans (a=-1 -> min, a=+1 -> max).
    p.add_argument("--fs_abs_min", type=float, default=None,
                   help="Fs (L/h) commanded at a = -1 (default: pensim_wrapper.FS_ABS_MIN = 0)")
    p.add_argument("--fs_abs_max", type=float, default=None,
                   help="Fs (L/h) commanded at a = +1 (default: pensim_wrapper.FS_ABS_MAX = 200). "
                        "Narrowing this band is the main knob for trading exploration coverage "
                        "against collapsed batches -- the recipe itself runs 8-150 L/h.")
    # multi-origin short-rollout optimisation (all optional; defaults reproduce stock MC-PILCO)
    p.add_argument("--optim_horizon", type=int, default=None,
                   help="cap the imagined GP-rollout to this many steps during policy optimisation")
    p.add_argument("--num_anchor_batches", type=int, default=0,
                   help="constant-mid-feed batches to launch short rollouts from (0 = disabled). "
                        "NOTE these are pure-RECIPE anchors only in residual mode; see module docstring")
    # must stay an int: it reaches setup_recipe_anchors(), which subsamples with it (None -> crash)
    p.add_argument("--num_anchors", type=int, default=12, help="anchor launch states spread across the batch")
    p.add_argument("--anchor_var", type=float, default=0.01, help="per-anchor particle-init variance")
    # risk-averse objective: mean + risk_weight * across-particle std (0 = stock risk-neutral mean).
    # The std runs ~25x the mean cost here, so useful values are small: ~0.005-0.02.
    p.add_argument("--risk_weight", type=float, default=0.0,
                   help="PARTICLE SPREAD PENTALTY. weight on imagined-outcome spread in the objective (0 = disabled)")
    # cost-shaping terms (see penicillin_cost.PeniConcentrationCost).
    p.add_argument("--visc_penalty", type=float, default=0.02,
                   help="lambda_visc: viscosity-collapse constraint weight (0 = disabled)")
    p.add_argument("--constraint_strength", type=float, default=0.75,
                   help="global knob scaling soft_penalty/visc_penalty/risk_weight together "
                        "(see penicillin_cost.py; NOT used to scale rate_penalty)")
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false",
                   help="score only in-tank mass, ignoring penicillin drawn off by the discharge "
                        "pulses (~20%% of batch_yield_kg). Default is to credit it.")
    # targeted probing: fixed-profile batches added to the GP training set before trial 0, on top
    # of the exploration batches, so the GPs see sustained off-recipe feed (0 = disabled).
    p.add_argument("--num_high_feed_probes", type=int, nargs="?", const=4, default=0,
                   help="fixed-profile probe batches added to the GP training set before training "
                        "starts, ON TOP OF --num_explorations. Bare flag = 4 = one of each probe "
                        "in PROBE_PLAN. NOTE the PROBE_PLAN shapes are excursions around mid-range "
                        "feed here, not around the recipe (0 = disabled)")
    p.add_argument("--pms_visc_delay", action="store_true",
                   help="use the MC-PILCO4PMS-style delayed/held Viscosity measurement (12h "
                        "sampling + 4h analysis delay) as control_policy's input, instead of the "
                        "true instantaneous value (default: off)")
    p.add_argument("--use_offline_measurements", action="store_true",
                   help="read Viscosity from the simulator's delayed lab-assay proxy (12h "
                        "sampling + 4h analysis delay) EVERYWHERE -- GP training, cost, and the "
                        "policy alike, no true-vs-measured split -- instead of the always-live "
                        "online value. Mutually exclusive with --pms_visc_delay (PenSimWrapper "
                        "raises if both are set). Default: off (plain online Viscosity).")
    p.add_argument("--t_sampling", type=float, default=None,
                   help="override the global decision-step interval in hours (repo default: 5h). "
                        "Affects the WHOLE process (see pensim_wrapper.set_t_sampling) -- the "
                        "simulator's decision cadence, the deterministic time channel's per-step "
                        "delta, and every hour<->step conversion, not just this run's config.")
    p.add_argument("--cost_function", type=str, default=None, choices=_COST_FUNCTION_NAMES,
                   help="reward/cost class (see mcpilco/penicillin_cost.py); default is this "
                        "config's own default (PeniConcentrationDenseCost)")
    p.add_argument("--num_explorations", type=int, default=None,
                   help="initial random-exploration episodes collected before the first trial "
                        "(default: 5). Each draws a fresh U(-1,1) level per ~19h segment, i.e. a "
                        "uniform draw over the whole absolute feed band")
    args = p.parse_args()
    # keyword args: this call has outgrown safe positional matching, and a mis-ordered pair here
    # would silently train on the wrong config rather than fail.
    main(seed=args.seed, num_trials=args.num_trials, fast=args.fast, out_dir=args.out_dir,
         optim_horizon=args.optim_horizon, num_anchor_batches=args.num_anchor_batches,
         num_anchors=args.num_anchors, anchor_var=args.anchor_var,
         risk_weight=args.risk_weight, visc_penalty=args.visc_penalty,
         constraint_strength=args.constraint_strength,
         harvest_reward=args.harvest_reward, num_high_feed_probes=args.num_high_feed_probes,
         pms_visc_delay=args.pms_visc_delay, use_offline_measurements=args.use_offline_measurements,
         t_sampling=args.t_sampling, cost_function=args.cost_function,
         num_explorations=args.num_explorations,
         fs_abs_min=args.fs_abs_min, fs_abs_max=args.fs_abs_max)

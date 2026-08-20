"""
 PYTHONPATH=.. python -m experiments.02_mcpilco_single_phase_baseline_cer --seed 1 --num_trials 5 --fast

CER-state ablation: identical in every respect to 02_mcpilco_single_phase_baseline.py (same
config_single_phase_baseline, same plain-RBF model, same cost, same CLI) EXCEPT that the observed
state gains a sixth channel -- CER, the carbon evolution rate:

    ["Wt", "X", "P", "Viscosity", "time"]  ->  ["Wt", "X", "P", "Viscosity", "time", "CER"]

Logs to results/single_phase_baseline_cer/ so it never collides with the 5-channel baseline's
tree. Comparing a seed here against the SAME seed under 02_mcpilco_single_phase_baseline.py is
the intended A/B: everything else is held fixed.

WHY CER, AND THE CAVEAT THIS RUN EXISTS TO SETTLE
-------------------------------------------------
CER is the best-scoring ONLINE phase coordinate measured over the 40-batch diagnostic sweep in
evaluations/ryu_mu_check: binning all rows by a coordinate and measuring the residual across-batch
variance of penicillin production rate gives CER 0.52 vs biomass X 0.61, mu-from-X 0.62, time 0.82,
OUR 0.94 (lower = better). It is also the only candidate that is smooth through the CO2 inhibition
excursions -- it contributes 1.15x its share of total variation at those timesteps, against 5.2x
for OUR -- because the simulator builds it from a biomass LEVEL, (a0+a1)*q_co2*V, rather than from
a rate. And it costs nothing to observe: online off-gas, no lab assay, no 12h/4h measurement delay.

BUT it is very nearly redundant given the state already observed here. Over those same 40 batches:

    corr(CER, X)   = 0.968      corr(CER, X*V) = 0.992
    linear    R^2(CER | X, P, Viscosity, Wt, time) = 0.994
    quadratic R^2 (adds X^2, X*t, t^2, X*Wt)       = 0.999

So CER carries ~0.1% new information; what it really supplies is a better-CONDITIONED coordinate,
not a new one. The one mechanistically distinct part is that CER counts only the metabolically
active regions a0+a1 while X counts a0+a1+a3+a4 including degenerated and autolysed biomass -- the
active fraction falls 1.00 -> 0.86 across the batch -- but even that is 93% linearly explained by
the existing channels, mostly through time.

That makes this a genuine coin-flip worth running rather than reasoning about, and the two
outcomes are both informative:
  - it HELPS  -> the GP could not construct that nonlinear combination of {X, Wt, time} itself, and
                 handing it over as a ready-made input is worth a state dimension;
  - it HURTS  -> the extra dimension costs more (near-degenerate input manifold, harder ARD
                 lengthscale estimation, more particles needed) than the conditioning gains, which
                 is the collinearity failure mode already seen in this repo.
Judge it on final yield AND on the GP diagnostics, not on yield alone.
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

# NOTE: mcpilco.pensim_wrapper is imported here, but NOT PenSimWrapper/PenSimMCPILCODelayed -- see
# main()'s deferred imports. Importing the MODULE is safe (it defines STATE_NAMES rather than
# reading it from somewhere else); importing the names that BAKE IN channel indices is not.
_RESULTS_ROOT = Path(_ROOT) / "results" / "single_phase_baseline_cer"

# The state this driver runs with. Append-only: set_state_names() rejects any reordering, because
# every X_IDX in penicillin_cost / wt_mass_balance / model_learning_det_time is positional.
CER_STATE_NAMES = ["Wt", "X", "P", "Viscosity", "time", "CER"]

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
         state_names=None):
    # Both overrides below mutate module-level constants that OTHER mcpilco modules snapshot at
    # THEIR import time (`from mcpilco.pensim_wrapper import T_SAMPLING` / `STATE_NAMES.index(...)`),
    # so both must run before config_single_phase_baseline -- and everything it transitively
    # imports -- is EVER imported in this process. Hence get_config and the wrapper classes are
    # imported inside main(), deferred, rather than at module level. See set_t_sampling's and
    # set_state_names' own docstrings for the full rationale and the one-process-per-value caveat.
    import mcpilco.pensim_wrapper as _pw
    if t_sampling is not None:
        _pw.set_t_sampling(t_sampling)
    # This is the ONLY thing that differs from 02_mcpilco_single_phase_baseline.py's behaviour.
    # state_names=None means "leave pensim_wrapper's default 5-channel STATE_NAMES alone", which
    # is what --no_cer selects to produce the A/B control through these same code paths.
    if state_names is not None:
        _pw.set_state_names(state_names)

    from mcpilco.config_single_phase_baseline import get_config
    from mcpilco.pensim_wrapper import PenSimWrapper, PenSimMCPILCODelayed

    cfg = get_config(seed=seed, num_trials=num_trials, fast=fast,
                     optim_horizon_steps=optim_horizon, num_anchor_batches=num_anchor_batches,
                     num_anchors=num_anchors, anchor_var=anchor_var, risk_weight=risk_weight,
                     visc_penalty=visc_penalty, constraint_strength=constraint_strength,
                     harvest_reward=harvest_reward,
                     num_high_feed_probes=num_high_feed_probes, pms_visc_delay=pms_visc_delay,
                     use_offline_measurements=use_offline_measurements,
                     cost_function=cost_function, num_explorations=num_explorations)
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
                  "t_sampling": cfg["mc_pilco_init"]["T_sampling"],
                  "cost_function": cfg["mc_pilco_init"]["f_cost_function"].__name__,
                  "num_explorations": cfg["reinforce_par"]["num_explorations"],
                  # Recorded so eval code can rebuild the right state layout: a run logged under
                  # this driver is NOT interchangeable with a 5-channel baseline run.
                  "state_names": _pw.STATE_NAMES,
                  "state_dim": _pw.STATE_DIM}
    _write_note(log_path, run_params, cfg)
    print(f"state ({_pw.STATE_DIM}ch): {_pw.STATE_NAMES}")

    wrapper = PenSimWrapper(**cfg["wrapper_par"])
    agent = PenSimMCPILCODelayed(pensim_wrapper=wrapper, **cfg["mc_pilco_init"])
    # multi-origin short rollouts: build the fixed anchor set once, before training (no-op if disabled)
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
    # multi-origin short-rollout optimisation (all optional; defaults reproduce stock MC-PILCO)
    p.add_argument("--optim_horizon", type=int, default=None,
                   help="cap the imagined GP-rollout to this many steps during policy optimisation")
    p.add_argument("--num_anchor_batches", type=int, default=0,
                   help="pure-recipe batches to launch short rollouts from (0 = disabled)")
    # must stay an int: it reaches setup_recipe_anchors(), which subsamples with it (None -> crash)
    p.add_argument("--num_anchors", type=int, default=12, help="anchor launch states spread across the batch")
    p.add_argument("--anchor_var", type=float, default=0.01, help="per-anchor particle-init variance")
    # risk-averse objective: mean + risk_weight * across-particle std (0 = stock risk-neutral mean).
    # The std runs ~25x the mean cost here, so useful values are small: ~0.005-0.02.
    p.add_argument("--risk_weight", type=float, default=0.0,
                   help="PARTICLE SPREAD PENTALTY. weight on imagined-outcome spread in the objective (0 = disabled)")
    # cost-shaping terms (see penicillin_cost.PeniConcentrationCost). constraint_strength default is
    # 0.75, not the old 1.5: the 1.5 sweep predates the sigma_n^2 predictive-variance fix and a
    # 3-seed re-sweep found cs is not a significant lever in [0.5,1.5] (see config_single_phase.py's
    # constraint_strength comment and results/single_phase_baseline/retune_sweep_stage2_across_seeds.csv).
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
                        "in PROBE_PLAN (slow dither, fast dither, sustained +/- production-window "
                        "steps); higher counts replicate the set under fresh batch realisations "
                        "(0 = disabled)")
    p.add_argument("--pms_visc_delay", action="store_true",
                   help="use the MC-PILCO4PMS-style delayed/held Viscosity measurement (12h "
                        "sampling + 4h analysis delay) as control_policy's input, instead of the "
                        "true instantaneous value (default: off, matching config_single_phase.py's "
                        "current default)")
    p.add_argument("--use_offline_measurements", action="store_true",
                   help="read Viscosity from the simulator's delayed lab-assay proxy (12h "
                        "sampling + 4h analysis delay) EVERYWHERE -- GP training, cost, and the "
                        "policy alike, no true-vs-measured split -- instead of the always-live "
                        "online value. Mutually exclusive with --pms_visc_delay (PenSimWrapper "
                        "raises if both are set). Default: off (plain online Viscosity). NOTE: CER "
                        "is unaffected either way -- it is an online off-gas channel with no lab "
                        "assay and no delay, which is a large part of why it is worth testing.")
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
                        "(default: 5)")
    p.add_argument("--no_cer", dest="use_cer", action="store_false",
                   help="run the plain 5-channel state instead, so this driver can produce the "
                        "A/B control itself under identical code paths (equivalent to "
                        "02_mcpilco_single_phase_baseline.py, but logged under this tree)")
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
         state_names=CER_STATE_NAMES if args.use_cer else None)

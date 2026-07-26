"""BO-driven reward-hacking probe: does optimizing a (proxy) cost as hard as a real optimizer can
actually manage lead to worse real yield than optimizing yield directly, over the exact same
action space? Compares 4 named reward-function candidates side by side.

Why this is a different, stronger test than cost_offmanifold_probe.py
-----------------------------------------------------------------------
That script samples the 5-segment action space randomly (plus a few hand-designed shapes) and
checks whether any sampled pair happens to disagree with yield. Finding one is real evidence a
hack CAN exist; not finding one is weak evidence, because 30 samples cover almost none of a
continuous 5-D space -- an unlucky search proves nothing about what an optimizer would find.

Reward hacking is not "a bad trajectory exists somewhere". It's "the thing whose job is to
minimize this cost, minimizing it as hard as it can, ends up somewhere bad". That is a claim
about what an OPTIMIZER converges to, not about what random sampling happens to hit. So this
script points a real optimizer -- `gp_minimize`, the same Bayesian optimizer already used in
experiments/01_bo_baseline_adapted_action.ipynb to find real high-yield recipes -- directly at
each cost function, and reads off what it converges to.

The test
--------
5 BO runs, same 5-segment Fs action space (via cost_offmanifold_probe.segment_policy), same
search budget:

  A. maximize real batch_yield_kg                       -- a ceiling CANDIDATE
  B. minimize PeniConcentrationDenseCost   (p_weight * P)                          -- dense concentration
  C. minimize PeniConcentrationChangeCost  (p_weight * (P_t - P_(t-1)))            -- dense concentration change
  D. minimize PeniMassTerminalCost         (p_weight * P_T*Wt_T/1000, sparse)      -- terminal mass
  E. minimize PeniMassChangeCost           (p_weight * dmass [+ discharge credit]) -- dense mass change

All 4 classes live in mcpilco/penicillin_cost.py (not re-implemented here), so this always scores
the literal code a real training run would use, not a copy that can silently drift -- see that
file for exact formulas. IMPORTANT ASYMMETRY, so it isn't misread as an oversight: only D and E's
successor E carries a discharge/harvest credit, and only because `harvest_reward=True` is passed
to it specifically -- B, C, and D never get one (D is deliberately naive, see its docstring in
penicillin_cost.py for why: it exists to demonstrate the "ignores discharge" failure mode as a
contrast against E). This is NOT "no arm has discharge credit for a fair comparison" -- it's a
DELIBERATE asymmetry, because the 4 candidates are testing 4 different hypotheses, not 4
implementations of the same idea with one variable changed.

Each winning profile is then re-evaluated on held-out seeds for a noise-robust final readout. The
REPORTED CEILING is the best-performing candidate among all 6 evaluated (pooled search pick,
dedicated yield-only winner, and all 4 variant winners) -- selected and measured on two DISJOINT
seed blocks, see point 3 below for why that split is not optional.

Four problems found and fixed across this script's revisions (kept here because the reasoning
matters for anyone re-tuning the seed counts or adding a 5th candidate later)
------------------------------------------------------------------------------------------------
1. SEARCH-SEED OVERFITTING. An early version scored every BO candidate on ONE search seed. A
   "pooled ceiling" taken as the best-of-all-evaluated-points on that single seed reached 4,198 kg
   in-search, but validated at only 3,640 +/- 1,093 kg across held-out seeds -- it had simply
   gotten lucky on that one seed and did badly elsewhere. Fix: every BO objective now averages
   over SEARCH_SEEDS (several seeds), so a point that's only good on one seed scores worse on
   average and the optimizer stops chasing it.
2. NO SIGNIFICANCE TEST ON THE GAP. An early verdict compared the ceiling-vs-candidate gap to a
   flat 5%/15% threshold and ignored the seed-to-seed standard error entirely -- so it printed
   "PASS" on gaps that were well within noise. Fix: the verdict is now a paired t-test (same tool
   used in cost_shape.ipynb C.2 and cost_offmanifold_probe.py) across held-out seeds, Bonferroni-
   corrected for the N_COMPARISONS run.
3. WINNER'S-CURSE IN THE CEILING SELECTION. Even with (1)+(2) fixed: picking the ceiling as
   whichever of 6 candidates has the best mean on a seed block, THEN testing every gap against
   that same ceiling on THE SAME seed block, is selecting on the data you then test on. The
   ceiling's measured mean is biased upward by construction -- with ~1,000 kg seed-to-seed spread
   and ~20 eval seeds (SE~224 kg), the expected inflation from taking a max over 6 noisy estimates
   is roughly SE * 1.27 ~ 285 kg, which is AS LARGE AS the ~166 kg gap an earlier version of this
   script was already treating as a real effect. Fix: eval seeds are split into two DISJOINT
   blocks -- CEILING_PICK_SEEDS decides which candidate is the ceiling, TEST_SEEDS (never used in
   that decision) is what every gap is actually measured and tested against. Same total seed
   budget as before (the split is 50/50 of --n_eval_seeds), not an added cost, but it does halve
   the per-block sample size, so consider raising --n_eval_seeds if the old power level matters.
4. NO ENVELOPE CHECK ON THE WINNING PROFILES. batch_yield_kg has no notion of the operating
   envelope (Visc < VISC_MAX, Wt in range) baked in, and neither did this script's readout -- so a
   candidate (including the ceiling itself) could "win" by finding a trajectory that pushes
   viscosity over the limit, and nothing here would flag it. A PASS was never actually a guarantee
   of an operationally valid batch. Fix: every one of the 6 final candidates now also reports mean
   max-Viscosity and max-Wt across its test seeds, and how many of those seeds breach VISC_MAX /
   WT_OVERFLOW -- printed for the ceiling too, since that's the most consequential place for an
   undetected violation to hide.

SEARCH_SEEDS, CEILING_PICK_SEEDS and TEST_SEEDS are three DISJOINT pools (search: 700000+; eval:
700100+, split in half). Any overlap between "seeds used to decide something" and "seeds used to
measure it" leaks optimism into the measurement -- that's the common thread behind problems 1 and
3 above, just at two different stages of the pipeline.

Cache is keyed by cost-function config (weights) AND candidate name, not just (label, x, seed) --
otherwise re-running with different --p_weight/--visc_penalty/etc, or comparing two variants that
happen to explore the same x, could silently serve a stale or wrong-variant cached value. The
cache schema also now includes max_visc/max_wt (problem 4's fix); an existing cache file written
before this version won't have those columns, so it's treated as incompatible and started fresh
rather than silently reporting missing envelope data as if it had been checked.

Honesty caveat (same theme as cost_offmanifold_probe.py)
----------------------------------------------------------
This searches OPEN-LOOP piecewise trajectories (5 numbers fixed before the batch starts), not
MC-PILCO's actual CLOSED-LOOP policy space (a network reacting to state every decision). Closed-
loop policies are strictly more expressive: any open-loop trajectory is reproducible by a closed-
loop policy that happens to ignore the state. So a hack found here is guaranteed reachable by the
real policy too -- this is a LOWER BOUND on the risk, not an upper bound. Failing to find a
significant gap here does not certify the closed-loop policy is safe, only that this particular
restricted space doesn't contain an obvious exploit.

Usage
-----
    python experiments/cost_reward_hacking_bo.py
    python experiments/cost_reward_hacking_bo.py --n_calls 40 --n_search_seeds 2 --n_eval_seeds 10

Cost: search rollouts = n_calls * n_search_seeds * 5 runs; eval rollouts = n_eval_seeds * 6
candidates (pooled ceiling, dedicated-yield winner, 4 variant winners) -- unchanged by the
ceiling-pick/test split, since that only partitions the existing eval seeds, it doesn't add more.
Defaults (100, 10, 3, 20) run ~1620 rollouts total. Try a smaller budget first before committing.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from skopt import gp_minimize
from skopt.space import Real

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, FS_SCALE,
                                    VISC_MAX, WT_OVERFLOW, batch_yield_kg)
from mcpilco.penicillin_cost import (PeniConcentrationDenseCost, PeniConcentrationChangeCost,
                                     PeniMassTerminalCost, PeniMassChangeCost)
from pensim_mcpilco.evaluations.cost_offmanifold_probe import segment_policy, score, N_DECISIONS

ALPHA = 0.05
N_COMPARISONS = 4  # one per reward-function variant -- for Bonferroni correction

# name -> class, in the order they should print. All 4 take the same cost_kwargs constructor
# signature (p_weight, soft_penalty, rate_penalty, visc_penalty, harvest_reward) inherited from
# PeniConcentrationCost -- see mcpilco/penicillin_cost.py for exact formulas and the reasoning
# behind each (naive terminal mass vs harvest-aware mass-change is a deliberate contrast pair).
VARIANTS = {
    "concentration_dense": PeniConcentrationDenseCost,
    "concentration_change": PeniConcentrationChangeCost,
    "mass_terminal": PeniMassTerminalCost,
    "mass_change_discharge": PeniMassChangeCost,
}

# Three disjoint pools -- see "SEARCH-SEED OVERFITTING" and "WINNER'S-CURSE" in the module
# docstring for why each split matters. CEILING_PICK and TEST are two halves of one eval pool,
# sliced apart in main() by --n_eval_seeds, not separate CLI-sized budgets.
SEARCH_SEED_POOL = [700000 + i for i in range(5)]
EVAL_SEED_POOL = [700100 + i for i in range(50)]

CACHE_COLUMNS = ["label", "x", "seed", "yield_kg", "cost", "max_visc", "max_wt"]


def rollout(x, seed, n_segments):
    seg_len = N_DECISIONS // n_segments
    policy = segment_policy(x, seg_len)
    wrapper = PenSimWrapper()
    with contextlib.redirect_stdout(io.StringIO()):
        st, ins, _ = wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0, seed=int(seed))
    mon = wrapper.monitor[-1]
    return st, ins, mon


def cost_of(cost_obj, st, ins):
    t = score(cost_obj, st, ins)
    return -t["reward"] + t["soft"] + t["visc_soft"] + t["action_rate"]


class Cache:
    """BO with a fixed random_state re-queries the same points on many reruns, so caching by
    (label, rounded x, seed) makes iteration much cheaper -- same reasoning as the CSV caches in
    cost_shape.ipynb / cost_offmanifold_probe.py, keyed on the query point since x is continuous
    rather than a grid index. `label` must fully identify anything that changes the cached
    mapping -- see CONFIG_TAG in main(), which folds the cost weights AND the variant name into
    every label so neither a changed weight nor a mixed-up variant can silently serve stale data.

    Schema includes max_visc/max_wt (envelope tracking -- see module docstring problem 4). A cache
    file written before this column existed is INCOMPATIBLE, not just incomplete: silently
    treating those old rows as cached would report "no envelope data" as if it had been checked
    and found clean. So an old-schema file is discarded (fresh start) rather than partially reused.
    """
    def __init__(self, path):
        self.path = path
        if path.exists():
            df = pd.read_csv(path)
            if not set(CACHE_COLUMNS).issubset(df.columns):
                print(f"[cache] {path.name} predates envelope tracking (missing max_visc/max_wt) "
                      f"-- starting fresh rather than reusing rows with no envelope data.")
                df = pd.DataFrame(columns=CACHE_COLUMNS)
        else:
            df = pd.DataFrame(columns=CACHE_COLUMNS)
        self.df = df
        self._rows = {(r.label, r.x, r.seed): (r.yield_kg, r.cost, r.max_visc, r.max_wt)
                      for r in self.df.itertuples()}
        self._new = []

    @staticmethod
    def _key_x(x):
        return "|".join(f"{v:.6f}" for v in x)

    def get(self, label, x, seed):
        return self._rows.get((label, self._key_x(x), int(seed)))

    def put(self, label, x, seed, yield_kg, cost, max_visc, max_wt):
        key_x = self._key_x(x)
        self._rows[(label, key_x, int(seed))] = (yield_kg, cost, max_visc, max_wt)
        self._new.append({"label": label, "x": key_x, "seed": int(seed), "yield_kg": yield_kg,
                          "cost": cost if cost is not None else np.nan,
                          "max_visc": max_visc, "max_wt": max_wt})

    def flush(self):
        if self._new:
            new_df = pd.DataFrame(self._new)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(self.path, index=False)
            self._new = []


def eval_seeds(x, seeds, n_segments, cache, label, cost_obj=None):
    """Roll out x on every seed in `seeds`, using the cache per-seed so partial averages are
    reusable. Returns a dict of means and per-seed lists, including envelope data (max_visc,
    max_wt) for every seed, AND an envelope-VALID yield series (`env_valid_ys`) that zeroes out
    any seed that breaches VISC_MAX or WT_OVERFLOW.

    Zeroed, not excluded/NaN'd: this keeps every candidate's per-seed array the same length and
    aligned by seed, which paired_verdict's paired t-test requires -- excluding a breaching seed
    for one candidate but not another would desync the pairing. Zero also has a defensible reading
    on its own: an envelope-breaching batch is operationally rejected, i.e. contributes no usable
    product. `env_valid_ys`/`mean_env_valid_yield` is what ceiling selection and the paired verdict
    actually use (see main()) -- `ys`/`mean_yield` (raw, unmasked) is kept only for side-by-side
    reporting, never for a decision. See module docstring problem 4 / the envelope-gating fix."""
    ys, cs, viscs, wts = [], [], [], []
    for s in seeds:
        cached = cache.get(label, x, s)
        if cached is not None:
            y, c, mv, mw = cached
        else:
            st, ins, mon = rollout(x, s, n_segments)
            y = batch_yield_kg(mon)
            c = cost_of(cost_obj, st, ins) if cost_obj is not None else None
            mv = float(np.max(mon["Viscosity"]))
            mw = float(np.max(mon["Wt"]))
            cache.put(label, x, s, y, c, mv, mw)
        ys.append(y); cs.append(c); viscs.append(mv); wts.append(mw)
    env_valid_ys = [0.0 if (v >= VISC_MAX or w >= WT_OVERFLOW) else y
                   for y, v, w in zip(ys, viscs, wts)]
    return {
        "mean_yield": float(np.mean(ys)), "ys": ys,
        "env_valid_ys": env_valid_ys, "mean_env_valid_yield": float(np.mean(env_valid_ys)),
        "mean_cost": float(np.mean(cs)) if cost_obj is not None else None, "cs": cs,
        "mean_visc": float(np.mean(viscs)), "viscs": viscs,
        "mean_wt": float(np.mean(wts)), "wts": wts,
        "n_visc_breach": sum(v >= VISC_MAX for v in viscs),
        "n_wt_breach": sum(w >= WT_OVERFLOW for w in wts),
    }


def make_objective(label, cache, n_segments, search_seeds, cost_obj, minimize_cost, log):
    """cost_obj=None -> objective is -mean(yield) over search_seeds (BO maximizes real yield, the
    ceiling run). cost_obj set -> objective is mean(cost) over search_seeds (BO minimizes the
    proxy). Averaging over search_seeds -- rather than scoring on one seed -- is what stops the
    optimizer from converging on a point that's only good by luck on a single realization; see the
    module docstring's "SEARCH-SEED OVERFITTING" note for why this matters."""
    def objective(x):
        r = eval_seeds(x, search_seeds, n_segments, cache, label, cost_obj)
        log.append({"x": list(x), "yield_kg": r["mean_yield"], "cost": r["mean_cost"]})
        return r["mean_cost"] if minimize_cost else -r["mean_yield"]
    return objective


def run_bo(objective, n_segments, n_calls, n_random, seed=0):
    space = [Real(-1.0, 1.0, name=f"seg{k}") for k in range(n_segments)]
    return gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_random,
                       acq_func="EI", random_state=seed, noise=1e-10)


def convergence_trace(name, log, minimize):
    """Best-value-so-far at each quartile of the call budget, so a bad final result can be told
    apart from a proxy whose landscape BO simply hadn't converged on within budget -- a harder
    surface to search produces a worse a*_B for reasons that are about the SEARCH, not about
    alignment with yield, and a flat plateau vs a still-improving tail look identical in the final
    number alone. Uses `cost` for minimize=True runs (proxy searches) and `yield_kg` for
    minimize=False (the ceiling search)."""
    key = "cost" if minimize else "yield_kg"
    vals = [row[key] for row in log]
    if minimize:
        running_best = np.minimum.accumulate(vals)
    else:
        running_best = np.maximum.accumulate(vals)
    n = len(running_best)
    checkpoints = sorted(set(max(1, round(n * f)) - 1 for f in (0.25, 0.5, 0.75, 1.0)))
    trace = " -> ".join(f"{running_best[i]:,.2f}" for i in checkpoints)
    # last 20% improvement relative to the value at the 80% mark
    i80 = max(0, round(n * 0.8) - 1)
    tail_improve = abs(running_best[-1] - running_best[i80]) / (abs(running_best[i80]) + 1e-9)
    flag = "STILL IMPROVING" if tail_improve > 0.01 else "converged"
    print(f"  {name:24} best@25/50/75/100%: {trace}   [{flag}, last-20%-move={100*tail_improve:.1f}%]")
    return running_best


def paired_verdict(ceiling_ys, cand_ys, ceiling_mean, cand_mean, alpha_bonf, name):
    gap = ceiling_mean - cand_mean
    pct = 100 * gap / ceiling_mean
    _, p = stats.ttest_rel(ceiling_ys, cand_ys)
    if np.isnan(p):
        verdict = "INCONCLUSIVE"
        print(f"[INCONCLUSIVE] {name}: gap {gap:+,.0f} kg ({pct:+.1f}%) -- paired t-test undefined "
              f"(near-zero variance in the differences; likely this candidate IS the ceiling, or "
              f"converged to almost the same trajectory).")
    elif p >= alpha_bonf:
        verdict = "INCONCLUSIVE"
        print(f"[INCONCLUSIVE] {name}: gap {gap:+,.0f} kg ({pct:+.1f}%) not distinguishable from "
              f"seed-to-seed noise at alpha={alpha_bonf:.4f} (p={p:.3f}, n={len(ceiling_ys)} test "
              f"seeds). Need more eval seeds or a bigger search budget to resolve this.")
    elif gap <= 0:
        verdict = "PASS"
        print(f"[PASS] {name}: gap {gap:+,.0f} kg -- statistically no worse than the ceiling "
              f"(p={p:.2e}), or actually better, at this search budget.")
    else:
        verdict = "FAIL"
        print(f"[FAIL] {name}: statistically real gap of {gap:,.0f} kg ({pct:.1f}%) vs ceiling "
              f"(p={p:.2e}, survives Bonferroni correction) -- concrete reward-hacking evidence, "
              f"not noise.")
    return {"gap_kg": gap, "gap_pct": pct, "p_value": p, "verdict": verdict}


def envelope_line(name, r, is_ceiling):
    breach = r["n_visc_breach"] or r["n_wt_breach"]
    tag = " <- CEILING" if is_ceiling else ""
    flag = "  [ENVELOPE BREACH]" if breach else ""
    print(f"  {name:24} max_visc={r['mean_visc']:>6.1f} (breach {r['n_visc_breach']}/{len(r['viscs'])})  "
          f"max_wt={r['mean_wt']:>9,.0f} (breach {r['n_wt_breach']}/{len(r['wts'])}){flag}{tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_segments", type=int, default=5)
    p.add_argument("--n_calls", type=int, default=100)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--n_search_seeds", type=int, default=3,
                   help="seeds averaged INSIDE the BO objective (from SEARCH_SEED_POOL)")
    p.add_argument("--n_eval_seeds", type=int, default=20,
                   help="held-out seeds for the final readout, SPLIT in half between ceiling "
                        "selection and testing (from EVAL_SEED_POOL, disjoint from search seeds)")
    p.add_argument("--p_weight", type=float, default=0.05)
    p.add_argument("--soft_penalty", type=float, default=0.5)
    p.add_argument("--rate_penalty", type=float, default=0.5)
    p.add_argument("--visc_penalty", type=float, default=0.5)
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false")
    args = p.parse_args()

    assert FS_SCALE != 0, "FS_SCALE is 0 -- every profile collapses to the recipe"
    assert args.n_search_seeds <= len(SEARCH_SEED_POOL)
    assert args.n_eval_seeds <= len(EVAL_SEED_POOL)
    assert args.n_eval_seeds >= 2, "need at least 2 eval seeds to split into ceiling-pick/test"

    search_seeds = SEARCH_SEED_POOL[:args.n_search_seeds]
    eval_pool = EVAL_SEED_POOL[:args.n_eval_seeds]
    half = len(eval_pool) // 2
    ceiling_pick_seeds, test_seeds = eval_pool[:half], eval_pool[half:]

    cost_kwargs = dict(p_weight=args.p_weight, soft_penalty=args.soft_penalty,
                       rate_penalty=args.rate_penalty, visc_penalty=args.visc_penalty,
                       harvest_reward=args.harvest_reward)
    CONFIG_TAG = "|".join(f"{k}={v}" for k, v in sorted(cost_kwargs.items()))

    cache = Cache(Path(_ROOT) / "results" / "cost_reward_hacking_bo_cache.csv")
    logs = {}

    print(f"BO search: n_segments={args.n_segments} n_calls={args.n_calls} n_random={args.n_random} "
          f"search_seeds={search_seeds}")
    print(f"eval: {len(ceiling_pick_seeds)} ceiling-pick seeds + {len(test_seeds)} test seeds "
          f"(disjoint) | config={CONFIG_TAG}")

    print("\nRun A: maximize real batch_yield_kg (a ceiling candidate)...")
    logs["yield"] = []
    obj_yield = make_objective(f"yield|{CONFIG_TAG}", cache, args.n_segments, search_seeds,
                               None, minimize_cost=False, log=logs["yield"])
    res_yield = run_bo(obj_yield, args.n_segments, args.n_calls, args.n_random)
    cache.flush()

    variant_res = {}
    for name, cls in VARIANTS.items():
        print(f"Run: minimize {cls.__name__} ({name})...")
        logs[name] = []
        cost_obj = cls(**cost_kwargs)
        obj = make_objective(f"{name}|{CONFIG_TAG}", cache, args.n_segments, search_seeds,
                             cost_obj, minimize_cost=True, log=logs[name])
        variant_res[name] = run_bo(obj, args.n_segments, args.n_calls, args.n_random)
        cache.flush()

    print("\nConvergence check (best-so-far at 25/50/75/100% of the call budget):")
    convergence = {"yield": convergence_trace("yield (ceiling search)", logs["yield"], minimize=False)}
    for name in VARIANTS:
        convergence[name] = convergence_trace(name, logs[name], minimize=True)

    # Pooled search-time pick: best of all points evaluated across all 5 runs, by SEARCH-SEED-
    # AVERAGED yield. Still only a CANDIDATE ceiling -- final selection happens below, on a
    # disjoint eval block (see "WINNER'S-CURSE" in the module docstring).
    all_points = logs["yield"] + [pt for name in VARIANTS for pt in logs[name]]
    pooled_best = max(all_points, key=lambda r: r["yield_kg"])
    print(f"\nPooled search-time pick: best of {len(all_points)} points evaluated across all 5 "
          f"runs (mean over {len(search_seeds)} search seeds) -> {pooled_best['yield_kg']:,.0f} kg "
          f"at x={[round(v, 3) for v in pooled_best['x']]}")

    candidates_x = {"pooled": pooled_best["x"], "dedicated_yield": res_yield.x,
                    **{name: variant_res[name].x for name in VARIANTS}}

    print(f"\nSelecting the ceiling on {len(ceiling_pick_seeds)} ceiling-pick seeds "
          f"(never used for testing), on ENVELOPE-VALID yield -- a breaching batch scores 0 here, "
          f"so a candidate can't win the ceiling slot by overfeeding into a viscosity/Wt breach...")
    ceiling_pick_results = {name: eval_seeds(x, ceiling_pick_seeds, args.n_segments, cache,
                                             f"{name}_ceilingpick|{CONFIG_TAG}")
                            for name, x in candidates_x.items()}
    ceiling_name = max(ceiling_pick_results,
                       key=lambda k: ceiling_pick_results[k]["mean_env_valid_yield"])
    cache.flush()
    cp = ceiling_pick_results[ceiling_name]
    print(f"  -> '{ceiling_name}' selected (envelope-valid mean {cp['mean_env_valid_yield']:,.0f} kg, "
          f"raw mean {cp['mean_yield']:,.0f} kg, on the ceiling-pick block)")

    print(f"\nMeasuring every candidate on {len(test_seeds)} DISJOINT test seeds "
          f"(including the selected ceiling)...")
    test_results = {name: eval_seeds(x, test_seeds, args.n_segments, cache,
                                     f"{name}_test|{CONFIG_TAG}")
                    for name, x in candidates_x.items()}
    cache.flush()

    # Every decision below (the printed CEILING number and paired_verdict) uses ENV-VALID yield,
    # not raw -- raw is reported alongside for transparency only, never as what selects or tests
    # anything. See eval_seeds' docstring / module docstring problem 4 for why raw would let a
    # candidate "win" or "beat the ceiling" by overfeeding into a breach.
    ceiling_mean = test_results[ceiling_name]["mean_env_valid_yield"]
    ceiling_ys = test_results[ceiling_name]["env_valid_ys"]

    print("\n" + "=" * 80)
    print(f"[CEILING] '{ceiling_name}', envelope-valid on the disjoint test block: "
          f"{ceiling_mean:>7,.0f} +/- {float(np.std(ceiling_ys)):,.0f} kg")
    for label, r in test_results.items():
        tag = " <- CEILING" if label == ceiling_name else ""
        print(f"  {label:24} env-valid {r['mean_env_valid_yield']:>7,.0f} +/- "
              f"{float(np.std(r['env_valid_ys'])):,.0f} kg   (raw {r['mean_yield']:>7,.0f} +/- "
              f"{float(np.std(r['ys'])):,.0f} kg){tag}")

    print(f"\nOperating envelope on the test block (VISC_MAX={VISC_MAX:g}, WT_OVERFLOW={WT_OVERFLOW:g}):")
    for label, r in test_results.items():
        envelope_line(label, r, is_ceiling=(label == ceiling_name))

    alpha_bonf = ALPHA / N_COMPARISONS
    print(f"\nPaired significance test (on envelope-valid yield) across {len(test_seeds)} DISJOINT "
          f"test seeds (Bonferroni alpha={alpha_bonf:.4f} for {N_COMPARISONS} comparisons):")
    verdicts = {}
    for name in VARIANTS:
        r = test_results[name]
        verdicts[name] = paired_verdict(ceiling_ys, r["env_valid_ys"], ceiling_mean,
                                        r["mean_env_valid_yield"], alpha_bonf, name)
    print("=" * 80)

    print(f"\nWinning profiles (per-segment action, seg 0..{args.n_segments - 1}):")
    for name, x in candidates_x.items():
        print(f"  {name:24}: {[round(v, 3) for v in x]}")

    # --- CSV export ---------------------------------------------------------------------
    results_dir = Path(_ROOT) / "results"
    summary_rows = []
    for name, r in test_results.items():
        v = verdicts.get(name, {})
        summary_rows.append({
            "name": name, "is_ceiling": name == ceiling_name,
            "mean_yield_raw": r["mean_yield"], "sd_yield_raw": float(np.std(r["ys"])),
            "mean_yield_env_valid": r["mean_env_valid_yield"],
            "sd_yield_env_valid": float(np.std(r["env_valid_ys"])),
            "mean_visc": r["mean_visc"], "n_visc_breach": r["n_visc_breach"],
            "mean_wt": r["mean_wt"], "n_wt_breach": r["n_wt_breach"],
            "n_test_seeds": len(test_seeds),
            "gap_kg": v.get("gap_kg"), "gap_pct": v.get("gap_pct"),
            "p_value": v.get("p_value"), "verdict": v.get("verdict"),
            "x": ",".join(f"{c:.4f}" for c in candidates_x[name]),
            "config": CONFIG_TAG, "n_calls": args.n_calls, "n_search_seeds": args.n_search_seeds,
        })
    summary_path = results_dir / "cost_reward_hacking_bo_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n[saved] {summary_path}")

    convergence_rows = []
    for name, running_best in convergence.items():
        for call_idx, val in enumerate(running_best):
            convergence_rows.append({"run": name, "call_idx": call_idx, "running_best": val,
                                     "config": CONFIG_TAG})
    convergence_path = results_dir / "cost_reward_hacking_bo_convergence.csv"
    pd.DataFrame(convergence_rows).to_csv(convergence_path, index=False)
    print(f"[saved] {convergence_path}")


if __name__ == "__main__":
    main()

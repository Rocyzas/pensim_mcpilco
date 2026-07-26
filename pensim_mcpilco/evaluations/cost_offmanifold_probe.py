"""Off-manifold / adversarial validity probe for PeniConcentrationCost.

Why this exists
----------------
`evaluations/cost_shape.ipynb` validates the cost along exactly ONE axis: a constant-in-time
Fs scale factor `a`, held fixed for the whole batch, at 21 grid points. That is a real test and
it has teeth (see its own C.5 positive control) -- but it is a 1-D probe of a trajectory space
the closed-loop MC-PILCO policy is free to leave. A constant-a rescaling moves broth mass (Wt)
and penicillin concentration (P) together, so a test confined to that manifold structurally
cannot see the one failure a concentration-only reward is most exposed to: G/L rising while
total KG falls. Passing cost_shape.ipynb is necessary, not sufficient.

This script complements it, it does not replace it:

  1. OFF-MANIFOLD SAMPLE -- piecewise-constant Fs(t) policies (independent action per time
     segment), both a handful of hand-designed shapes chosen to try to decouple P from Wt
     (front-load-then-cut, cut-then-boost, spikes...) and a batch of random segment profiles.
     None of these are reachable by any constant `a`.
  2. RANKING on that sample, same statistic as cost_shape.ipynb's C.2 (paired t-test across
     seeds, concordance on resolvable pairs) -- but here the concordance number is a means to
     the next point, not the headline.
  3. ADVERSARIAL HUNT -- explicitly searches the resolvable pairs for the worst offender: the
     pair where the cost's preference disagrees with yield most strongly. cost_shape.ipynb
     reports an aggregate rate; this reports the single most damning counterexample, if any.
  4. ENVELOPE CHECK -- flags every batch that actually breaches VISC_MAX or WT_OVERFLOW.
     cost_shape.ipynb's C.6 verdict never looks at this; a cost that ranks yield correctly while
     preferring constraint-violating batches would sail through it undetected.
  5. MASS vs CONCENTRATION -- every batch is also scored with `MassCost` (the P*Wt/1000 formula
     already sketched, commented, in penicillin_cost.py:122-123 -- copied here verbatim so it
     can't silently drift from what that file actually contains). Where the two costs rank a
     pair differently, that pair IS a live instance of concentration/mass divergence, found on
     real simulator trajectories rather than argued for.

Usage
-----
    python experiments/cost_offmanifold_probe.py
    python experiments/cost_offmanifold_probe.py --n_random 40 --n_segments 5
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from mcpilco.pensim_wrapper import (PenSimWrapper, CONTROL_H, T_SAMPLING, STATE_RANGES,
                                    FS_SCALE, VISC_MAX, WT_OVERFLOW, batch_yield_kg,
                                    decode_state_value)
from mcpilco.penicillin_cost import PeniConcentrationCost, P_IDX, WT_IDX, P_MAX, WT_MAX

ALPHA = 0.05
SEEDS = [700000 + i for i in range(5)]
N_DECISIONS = int(CONTROL_H / T_SAMPLING)  # 45 at CONTROL_H=229.8, T_SAMPLING=5.0

# Hand-designed profiles, one value per segment, each in [-1, 1] -- the same scale as
# cost_shape.ipynb's constant `a`, so results are directly comparable to that sweep's range.
# Each tries a specific mechanism for decoupling P from Wt under a single feed-rate action.
STRUCTURED = {
    "front_boost_late_cut": [1.0, 1.0, 0.0, -1.0, -1.0],   # grow early, arrest dilution late
    "cut_then_boost":       [-1.0, -1.0, 0.0, 1.0, 1.0],   # starve early, catch up late
    "spike_end":            [-1.0, -1.0, -1.0, -1.0, 1.0], # minimal feed, terminal spike
    "spike_start":          [1.0, -1.0, -1.0, -1.0, -1.0], # terminal starvation after a burst
    "mid_spike":            [-1.0, 1.0, 1.0, -1.0, -1.0],
    "sawtooth":             [1.0, -1.0, 1.0, -1.0, 1.0],
    "recipe_then_starve":   [0.0, 0.0, 0.0, -1.0, -1.0],   # closest to "cut dilution near the end"
    "recipe_then_boost":    [0.0, 0.0, 0.0, 1.0, 1.0],
}


class MassCost(PeniConcentrationCost):
    """Reference-only reward: kg-in-tank (P * Wt), NOT what the trainer currently optimises.

    Formula copied verbatim from the commented SEED2_6 block in penicillin_cost.py:122-123 so it
    cannot silently drift from what that file actually contains. Used here purely as a real,
    non-degenerate second opinion -- unlike C.5's sign-flipped InvertedCost, this is a cost
    someone could plausibly have shipped, so disagreement with it is a live finding, not a
    manufactured one.
    """
    def _terms(self, states_sequence, inputs_sequence, trial_index=None):
        t = super()._terms(states_sequence, inputs_sequence, trial_index)
        P = decode_state_value("P", self._dn(states_sequence[:, :, P_IDX], *STATE_RANGES["P"]))
        Wt = decode_state_value("Wt", self._dn(states_sequence[:, :, WT_IDX], *STATE_RANGES["Wt"]))
        P = torch.clamp(P, min=0.0, max=P_MAX)
        Wt = torch.clamp(Wt, min=0.0, max=WT_MAX)
        mass = P * Wt / 1000.0
        return {**t, "reward": self.p_weight * mass}


def segment_policy(a_segs, seg_len):
    """Piecewise-constant policy: decision index i uses segment i // seg_len (clamped)."""
    a_segs = np.asarray(a_segs, dtype=float)
    n = len(a_segs)
    def policy(state, i, a_segs=a_segs, seg_len=seg_len, n=n):
        return np.array([a_segs[min(i // seg_len, n - 1)]])
    return policy


def score(cost_obj, states, inputs):
    """Sum every cost term over one batch, dropping step 0 for the same reason cost_shape.ipynb
    and cost_term_report.py do: rollout only writes states[0] for non-pid_baseline runs, and
    normalised zero is not a null state -- see either file's docstring for the full rationale."""
    st = torch.tensor(np.asarray(states), dtype=torch.float64).unsqueeze(1)
    ins = torch.tensor(np.asarray(inputs), dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        t = cost_obj._terms(st, ins)
    return {k: float(v[1:].sum()) for k, v in t.items()}


def run_sweep(n_segments, n_random, rng_seed, cost_kwargs, cache_path, refresh):
    seg_len = N_DECISIONS // n_segments
    assert seg_len * n_segments <= N_DECISIONS, "n_segments must divide N_DECISIONS evenly-ish"

    rng = np.random.RandomState(rng_seed)
    profiles = dict(STRUCTURED)
    for k in range(n_random):
        profiles[f"rand{k:02d}"] = rng.uniform(-1.0, 1.0, size=n_segments).tolist()

    config_tag = (f"offmanifold_v1|n_seg={n_segments}|" +
                  "|".join(f"{k}={v}" for k, v in sorted(cost_kwargs.items())))

    cache = pd.read_csv(cache_path) if (cache_path.exists() and not refresh) else pd.DataFrame()
    have = set()
    if len(cache):
        have = set(zip(cache.profile, cache.seed, cache.config))

    cost = PeniConcentrationCost(**cost_kwargs)
    mass_cost = MassCost(**cost_kwargs)
    wrapper = PenSimWrapper()

    todo = [(name, s) for name in profiles for s in SEEDS
            if (name, s, config_tag) not in have]
    print(f"{len(profiles)} profiles x {len(SEEDS)} seeds = {len(profiles) * len(SEEDS)} batches "
          f"| {len(todo)} to run ({len(profiles) * len(SEEDS) - len(todo)} cached) | config={config_tag}")

    import contextlib, io
    rows = []
    for i, (name, s) in enumerate(todo):
        policy = segment_policy(profiles[name], seg_len)
        with contextlib.redirect_stdout(io.StringIO()):
            st, ins, _ = wrapper.rollout(None, policy, CONTROL_H, T_SAMPLING, 0, seed=int(s))
        mon = wrapper.monitor[-1]
        t_conc = score(cost, st, ins)
        t_mass = score(mass_cost, st, ins)
        rows.append({
            "profile": name, "seed": s, "config": config_tag,
            "cost_conc": -t_conc["reward"] + t_conc["soft"] + t_conc["visc_soft"] + t_conc["action_rate"],
            "cost_mass": -t_mass["reward"] + t_mass["soft"] + t_mass["visc_soft"] + t_mass["action_rate"],
            "yield_kg": batch_yield_kg(mon),
            "max_visc": float(np.max(mon["Viscosity"])),
            "max_wt": float(np.max(mon["Wt"])),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}")

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True) if len(cache) else pd.DataFrame(rows)
        cache.to_csv(cache_path, index=False)

    df = cache[cache.config == config_tag].copy()
    return df


def paired_concordance(PY, PC, alpha):
    """Same statistic as cost_shape.ipynb C.2: paired t-test per pair across seeds, concordance
    on resolvable pairs only. Returns (pairs_df, resolvable_df, concordance)."""
    names = list(PY.columns)
    pairs = []
    for i, ai in enumerate(names):
        for aj in names[i + 1:]:
            _, p = stats.ttest_rel(PY[aj], PY[ai])
            pairs.append({"ai": ai, "aj": aj, "dy": PY[aj].mean() - PY[ai].mean(),
                          "dc": PC[aj].mean() - PC[ai].mean(), "p": p})
    pairs = pd.DataFrame(pairs)
    res = pairs[pairs.p < alpha].copy()
    res["concordant"] = np.sign(res.dy) == -np.sign(res.dc)
    conc = res.concordant.mean() if len(res) else np.nan
    return pairs, res, conc


def envelope_report(df):
    visc_bad = df[df.max_visc >= VISC_MAX]
    wt_bad = df[df.max_wt >= WT_OVERFLOW]
    bad = pd.concat([visc_bad, wt_bad]).drop_duplicates(subset=["profile", "seed"])
    print(f"\n[ENVELOPE] {len(bad)}/{len(df)} batches breach VISC_MAX={VISC_MAX:g} or "
          f"WT_OVERFLOW={WT_OVERFLOW:g}")
    if len(bad):
        by_profile = bad.groupby("profile").size().sort_values(ascending=False)
        print("           profiles with violations (n seeds):")
        for name, n in by_profile.items():
            print(f"             {name:24} {n}/{len(SEEDS)}")
        # Does the cost still charge these batches appropriately, or does yield alone flag them
        # while the cost stays indifferent? Compare mean cost of violating vs clean batches
        # against yield's own signal.
        clean = df[~df.index.isin(bad.index)]
        if len(clean):
            print(f"           mean yield  violating={bad.yield_kg.mean():,.0f} kg  "
                  f"clean={clean.yield_kg.mean():,.0f} kg")
            print(f"           mean cost   violating={bad.cost_conc.mean():.2f}       "
                  f"clean={clean.cost_conc.mean():.2f}")
            if bad.cost_conc.mean() >= clean.cost_conc.mean() and bad.yield_kg.mean() < clean.yield_kg.mean():
                print("           -> cost is HIGHER (worse) on violating batches, consistent with yield. OK.")
            else:
                print("           -> cost does NOT clearly penalise violating batches relative to clean "
                      "ones -- the C.6-style verdict would not catch this on its own.")
    return bad


def mass_vs_concentration_report(df):
    PY = df.pivot_table(index="seed", columns="profile", values="yield_kg")
    PC_conc = df.pivot_table(index="seed", columns="profile", values="cost_conc")
    PC_mass = df.pivot_table(index="seed", columns="profile", values="cost_mass")
    names = list(PY.columns)
    disagreements = []
    for i, ai in enumerate(names):
        for aj in names[i + 1:]:
            dconc = PC_conc[aj].mean() - PC_conc[ai].mean()
            dmass = PC_mass[aj].mean() - PC_mass[ai].mean()
            dy = PY[aj].mean() - PY[ai].mean()
            if np.sign(dconc) != np.sign(dmass) and dconc != 0 and dmass != 0:
                disagreements.append({"ai": ai, "aj": aj, "dy": dy, "d_conc": dconc, "d_mass": dmass})
    print(f"\n[MASS vs CONCENTRATION] {len(disagreements)}/{len(names) * (len(names) - 1) // 2} "
          f"profile pairs where the two costs disagree on which is better")
    if disagreements:
        dd = pd.DataFrame(disagreements)
        dd["agrees_with_yield_conc"] = np.sign(dd.dy) == -np.sign(dd.d_conc)
        dd["agrees_with_yield_mass"] = np.sign(dd.dy) == -np.sign(dd.d_mass)
        print(dd.round(3).to_string(index=False))
        n_conc_wrong = (~dd.agrees_with_yield_conc).sum()
        n_mass_wrong = (~dd.agrees_with_yield_mass).sum()
        print(f"           on these disagreement pairs: concentration cost sides with yield "
              f"{len(dd) - n_conc_wrong}/{len(dd)} times, mass cost {len(dd) - n_mass_wrong}/{len(dd)} times")
    else:
        print("           none found in this sample -- no evidence of concentration/mass divergence here.")
    return disagreements


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_segments", type=int, default=5)
    p.add_argument("--n_random", type=int, default=22)
    p.add_argument("--rng_seed", type=int, default=12345)
    p.add_argument("--p_weight", type=float, default=0.05)
    p.add_argument("--soft_penalty", type=float, default=0.5)
    p.add_argument("--rate_penalty", type=float, default=0.5)
    p.add_argument("--visc_penalty", type=float, default=0.5)
    p.add_argument("--no_harvest_reward", dest="harvest_reward", action="store_false")
    p.add_argument("--refresh", action="store_true", help="ignore any existing cache")
    args = p.parse_args()

    assert FS_SCALE != 0, "FS_SCALE is 0 -- every profile collapses to the recipe"

    cost_kwargs = dict(p_weight=args.p_weight, soft_penalty=args.soft_penalty,
                       rate_penalty=args.rate_penalty, visc_penalty=args.visc_penalty,
                       harvest_reward=args.harvest_reward)
    cache_path = Path(_ROOT) / "results" / "cost_offmanifold_probe_cache.csv"

    df = run_sweep(args.n_segments, args.n_random, args.rng_seed, cost_kwargs, cache_path, args.refresh)

    PY = df.pivot_table(index="seed", columns="profile", values="yield_kg")
    PC = df.pivot_table(index="seed", columns="profile", values="cost_conc")
    pairs, res, conc = paired_concordance(PY, PC, ALPHA)

    # Bonferroni-corrected robustness pass -- with 435 pairs at alpha=0.05 uncorrected, ~22 false
    # positives are expected by chance alone (this is the exact lesson cost_shape.ipynb's C.2
    # needed: see its Bonferroni cell). The uncorrected concordance number below is a useful
    # headline but a single "largest effect size" discordant pair picked from it can easily be
    # noise; only a pair that ALSO survives this correction should be trusted as a real
    # counterexample.
    alpha_bonf = ALPHA / len(pairs)
    res_bonf = pairs[pairs.p < alpha_bonf].copy()
    res_bonf["concordant"] = np.sign(res_bonf.dy) == -np.sign(res_bonf.dc)
    conc_bonf = res_bonf.concordant.mean() if len(res_bonf) else np.nan

    print("\n" + "=" * 80)
    print(f"[{'PASS' if (np.isnan(conc) or conc >= 0.95) else 'FAIL'}] OFF-MANIFOLD RANKING   "
          f"{100 * conc:.1f}% concordance on {len(res)} resolvable pairs (of {len(pairs)} total, "
          f"alpha={ALPHA})")
    print(f"[{'PASS' if (np.isnan(conc_bonf) or conc_bonf >= 0.95) else 'FAIL'}] "
          f"  Bonferroni-corrected   {100 * conc_bonf:.1f}% concordance on {len(res_bonf)} "
          f"resolvable pairs (alpha={alpha_bonf:.2e})")

    discordant_bonf = res_bonf[~res_bonf.concordant].copy() if len(res_bonf) else res_bonf
    discordant = res[~res.concordant].copy() if len(res) else res
    if len(discordant_bonf):
        worst = discordant_bonf.sort_values("p").iloc[0]
        print(f"[FOUND]    ROBUST ADVERSARIAL EXAMPLE   {worst.ai} vs {worst.aj}: "
              f"dy={worst.dy:+.0f} kg, dc={worst.dc:+.3f}, p={worst.p:.2e} "
              f"(survives Bonferroni correction)")
        print("           -> the cost prefers the LOWER-yield profile here, and this is not "
              "explainable by the 435-comparisons multiple-testing problem.")
    elif len(discordant):
        discordant["severity"] = discordant.dy.abs()
        worst = discordant.sort_values("severity", ascending=False).iloc[0]
        print(f"[WEAK]     largest uncorrected-significant discordant pair: {worst.ai} vs {worst.aj}: "
              f"dy={worst.dy:+.0f} kg, dc={worst.dc:+.3f}, p={worst.p:.4f} -- does NOT survive "
              f"Bonferroni correction (needs p<{alpha_bonf:.2e}), so treat as suggestive, not proof.")
    else:
        print(f"[NONE]     ADVERSARIAL EXAMPLE   none found among {len(res)} resolvable off-manifold "
              f"pairs (n_random={args.n_random}, n_segments={args.n_segments}).")
        print("           -> no counterexample in this sample; does not prove none exists elsewhere "
              "in the policy space, only that this search didn't find one.")

    envelope_report(df)
    mass_vs_concentration_report(df)
    print("=" * 80)


if __name__ == "__main__":
    main()

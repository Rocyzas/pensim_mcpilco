# `cost_reward_hacking_bo.py` — results

Run configuration: `n_calls=100`, `n_random=10`, `n_search_seeds=3`, `n_eval_seeds=20` (split 10
ceiling-pick / 10 test), `p_weight=0.05`, `soft_penalty=0.5`, `rate_penalty=0.5`,
`visc_penalty=0.5` (all three penalty weights inert — `soft`/`visc_soft`/`action_rate` are
hard-zeroed in every candidate; see `cost_reward_hacking_bo.md`), `harvest_reward=True`.

For what each column means and why the pipeline is built this way (envelope gating, the
disjoint-seed-block ceiling selection, Bonferroni correction), see `cost_reward_hacking_bo.md` in
this directory. This file is the results only.

## Headline

**2 of 4 reward functions are safe, 2 fail, and both failures are the same mechanism**: driving
the batch into viscosity collapse, invisible if you only look at raw yield.

| candidate | verdict | envelope-valid yield (kg) | raw yield (kg) | gap vs ceiling | p-value | visc breaches |
|---|---|---|---|---|---|---|
| `concentration_dense` | **safe** (inconclusive, non-sig.) | 3,892 ± 272 | 3,892 ± 272 | −11 kg (−0.3%) | 0.516 | 0/10 |
| `mass_change_discharge` | **safe** (identical to ceiling) | 3,880 ± 268 | 3,880 ± 268 | 0 kg (0.0%) | undefined (zero variance) | 0/10 |
| `concentration_change` | **FAILS** | 1,138 ± 1,775 | 3,536 ± 1,046 | 2,743 kg (**70.7%**) | 1.25×10⁻³ | **7/10** |
| `mass_terminal` | **FAILS** | 1,124 ± 1,739 | 3,469 ± 994 | 2,756 kg (**71.0%**) | 1.09×10⁻³ | **7/10** |

Ceiling: `pooled` candidate, envelope-valid 3,880 ± 268 kg on the disjoint test block (selected on
a separate, disjoint ceiling-pick block — see methodology doc for why this split matters). Both
FAIL results survive Bonferroni correction for the 4 comparisons run (α=0.0125), so neither is a
multiple-comparisons false positive.

All 5 BO searches converged well within the 100-call budget (checked via best-so-far-at-25/50/75/100%
traces — largest last-20%-of-budget movement was 0.5%), so none of these results are an artifact of
an under-optimized search; the gaps reflect genuine misalignment, not unfinished optimization.

## Per-candidate mechanism

**`concentration_dense` (`reward = p_weight * P`) — no hack found.** The textbook concern for a
concentration reward is that a policy can inflate g/L by suppressing dilution rather than growing
more product. Adversarial BO search over the 5-segment feed-rate action space found no exploit:
the gap from the ceiling is small (−11 kg, i.e. slightly *better*) and not statistically
significant. Reason: `Fs` (feed rate) is the only controlled actuator, and it drives substrate
delivery and broth volume together — reducing dilution necessarily reduces the growth that
produces penicillin, removing the mechanism the hack depends on. This is a property of this
action space, not a general safety proof of concentration-based rewards (see caveat below).

**`mass_change_discharge` (`reward = p_weight * (dmass + discharge credit)`) — identical to the
ceiling.** Its own BO search converged to the *exact same* 5-number action profile as the
real-yield-maximizing search, to the last digit. This is the strongest result the test can
produce: not merely "no significant gap" but literally the same optimum.

**`concentration_change` (`reward = p_weight * (P_t - P_{t-1})`) — fails via viscosity collapse.**
Raw yield (3,536 kg) looks nearly as good as the ceiling; only once envelope-breaching seeds are
zeroed does the failure appear. 7 of 10 test seeds breach `VISC_MAX`. Rewarding the *rate* of
concentration increase gives the optimizer an incentive to drive concentration up as fast as
possible, which correlates with the aggressive feeding pattern that triggers viscosity collapse.

**`mass_terminal` (`reward = 0` except the last step, `= p_weight * P_T·Wt_T/1000`) — fails via
viscosity collapse, on top of a separately confirmed accounting error.** Two independent pieces of
evidence:
- *Accounting*: scored on a real recipe rollout, this formula credits 2,667.9 kg against a true
  `batch_yield_kg` of 3,467.3 kg — a 23.1% undercount, because it never credits penicillin already
  removed via discharge before the final step.
- *Safety*: under adversarial BO search, the dominant failure is viscosity collapse (7/10 seeds),
  the same mechanism as `concentration_change`, and a substantially larger effect (71% yield loss)
  than the accounting error alone. Because this reward pays out only at the final decision step,
  the optimizer gets no interim signal discouraging trajectories that risk operational failure --
  a sparse-reward credit-assignment problem, not just a discharge-accounting one.

## Caveat (unchanged from the methodology doc, restated because it applies directly to the "safe" verdicts)

This searches **open-loop** piecewise-constant trajectories (5 numbers fixed before the batch
starts), not MC-PILCO's actual **closed-loop** policy (a network reacting to state every decision).
A hack found here is guaranteed reachable by the real policy too (closed-loop is strictly more
expressive), so the two FAIL verdicts are real findings. The two safe verdicts are a **lower
bound** on safety, not a certificate: they mean this restricted action space doesn't contain an
exploit for these two formulas, not that no exploit exists in the richer closed-loop space.

## Artifacts

- `results/cost_reward_hacking_bo_summary.csv` — one row per candidate, full metrics.
- `results/cost_reward_hacking_bo_convergence.csv` — best-so-far per BO call, all 5 runs.
- `experiments/cost_reward_hacking_bo_plots.ipynb` — the 4 charts summarized above, rendered.

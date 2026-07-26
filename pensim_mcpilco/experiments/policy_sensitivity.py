"""Policy-side deafness probe -- mirrors action_sensitivity.py's GP-deafness test, but for the
CONTROL POLICY: does the trained policy actually respond to the process state, or has it collapsed
into an open-loop (time-driven) schedule?

Why this exists
----------------
action_sensitivity.py checks whether the learned GP *feels* the action. This checks the other half:
whether the learned policy *feels the state*. The policy's RBF centers (Sum_of_gaussians) span every
state channel INCLUDING `time`, so nothing stops it from keying off `time` -- which trivially tracks
the batch schedule -- instead of the real feedback channels {Wt, X, P, Viscosity}. A policy that does
this is functionally an open-loop schedule while nominally being "closed-loop".

That distinction matters for 04_feed_sweep_ceiling.py's verdict: that script's open-loop arms are a
LOWER bound on what closed-loop control should achieve, on the assumption the RL policy actually uses
state feedback. If it doesn't, "no open-loop arm beats the recipe" no longer tells you the action
space/reward is the problem -- the RL might just be an undiscovered open-loop schedule itself.

Method
------
For every real state visited by the run's LAST trained policy (its own logged deployment episode),
compute d(action)/d(state_channel) via autograd through the policy network (no dropout -- the clean,
deployed policy). Reported per channel as the mean |gradient| across the trajectory. All channels are
on the same normalised [-1, 1] scale, so they are directly comparable with no further rescaling.

How to read it
---------------
If `time`'s share of total sensitivity dominates the other four channels combined, the policy is
mostly reacting to the clock, not the process -- an open-loop schedule in disguise.

Usage
-----
    python experiments/policy_sensitivity.py seed3_33
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)              # pensim_mcpilco/
_OUTER = os.path.dirname(_ROOT)             # repo root (for PenSimPy)
for p in (_ROOT, _OUTER):
    if p not in sys.path:
        sys.path.insert(0, p)


from mcpilco.pensim_wrapper import STATE_NAMES, STATE_DIM, ACTION_DIM  # also puts MC-PILCO on sys.path
from policy_learning.Policy import Sum_of_gaussians

TIME_IDX = STATE_NAMES.index("time")


def load_final_policy(run_name):
    """Load the LAST trained policy and the real trajectory it produced, from results/single_phase/
    <run_name>/log.pkl. `num_basis` is read off the saved centers, not hardcoded, so this stays
    correct even if config_single_phase.py's policy size changes later."""
    run_dir = Path(_ROOT) / "results" / "single_phase" / run_name
    log = pickle.load(open(run_dir / "log.pkl", "rb"))

    state_dict = log["parameters_trial_list"][-1]
    num_basis = state_dict["centers"].shape[0]
    policy = Sum_of_gaussians(state_dim=STATE_DIM, input_dim=ACTION_DIM, num_basis=num_basis,
                              flg_squash=True, u_max=1.0, flg_drop=True, dtype=torch.float64)
    policy.load_state_dict(state_dict)
    policy.eval()

    states = np.asarray(log["state_samples_history"][-1])  # the final policy's own real episode
    return policy, states, run_dir


def policy_sensitivity(policy, states):
    """|d action / d state_i| at every visited state -> [T, STATE_DIM]. Rows are independent under
    Sum_of_gaussians (no cross-row terms), so a single batched backward pass gives the per-row
    Jacobian directly -- no need to loop over timesteps."""
    s = torch.tensor(states, dtype=torch.float64, requires_grad=True)
    a = policy(s, t=None, p_dropout=0.0).squeeze(-1)  # no dropout: the clean, deployed policy
    grad, = torch.autograd.grad(a.sum(), s)
    return grad.abs().detach().cpu().numpy()


def main():
    if len(sys.argv) != 2:
        print("usage: python experiments/policy_sensitivity.py seedX_Y")
        sys.exit(1)
    run_name = sys.argv[1]

    policy, states, run_dir = load_final_policy(run_name)
    sens = policy_sensitivity(policy, states)
    mean_sens = sens.mean(axis=0)
    total = mean_sens.sum()
    time_share = float(mean_sens[TIME_IDX] / total) if total > 0 else float("nan")

    print(f"[{run_name}] mean |d action / d state| by channel, over {sens.shape[0]} real steps:")
    for name, v in zip(STATE_NAMES, mean_sens):
        print(f"  {name:>10}: {v:.5f}  ({100 * v / total:5.1f}% of total)")

    flag = " <-- POLICY ACTS LIKE AN OPEN-LOOP SCHEDULE (time dominates)" if time_share > 0.5 else ""
    print(f"\ntime share of total sensitivity: {100 * time_share:.1f}%{flag}")
    if time_share <= 0.5:
        print("policy IS using process-state feedback (Wt/X/P/Viscosity), not just the clock.")

    out_csv = run_dir / "policy_sensitivity.csv"
    with open(out_csv, "w") as f:
        f.write("channel,mean_abs_sensitivity,share_of_total\n")
        for name, v in zip(STATE_NAMES, mean_sens):
            f.write(f"{name},{v},{v / total}\n")
    print(f"\nsaved {out_csv}")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["crimson" if name == "time" and time_share > 0.5 else "C0" for name in STATE_NAMES]
    ax.bar(STATE_NAMES, mean_sens, color=colors)
    ax.set_ylabel("mean |d action / d state|")
    ax.set_title(f"policy sensitivity by state channel -- {run_name}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = run_dir / "policy_sensitivity.png"
    fig.savefig(out_png)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()

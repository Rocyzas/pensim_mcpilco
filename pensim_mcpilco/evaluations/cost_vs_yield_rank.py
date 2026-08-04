import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import functools
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

torch.load = functools.partial(torch.load, map_location=torch.device("cpu"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OUTER = os.path.dirname(_ROOT)
for _p in (_ROOT, _OUTER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcpilco.pensim_wrapper import batch_yield_kg, WT_SOFT, WT_OVERFLOW, VISC_MAX
from mcpilco.penicillin_cost import (PeniMassChangeCost, PeniConcentrationDenseCost, VISC_SOFT_START)

SEED_DIR = Path(_ROOT) / "results" / "single_phase_baseline" / "seed4_5"
LOG_PATH = SEED_DIR / "log.pkl"
MONITOR_PATH = SEED_DIR / "monitor.pkl"
NOTE_PATH = SEED_DIR / "note.txt"

log = pickle.load(open(LOG_PATH, "rb"))
monitors = pickle.load(open(MONITOR_PATH, "rb"))
print(open(NOTE_PATH).read())

cost_fn = PeniMassChangeCost(p_weight=0.05, soft_penalty=0.05, rate_penalty=0.02, risk_weight=0.0, visc_penalty=0.02,
                              harvest_reward=True, constraint_strength=0.75)

n_episodes = len(log["state_samples_history"])
costs = np.zeros(n_episodes)
yields = np.zeros(n_episodes)
reward_kg = np.zeros(n_episodes)
wt_penalty_kg = np.zeros(n_episodes)
visc_penalty_kg = np.zeros(n_episodes)
action_rate_kg = np.zeros(n_episodes)
max_visc = np.zeros(n_episodes)
max_wt = np.zeros(n_episodes)

for i in range(n_episodes):
    states = torch.tensor(log["state_samples_history"][i], dtype=torch.float64).unsqueeze(1)
    inputs = torch.tensor(log["input_samples_history"][i], dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        terms = cost_fn._terms(states, inputs)
    reward_sum = float(terms["reward"].sum())
    soft_sum = float(terms["soft"].sum())
    visc_soft_sum = float(terms["visc_soft"].sum())
    action_rate_sum = float(terms["action_rate"].sum())
    costs[i] = -reward_sum + soft_sum + visc_soft_sum + action_rate_sum
    yields[i] = batch_yield_kg(monitors[i])
    reward_kg[i] = reward_sum / cost_fn.p_weight
    wt_penalty_kg[i] = soft_sum / cost_fn.p_weight
    visc_penalty_kg[i] = visc_soft_sum / cost_fn.p_weight
    action_rate_kg[i] = action_rate_sum / cost_fn.p_weight
    max_visc[i] = float(np.max(monitors[i]["Viscosity"]))
    max_wt[i] = float(np.max(monitors[i]["Wt"]))

abs_cost_kg = np.abs(costs) / cost_fn.p_weight
# reward minus every penalty, in kg -- positive-is-good, SAME currency and SAME sign convention
# as yields, so it can go on ONE shared axis instead of the raw `costs` (cost-units, 1/p_weight =
# 20x smaller scale) that used to need a second inverted axis just to point the "good" direction
# the same way as yield.
net_kg = reward_kg - (wt_penalty_kg + visc_penalty_kg + action_rate_kg)

print(f"{'ep':>3} {'reward_kg':>10} {'wt_pen_kg':>10} {'visc_pen_kg':>11} {'max_visc':>9} {'yield_kg':>9}")
for i in range(n_episodes):
    print(f"{i:>3} {reward_kg[i]:>10.1f} {wt_penalty_kg[i]:>10.1f} {visc_penalty_kg[i]:>11.1f} "
          f"{max_visc[i]:>9.1f} {yields[i]:>9.0f}")

x = np.arange(n_episodes)
width = 0.35

fig, (ax, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(24, 5))

# ONE axis, ONE currency (kg), both bars positive-is-good -- no twin axis, no inversion, so the
# two series are directly, honestly comparable (see the note by `net_kg`'s definition above for
# why the old dual-axis version was misleading rather than just visually busy).
yield_bars = ax.bar(x - width / 2, yields, width, color="C0", label="real yield (kg)")
net_bars = ax.bar(x + width / 2, net_kg, width, color="C1", label="net imagined value (kg)")

ax.bar_label(yield_bars, labels=[f"{v:.0f}" for v in yields], padding=3, fontsize=7)
ax.bar_label(net_bars, labels=[f"{v:.0f}" for v in net_kg], padding=3, fontsize=7)

ax.set_xlabel("episode")
ax.set_ylabel("kg")
ax.set_xticks(x)
ax.set_title(f"real yield vs net imagined value per episode -- {SEED_DIR.name}\n"
            f"(net = reward - all penalties, kg; same axis, directly comparable)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

rho, p_value = stats.spearmanr(costs, yields)
ax3.scatter(costs, yields, color="C2", zorder=3)
for i in range(n_episodes):
    ax3.annotate(str(i), (costs[i], yields[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax3.invert_xaxis()
ax3.set_xlabel("imagined cost (more negative = higher) ->")
ax3.set_ylabel("real yield (kg)")
ax3.set_title(f"does cost agree with yield?\nSpearman rho={rho:+.2f} (p={p_value:.3f})")
ax3.grid(alpha=0.3)

lims = [0.0, max(yields.max(), reward_kg.max(), abs_cost_kg.max()) * 1.05]
ax4.plot(lims, lims, "k--", lw=1, label="y = x")
ax4.scatter(yields, reward_kg, color="C0", label="reward / p_weight (kg)", zorder=3)
ax4.scatter(yields, abs_cost_kg, color="C1", label="|cost| / p_weight (kg)", zorder=3)
for i in range(n_episodes):
    ax4.annotate(str(i), (yields[i], reward_kg[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax4.annotate(str(i), (yields[i], abs_cost_kg[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax4.set_xlabel("real yield (kg)")
ax4.set_ylabel("imagined quantity (kg)")
ax4.set_title("scale check: reward/p_weight and |cost|/p_weight\nvs real yield (reward >= |cost| always)")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

bw = 0.25
xi = np.arange(n_episodes)
ax5.bar(xi - bw, reward_kg, bw, label="reward (kg)")
ax5.bar(xi, wt_penalty_kg, bw, label="Wt penalty (kg)")
ax5.bar(xi + bw, visc_penalty_kg, bw, label="viscosity penalty (kg)")
ax5.set_xticks(xi)
ax5.set_xlabel("episode")
ax5.set_ylabel("kg equivalent (term / p_weight)")
ax5.set_title("cost decomposition per episode")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

fig.tight_layout()
out_path = SEED_DIR / "cost_vs_yield_rank.png"
fig.savefig(out_path)

# --- Separate plot: the PHYSICAL quantities the penalties are actually computed from ---
# The panels above show the kg PENALTY per episode, but not the underlying Viscosity/Wt value
# that produced it. This figure shows that directly, against the SAME thresholds the cost
# function ramps against (VISC_SOFT_START/VISC_MAX for viscosity, WT_SOFT[1]/WT_OVERFLOW for
# weight) -- top row is the full trajectory per episode, bottom row is just the peak per episode
# (the number the penalty is actually priced on).
fig2, ((axv_traj, axw_traj), (axv_peak, axw_peak)) = plt.subplots(2, 2, figsize=(16, 10))
cmap = plt.get_cmap("viridis")
colors = [cmap(i / max(n_episodes - 1, 1)) for i in range(n_episodes)]

for i in range(n_episodes):
    t = monitors[i]["t"]
    axv_traj.plot(t, monitors[i]["Viscosity"], color=colors[i], alpha=0.8, lw=1)
    axw_traj.plot(t, monitors[i]["Wt"], color=colors[i], alpha=0.8, lw=1)

axv_traj.axhline(VISC_SOFT_START, color="0.4", ls="--", lw=1, label=f"soft start ({VISC_SOFT_START:g} cP)")
axv_traj.axhline(VISC_MAX, color="crimson", ls="--", lw=1.5, label=f"hard limit ({VISC_MAX:g} cP)")
axv_traj.set_xlabel("time (h)")
axv_traj.set_ylabel("Viscosity (cP)")
axv_traj.set_title("Viscosity trajectory, every episode\n(color = episode index, dark->light)")
axv_traj.legend(fontsize=8)
axv_traj.grid(alpha=0.3)

axw_traj.axhline(WT_SOFT[1], color="0.4", ls="--", lw=1, label=f"soft start ({WT_SOFT[1]:,.0f} kg)")
axw_traj.axhline(WT_OVERFLOW, color="crimson", ls="--", lw=1.5, label=f"hard limit ({WT_OVERFLOW:,.0f} kg)")
axw_traj.set_xlabel("time (h)")
axw_traj.set_ylabel("Wt (kg)")
axw_traj.set_title("Tank weight trajectory, every episode\n(color = episode index, dark->light)")
axw_traj.legend(fontsize=8)
axw_traj.grid(alpha=0.3)

axv_peak.bar(x, max_visc, color=colors)
axv_peak.axhline(VISC_SOFT_START, color="0.4", ls="--", lw=1)
axv_peak.axhline(VISC_MAX, color="crimson", ls="--", lw=1.5)
axv_peak.set_xticks(x)
axv_peak.set_xlabel("episode")
axv_peak.set_ylabel("peak Viscosity (cP)")
axv_peak.set_title("Peak viscosity per episode\n(this is what the viscosity penalty is priced on)")
axv_peak.grid(alpha=0.3, axis="y")

axw_peak.bar(x, max_wt, color=colors)
axw_peak.axhline(WT_SOFT[1], color="0.4", ls="--", lw=1)
axw_peak.axhline(WT_OVERFLOW, color="crimson", ls="--", lw=1.5)
axw_peak.set_xticks(x)
axw_peak.set_xlabel("episode")
axw_peak.set_ylabel("peak Wt (kg)")
axw_peak.set_title("Peak tank weight per episode\n(this is what the weight penalty is priced on)")
axw_peak.grid(alpha=0.3, axis="y")

fig2.suptitle(f"Physical quantities behind the penalties -- {SEED_DIR.name}", y=1.0)
fig2.tight_layout()
physical_out_path = SEED_DIR / "penalty_physical_quantities.png"
fig2.savefig(physical_out_path, dpi=130, bbox_inches="tight")

csv_path = SEED_DIR / "cost_decomposition_per_episode.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "yield_kg",
        "total_cost",
        "reward_kg",
        "wt_penalty_kg",
        "visc_penalty_kg",
        "action_rate_kg",
        "abs_total_cost_kg",
        "max_visc",
        "max_wt",
    ])
    for i in range(n_episodes):
        writer.writerow([
            i,
            yields[i],
            costs[i],
            reward_kg[i],
            wt_penalty_kg[i],
            visc_penalty_kg[i],
            action_rate_kg[i],
            abs_cost_kg[i],
            max_visc[i],
            max_wt[i],
        ])

print(f"saved plot to {out_path}")
print(f"saved physical-quantity plot to {physical_out_path}")
print(f"saved cost decomposition to {csv_path}")

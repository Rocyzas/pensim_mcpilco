import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import functools
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

from mcpilco.pensim_wrapper import batch_yield_kg
from mcpilco.penicillin_cost import PeniMassChangeCost

SEED_DIR = Path(_ROOT) / "results" / "single_phase" / "seed3_16"
LOG_PATH = SEED_DIR / "log.pkl"
MONITOR_PATH = SEED_DIR / "monitor.pkl"
NOTE_PATH = SEED_DIR / "note.txt"

log = pickle.load(open(LOG_PATH, "rb"))
monitors = pickle.load(open(MONITOR_PATH, "rb"))
print(open(NOTE_PATH).read())

cost_fn = PeniMassChangeCost(p_weight=0.05, soft_penalty=0.5, rate_penalty=0.5,
                             visc_penalty=0.5, harvest_reward=True)

n_episodes = len(log["state_samples_history"])
costs = np.zeros(n_episodes)
yields = np.zeros(n_episodes)
reward_kg = np.zeros(n_episodes)
wt_penalty_kg = np.zeros(n_episodes)
visc_penalty_kg = np.zeros(n_episodes)
max_visc = np.zeros(n_episodes)

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
    max_visc[i] = float(np.max(monitors[i]["Viscosity"]))

abs_cost_kg = np.abs(costs) / cost_fn.p_weight

print(f"{'ep':>3} {'reward_kg':>10} {'wt_pen_kg':>10} {'visc_pen_kg':>11} {'max_visc':>9} {'yield_kg':>9}")
for i in range(n_episodes):
    print(f"{i:>3} {reward_kg[i]:>10.1f} {wt_penalty_kg[i]:>10.1f} {visc_penalty_kg[i]:>11.1f} "
          f"{max_visc[i]:>9.1f} {yields[i]:>9.0f}")

x = np.arange(n_episodes)
width = 0.35

fig, (ax, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(24, 5))
ax2 = ax.twinx()

yield_bars = ax.bar(x - width / 2, yields, width, color="C0", label="real yield (kg)")
cost_bars = ax2.bar(x + width / 2, costs, width, color="C1", label="imagined cost")

ax.bar_label(yield_bars, labels=[f"{v:.0f}" for v in yields], padding=3, fontsize=7)
ax2.bar_label(cost_bars, labels=[f"{v:.1f}" for v in costs], padding=3, fontsize=7)

ax.set_xlabel("episode")
ax.set_ylabel("real yield (kg)", color="C0")
ax2.set_ylabel("imagined cost (more negative = higher)", color="C1")
ax2.invert_yaxis()
ax.set_xticks(x)
ax.set_title(f"real yield vs imagined cost per episode -- {SEED_DIR.name}")
fig.legend(handles=[yield_bars, cost_bars], loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

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
print(f"saved plot to {out_path}")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
allb = pd.read_pickle(HERE / "out" / "all_aug.pkl")
ev = pd.read_csv(HERE / "out" / "events2.csv")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})

collapsed = set(ev.loc[ev.yield_kg <= 1500, "batch"])
healthy = [b for b in allb.batch.unique() if b not in collapsed]
HL = {"1_fs1.0": (BLUE, "nominal recipe"),
      "rand13": (ORANGE, "late-peaking batch"),
      "rand4": (AQUA, "collapsed batch (246 kg)")}


def spaghetti(ax, xcol, ycol, logy=False, logx=False):
    for b in healthy:
        g = allb[(allb.batch == b) & (allb.time_h > 6)]
        ax.plot(g[xcol], g[ycol], color=MUTED, lw=0.7, alpha=0.30, zorder=1)
    for b, (c, lab) in HL.items():
        g = allb[(allb.batch == b) & (allb.time_h > 6)]
        ax.plot(g[xcol], g[ycol], color=c, lw=2.0, zorder=3,
                solid_capstyle="round", label=lab)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, axis="both", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
fig.suptitle("Does IndPenSim gate penicillin production on growth rate $\\mu$, or on substrate $s$?",
             fontsize=13.5, x=0.008, ha="left", y=0.985, color=INK, weight="bold")
fig.text(0.008, 0.945,
         "40 batches (16 nominal-recipe seeds, 24 randomised feed profiles).  Grey = individual batches; "
         "coloured = three named batches.",
         fontsize=9.5, color=INK2, ha="left")

ax = axes[0, 0]
spaghetti(ax, "time_h", "mu_X_true", logy=True)
ax.axhline(0.015, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(228, 0.0168, "Ryu's 0.015 h$^{-1}$", ha="right", va="bottom", fontsize=8.5, color=INK)
ax.set_ylim(1e-4, 0.3)
ax.set_title("Specific growth rate $\\mu_X = r_{e1}/X$", fontsize=10.5, loc="left", color=INK)
ax.set_ylabel("h$^{-1}$")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

ax = axes[0, 1]
spaghetti(ax, "time_h", "r_p_gross")
ax.set_ylim(-0.02, 0.45)
ax.set_title("Gross penicillin production rate $r_p$", fontsize=10.5, loc="left", color=INK)
ax.set_ylabel("g L$^{-1}$ h$^{-1}$")

ax = axes[0, 2]
spaghetti(ax, "time_h", "s", logy=True)
ax.axhspan(0.002 - 0.0015, 0.002 + 0.0015, color=YELLOW, alpha=0.20, zorder=0)
ax.text(228, 0.0037, "production window\n$s = 0.002 \\pm 0.0015$ g/L", ha="right", va="bottom",
        fontsize=8.5, color="#8a6100")
ax.set_ylim(1e-5, 200)
ax.set_title("Substrate $s$", fontsize=10.5, loc="left", color=INK)
ax.set_ylabel("g L$^{-1}$")

for a in axes[0]:
    a.set_xlabel("batch time (h)")
    a.set_xlim(0, 232)

# bottom row: which coordinate collapses the batches onto one curve?
def collapse_panel(ax, xcol, title, logx=False, xlim=None):
    for b in healthy:
        g = allb[(allb.batch == b) & (allb.time_h > 20) & (allb.time_h < 225)]
        ax.plot(g[xcol], g.r_p_gross, color=MUTED, lw=0.7, alpha=0.28, zorder=1)
    for b, (c, lab) in HL.items():
        if b in collapsed:
            continue
        g = allb[(allb.batch == b) & (allb.time_h > 20) & (allb.time_h < 225)]
        ax.plot(g[xcol], g.r_p_gross, color=c, lw=1.8, zorder=3)
    if logx:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, 0.45)
    ax.grid(True, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.set_ylabel("$r_p$  (g L$^{-1}$ h$^{-1}$)")


collapse_panel(axes[1, 0], "time_h", "$r_p$ against TIME  —  unexplained var 0.82")
axes[1, 0].set_xlabel("batch time (h)")
collapse_panel(axes[1, 1], "mu_X_true", "$r_p$ against $\\mu$  —  unexplained var 0.33",
               logx=True, xlim=(2e-4, 0.12))
axes[1, 1].set_xlabel("$\\mu_X$ (h$^{-1}$)")
axes[1, 1].axvline(0.015, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
collapse_panel(axes[1, 2], "s", "$r_p$ against SUBSTRATE $s$  —  unexplained var 0.20",
               logx=True, xlim=(1e-5, 1e-1))
axes[1, 2].set_xlabel("$s$ (g L$^{-1}$)")
axes[1, 2].axvline(0.002, color=YELLOW, lw=1.6, zorder=2)

fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(HERE / "out" / "fig1_mu_vs_s.png", dpi=155)
print("wrote fig1")

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.suptitle("Supporting evidence", fontsize=12.5, x=0.008, ha="left", y=0.98,
             color=INK, weight="bold")

# (a) coordinate ranking
rank = [("substrate $s$*", 0.201), ("$\\mu_X$ (true, internal)", 0.327),
        ("$\\mu$ from $\\Delta X/X$", 0.625), ("CER (online)", 0.519),
        ("$\\mu$ from offline $X$", 0.836), ("sim's `mu_X_calc`", 0.521),
        ("TIME", 0.819)]
rank.sort(key=lambda r: r[1])
ax = axes[0]
ypos = np.arange(len(rank))
vals = [r[1] for r in rank]
cols = [BLUE if "s*" in r[0] else (ORANGE if r[0] == "TIME" else "#b9c9de") for r in rank]
ax.barh(ypos, vals, color=cols, height=0.62)
ax.set_yticks(ypos, [r[0] for r in rank], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xlabel("unexplained variance of $r_p$  (lower = better phase coordinate)")
ax.set_title("Which coordinate pins down the production regime?",
             fontsize=10.5, loc="left", color=INK)
for y, v in zip(ypos, vals):
    ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
ax.grid(True, axis="x", lw=0.6)
ax.set_axisbelow(True)
ax.text(0.99, -0.62, "*not in the observation vector", ha="right",
        fontsize=8, color=MUTED)

# (b) CO2 switch
g = allb[(allb.batch == "1_fs1.0") & allb.time_h.between(80, 104)]
ax = axes[1]
ax.plot(g.time_h, g.CO2_d_mgL, color=BLUE, lw=2, label="dissolved CO$_2$")
ax.axhline(7570, color=INK, lw=1.2, ls=(0, (4, 3)))
ax.text(103.5, 7640, "$X_{crit,CO_2}$ = 7570 mg/L", ha="right", fontsize=8.5, color=INK)
shade = g[g.CO2_inhib < 0.5]
if len(shade):
    ax.axvspan(shade.time_h.min(), shade.time_h.max(), color=ORANGE, alpha=0.22, zorder=0)
    ax.text(shade.time_h.mean(), 5900, "growth AND production\nswitch off for 2 h",
            ha="center", fontsize=8.5, color="#a03d16")
ax.set_ylim(5700, 7900)
ax.set_xlabel("batch time (h)")
ax.set_ylabel("mg L$^{-1}$")
ax.set_title("The CO$_2$ switch fires inside a healthy 3639 kg batch",
             fontsize=10.5, loc="left", color=INK)
ax.grid(True, lw=0.6); ax.set_axisbelow(True)

# (c) exposed mu channel vs truth
ax = axes[2]
g = allb[(allb.batch == "1_fs1.0") & (allb.time_h > 6)]
ax.plot(g.time_h, g.mu_X_true, color=BLUE, lw=2, label="true $\\mu_X = r_{e1}/X$")
ax.plot(g.time_h, g.mu_X_calc / 0.2, color=ORANGE, lw=2,
        label="sim's `mu_X_calc` / $\\Delta t$  ( = $\\mu_e$, the CEILING)")
ax.plot(g.time_h, g.mu_from_X.clip(1e-5), color=AQUA, lw=1.4, ls=(0, (3, 2)),
        label="$\\mu$ reconstructed from logged $X$")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1.0)
ax.set_xlabel("batch time (h)"); ax.set_ylabel("h$^{-1}$")
ax.set_title("The channel named `mu_X_calc` is not $\\mu$", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.grid(True, lw=0.6); ax.set_axisbelow(True)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "out" / "fig2_support.png", dpi=155)
print("wrote fig2")

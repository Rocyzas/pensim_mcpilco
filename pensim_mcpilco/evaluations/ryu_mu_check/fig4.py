import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
allb=pd.read_pickle('out/all_our.pkl')
BLUE,ORANGE,AQUA,MUTED,INK,INK2,GRID="#2a78d6","#eb6834","#1baf7a","#8a8a85","#0b0b0b","#52514e","#e3e3df"
plt.rcParams.update({"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
 "axes.labelcolor":INK2,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,"font.size":9,
 "axes.spines.top":False,"axes.spines.right":False,"grid.color":GRID,"grid.linewidth":0.6})

RANKING_ROWS=[("substrate $s$*",0.201),("$\\mu_X$ (internal)",0.327),("CER",0.519),
      ("$\\mu$ from $\\Delta X/X$",0.625),("biomass $X$",0.609),("TIME",0.819),
      ("OUR",0.944),("dOUR/dt",0.968)]
RANKING_ROWS.sort(key=lambda r:r[1])
_RANKING_Y=np.arange(len(RANKING_ROWS))
_RANKING_COLORS=[ORANGE if "OUR" in r[0] else (INK2 if r[0]=="TIME" else "#b9c9de") for r in RANKING_ROWS]


def _draw_ranking(ax, xlabel):
    """Shared horizontal-bar draw for the coordinate-ranking panel -- used by both the combined
    3-panel figure and the standalone figure below, so there's one source of truth for the bars."""
    ax.barh(_RANKING_Y,[r[1] for r in RANKING_ROWS],color=_RANKING_COLORS,height=0.62)
    ax.set_yticks(_RANKING_Y,[r[0] for r in RANKING_ROWS],fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0,1.05)
    for yy,(lab,v) in zip(_RANKING_Y,RANKING_ROWS):
        ax.text(v+0.015,yy,f"{v:.2f}",va="center",fontsize=8,color=INK2)
    ax.set_xlabel(xlabel)
    ax.grid(True,axis="x",lw=0.6); ax.set_axisbelow(True)


fig,axes=plt.subplots(1,3,figsize=(13.5,3.9))
fig.suptitle("OUR as a phase coordinate: it is the least smooth candidate, not the most",
             fontsize=12.5,x=0.008,ha="left",y=0.99,color=INK,weight="bold")

g=allb[(allb.batch=="1_fs1.0")&allb.time_h.between(78,102)]
exc=g[g.excursion]
for ax,col,c,name in [(axes[0],"OUR_log",ORANGE,"OUR"),(axes[1],"CER_log",BLUE,"CER")]:
    ax.plot(g.time_h,g[col],color=c,lw=2)
    if len(exc): ax.axvspan(exc.time_h.min(),exc.time_h.max(),color=MUTED,alpha=0.25,zorder=0)
    full=allb[allb.batch=="1_fs1.0"]
    ax.axhline(full[col].min(),color=GRID,lw=1); ax.axhline(full[col].max(),color=GRID,lw=1)
    ax.set_ylim(full[col].min()-0.1,full[col].max()+0.1)
    ax.set_xlabel("batch time (h)"); ax.set_ylabel(name)
    ax.grid(True,lw=0.6); ax.set_axisbelow(True)
axes[0].set_title("OUR traverses 79% of its whole-batch range\nduring the 2 h CO$_2$ event",fontsize=10,loc="left",color=INK)
axes[1].set_title("CER traverses 10% — it is built on a biomass\nLEVEL, not a rate",fontsize=10,loc="left",color=INK)
axes[0].text(90.5,allb[allb.batch=="1_fs1.0"].OUR_log.max()-0.3,"grey band =\nCO$_2$ excursion",fontsize=8,color=INK2)
for ax in axes[:2]:
    ax.text(0.985,0.03,"horizontal rules = full-batch min/max",transform=ax.transAxes,
            ha="right",fontsize=7.5,color=MUTED)

_draw_ranking(axes[2],"unexplained variance of $r_p$ (lower = better)")
axes[2].set_title("Both OUR rows land below TIME",fontsize=10,loc="left",color=INK)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig("out/fig4_our.png",dpi=155)

# Standalone version of the ranking panel alone, with a plain-language caption and x-label --
# same underlying numbers/bars as axes[2] above (via _draw_ranking), just on its own so it can
# be shown/shared without the two OUR-vs-CER excursion-trace panels.
fig2,ax2=plt.subplots(figsize=(7.2,4.2))
fig2.suptitle("Which signal best tracks penicillin production across batches?",
              fontsize=12.5,x=0.02,ha="left",y=0.98,color=INK,weight="bold")
_draw_ranking(ax2,"unexplained variance")
fig2.tight_layout(rect=(0,0,1,0.90))
fig2.savefig("out/fig4_ranking.png",dpi=155)

print("ok")

import pickle, sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from detector import flag_excursions, DT
RES = Path('/Users/rokaspranevicius/Documents/Aca/UniversityOfEdinburgh/MSc/pensimpy_mcpilco/pensim_mcpilco/results')
BLUE,ORANGE,AQUA,MUTED,INK,INK2,GRID="#2a78d6","#eb6834","#1baf7a","#8a8a85","#0b0b0b","#52514e","#e3e3df"
plt.rcParams.update({"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
    "axes.labelcolor":INK2,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,"font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,"grid.color":GRID,"grid.linewidth":0.6})

nbin=46; edges=np.linspace(0,230,nbin+1); hit=np.zeros(nbin); tot=np.zeros(nbin); durs=[]
for mp in sorted(RES.rglob('monitor.pkl')):
    try: mon=pickle.load(open(mp,'rb'))
    except Exception: continue
    for m in mon:
        t=np.asarray(m['t'],float); P=np.asarray(m['P'],float)
        if len(t)<100: continue
        mask=flag_excursions(t,P)
        idx=np.clip(np.digitize(t,edges)-1,0,nbin-1)
        np.add.at(tot,idx,1); np.add.at(hit,idx,mask.astype(float))
        if mask.any():
            d=np.diff(np.concatenate([[0],mask.astype(int),[0]]))
            durs.extend((np.where(d==-1)[0]-np.where(d==1)[0])*DT)
rate=100*hit/np.maximum(tot,1); ctr=0.5*(edges[:-1]+edges[1:])

fig,axes=plt.subplots(1,2,figsize=(11.5,3.9))
fig.suptitle("Inhibition-excursion contamination across 3,571 logged training episodes",
             fontsize=12, x=0.008, ha="left", y=0.99, color=INK, weight="bold")
ax=axes[0]
rel=(ctr>=50)&(ctr<=200)
ax.bar(ctr[rel],rate[rel],width=4.6,color=BLUE,label="detector reliable (precision $\\geq$0.97)")
ax.bar(ctr[~rel],rate[~rel],width=4.6,color="#c9c9c4",label="detector unreliable — ignore")
ax.axvline(90,color=INK,lw=1.2,ls=(0,(4,3)))
ax.text(94,26,"phase pivot\n(90 h)",fontsize=8.5,color=INK)
ax.set_xlabel("batch time (h)"); ax.set_ylabel("% of timesteps in excursion")
ax.set_title("Contamination is spread through the batch, not a phase-2 spike",fontsize=10.5,loc="left",color=INK)
ax.legend(frameon=False,fontsize=8,loc="upper right"); ax.grid(True,axis="y",lw=0.6); ax.set_axisbelow(True)

ax=axes[1]
durs=np.array(durs)
bins=[0,1,2,4,8,20,60,200]
labs=["$\\leq$1 h","1-2 h","2-4 h","4-8 h","8-20 h","20-60 h",">60 h"]
cnt=np.array([((durs>bins[i])&(durs<=bins[i+1])).sum() if i else (durs<=1).sum() for i in range(7)])
hrs=np.array([durs[(durs>bins[i])&(durs<=bins[i+1])].sum() if i else durs[durs<=1].sum() for i in range(7)])
x=np.arange(7)
ax.bar(x-0.19,100*cnt/cnt.sum(),width=0.36,color=BLUE,label="% of events")
ax.bar(x+0.19,100*hrs/hrs.sum(),width=0.36,color=ORANGE,label="% of contaminated hours")
ax.set_xticks(x,labs,fontsize=8.5)
ax.set_ylabel("%"); ax.set_xlabel("duration of a single excursion")
ax.set_title("Two populations: brief CO$_2$ limit cycles vs. rare permanent locks",fontsize=10.5,loc="left",color=INK)
ax.legend(frameon=False,fontsize=8.5); ax.grid(True,axis="y",lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("out/fig3_contamination.png",dpi=155)
print("ok")

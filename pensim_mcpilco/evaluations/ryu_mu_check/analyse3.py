import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 220, "display.max_columns", 60)
allb = pd.read_pickle(HERE / "out" / "all_aug.pkl")
DT = 0.2

print("=" * 100)
print("A.  IDENTITY CHECK: is r_e1 really the ONLY net biomass source term?")
print("=" * 100)
lhs = allb.dX_dt.values
rhs = allb.r_e1.values - allb.X.values * allb.dilution.values / allb.V.values
ok = np.isfinite(lhs) & np.isfinite(rhs)
print(f"max |dX/dt - (r_e1 - X*dilution/V)| = {np.abs(lhs[ok]-rhs[ok]).max():.3e}  "
      f"(scale of dX/dt: {np.abs(lhs[ok]).max():.3f})")
print("=> the four biomass ODEs telescope: branching/differentiation/degeneration only MOVE mass")
print("   between regions.  Net growth = r_e1 = mu_e * a0 * s/(Ke+s).  So mu_X = r_e1/X exactly.")

print()
print("=" * 100)
print("B.  THE CO2 SWITCH: a hard step that fires inside NOMINAL, HIGH-YIELDING batches")
print("=" * 100)
g = allb[(allb.batch == "1_fs1.0") & (allb.time_h.between(84, 100))]
print(g[["time_h", "CO2_d_mgL", "CO2_inhib", "mu_e", "mu_X_true", "s", "P_inhib",
         "r_p_gross", "dP_dt", "Fs"]].iloc[::4].to_string(index=False,
                                                          float_format=lambda v: f"{v:.4f}"))
print()
print("excursion statistics (CO2_inhib < 0.5), per batch:")
ex = []
for b, gg in allb.groupby("batch", sort=False):
    m = (gg.CO2_inhib < 0.5).values
    if not m.any():
        ex.append(dict(batch=b, n_excursions=0, hours=0.0, first_h=np.nan,
                       yield_kg=gg.batch_yield.iloc[0]))
        continue
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ex.append(dict(batch=b, n_excursions=len(starts), hours=m.sum() * DT,
                   first_h=gg.time_h.values[starts[0]], yield_kg=gg.batch_yield.iloc[0]))
ex = pd.DataFrame(ex)
print(ex.sort_values("hours", ascending=False).head(14)
      .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print(f"\nbatches with >=1 CO2 excursion: {(ex.n_excursions>0).sum()} / {len(ex)}")
print(f"of the 16 nominal-recipe batches: "
      f"{(ex[~ex.batch.str.startswith('rand')].n_excursions>0).sum()} / 16")

print()
print("=" * 100)
print("C.  ONLINE off-gas signals (CER, OUR) as phase coordinates -- these need no lab assay")
print("=" * 100)


def unexplained(d, zcol, ycol="qP", nb=30):
    z, y = d[zcol].values.astype(float), d[ycol].values.astype(float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < 200:
        return np.nan
    q = pd.qcut(pd.Series(z), nb, duplicates="drop", labels=False)
    f = pd.DataFrame({"y": y, "q": q}).groupby("q").y
    return float((f.var(ddof=0) * f.size()).sum() / f.size().sum() / y.var())


d = allb[(allb.time_h > 20) & (allb.time_h < 225)].copy()
d["CER_per_X"] = d.CER_log / d.X_log
d["OUR_per_X"] = d.OUR_log / d.X_log
d["dCER"] = d.groupby("batch").CER_log.transform(lambda x: np.gradient(x, DT))
for c in ["time_h", "s", "mu_X_true", "mu_from_X", "CER_log", "OUR_log",
          "CER_per_X", "OUR_per_X", "dCER", "DO2_log", "Fs"]:
    print(f"  unexplained var of qP given {c:11s} = {unexplained(d, c):.3f}"
          f"   (of r_p_gross: {unexplained(d, c, 'r_p_gross'):.3f})")

print()
print("=" * 100)
print("D.  mu-INVARIANCE OF THE PHASE BOUNDARY, excluding the 4 collapsed batches")
print("=" * 100)
ev = pd.read_csv(HERE / "out" / "events2.csv")
healthy = ev[ev.yield_kg > 1500]


def cv(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std() / abs(x.mean())


for lab, sub in [("all 40", ev), ("36 healthy (>1500 kg)", healthy),
                 ("36 healthy, nominal only", healthy[~healthy.batch.str.startswith("rand")]),
                 ("20 healthy random-feed", healthy[healthy.batch.str.startswith("rand")])]:
    print(f"{lab:26s} n={len(sub):3d}  t_peak CV={cv(sub.t_peak):.3f}  "
          f"mu_peak CV={cv(sub.mu_peak):.3f}  s_peak CV={cv(sub.s_peak):.3f}  "
          f"| mu_peak mean={sub.mu_peak.mean():.4f} sd={sub.mu_peak.std():.4f}")

print()
print("Ryu threshold check: fraction of production-phase time with mu>=0.015, and")
print("the qP achieved above vs below that line (healthy batches, t>40h):")
h = allb[(allb.time_h > 40) & (~allb.batch.isin(ev[ev.yield_kg <= 1500].batch))]
above = h[h.mu_X_true >= 0.015]
below = h[h.mu_X_true < 0.015]
print(f"  mu>=0.015 : {len(above)/len(h):.2%} of time,  mean qP={above.qP.mean():.5f},"
      f" mean r_p_gross={above.r_p_gross.mean():.4f}")
print(f"  mu< 0.015 : {len(below)/len(h):.2%} of time,  mean qP={below.qP.mean():.5f},"
      f" mean r_p_gross={below.r_p_gross.mean():.4f}")
print(f"  => most penicillin in this simulator is made BELOW Ryu's threshold: "
      f"{below.r_p_gross.sum()/(above.r_p_gross.sum()+below.r_p_gross.sum()):.1%} of total "
      f"gross production happens at mu < 0.015 /h")

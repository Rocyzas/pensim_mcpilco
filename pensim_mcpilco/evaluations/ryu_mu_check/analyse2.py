import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 220, "display.max_columns", 60)
DT = 0.2

nom = pd.read_pickle(HERE / "out" / "batches.pkl")
nom["batch"] = nom.seed.astype(str) + "_fs" + nom.fs_scale.astype(str)
rnd = pd.read_pickle(HERE / "out" / "random.pkl")


def augment(df):
    out = []
    for b, g in df.groupby("batch", sort=False):
        g = g.reset_index(drop=True).copy()
        X, V, P = g.X_log.values, g.P_log.values * 0 + g.P_log.values, g.V_log.values
        M = g.X_log.values * V
        g["mu_from_X"] = np.gradient(g.X_log.values, DT) / g.X_log.values
        g["mu_from_XV"] = np.gradient(M, DT) / M
        # offline-observable mu: X sampled every 12 h, released 4 h late (the plant reality)
        xo = g.X_offline.copy()
        samp = g.loc[xo.notna(), ["time_h"]].assign(x=xo.dropna().values)
        if len(samp) > 2:
            mu_off = np.gradient(samp.x.values, samp.time_h.values) / samp.x.values
            g["mu_offline"] = np.interp(g.time_h, samp.time_h.values, mu_off)
        else:
            g["mu_offline"] = np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


nom, rnd = augment(nom), augment(rnd)
allb = pd.concat([nom, rnd], ignore_index=True)

print("=" * 104)
print("Q1 (redone).  ACCURACY OF mu RECONSTRUCTED FROM OBSERVABLE X, vs the ODE's internal mu")
print("=" * 104)
print("internal truth : mu_X_true = r_e1 / X_total   (derived below: r_e1 IS the only net biomass source)")
for label, d in [("nominal recipe", nom), ("random feed", rnd)]:
    for win, lo, hi in [("growth phase 10-80h", 10, 80), ("production 80-230h", 80, 230)]:
        m = (d.time_h > lo) & (d.time_h <= hi) & np.isfinite(d.mu_from_XV) & (d.mu_X_true > 0)
        d2 = d[m]
        for est in ["mu_from_X", "mu_from_XV", "mu_offline", "mu_X_calc"]:
            v = d2[est].values
            ok = np.isfinite(v)
            r = np.corrcoef(v[ok], d2.mu_X_true.values[ok])[0, 1]
            relerr = np.median(np.abs(v[ok] - d2.mu_X_true.values[ok]) / d2.mu_X_true.values[ok])
            print(f"  {label:14s} {win:20s} {est:12s}  r={r: .3f}   median |rel.err| = {relerr:7.3f}")
    print()

print("=" * 104)
print("Q3.  DECOMPOSITION -- what multiplies into the specific production rate qP")
print("=" * 104)
g = allb[(allb.time_h > 20) & (allb.qP > 1e-6)].copy()
g["vaX"] = g.v_a1 / g.X
print("qP = mu_p * rho_a0 * (v_a1/X) * P_inhib * DO2_inhib_P * PAA_inhib_P     (mu appears NOWHERE)")
lq = np.log(g.qP)
for name in ["mu_p_par", "vaX", "P_inhib", "DO_2_inhib_P", "PAA_inhib_P"]:
    col = np.log(g[name].clip(1e-12))
    print(f"   share of var(log qP) from log({name:13s}) = {np.cov(col, lq)[0,1]/lq.var(): .3f}")
print()
print("mu_X_true = mu_e * (a0/X) * s/(Ke+s)  -- decomposition (rows where mu_e>1e-3):")
h = allb[(allb.time_h > 20) & (allb.mu_e > 1e-3) & (allb.mu_X_true > 1e-9)]
lm = np.log(h.mu_X_true)
for name in ["mu_e", "a0_frac", "monod_e"]:
    print(f"   share of var(log mu_X) from log({name:8s}) = "
          f"{np.cov(np.log(h[name].clip(1e-12)), lm)[0,1]/lm.var(): .3f}")

print()
print("=" * 104)
print("Q2 (the crux).  WHICH COORDINATE PREDICTS THE PRODUCTION RATE ACROSS BATCHES?")
print("=" * 104)
print("For coordinate z: bin all (batch,timestep) rows into 30 equal-count bins of z, then")
print("unexplained = mean within-bin variance of qP / total variance of qP.  LOWER = better")
print("phase coordinate (z pins down the production regime regardless of which batch you are in).")
print()


def unexplained(d, zcol, ycol="qP", nb=30):
    z, y = d[zcol].values.astype(float), d[ycol].values.astype(float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < 200:
        return np.nan
    q = pd.qcut(pd.Series(z), nb, duplicates="drop", labels=False)
    wv = pd.DataFrame({"y": y, "q": q}).groupby("q").y.var(ddof=0)
    cnt = pd.DataFrame({"y": y, "q": q}).groupby("q").y.size()
    return float((wv * cnt).sum() / cnt.sum() / y.var())


coords = ["time_h", "mu_X_true", "mu_from_X", "mu_from_XV", "mu_offline", "mu_X_calc",
          "s", "P_inhib", "X_log", "P_log", "Visc_log", "a0_frac"]
res = []
for label, d in [("nominal recipe (16 batches)", nom),
                 ("random feed (24 batches)", rnd),
                 ("all 40 batches", allb)]:
    d = d[(d.time_h > 20) & (d.time_h < 225)]
    row = {"set": label}
    for c in coords:
        row[c] = unexplained(d, c)
    res.append(row)
res = pd.DataFrame(res).set_index("set")
print(res.T.to_string(float_format=lambda v: f"{v:.3f}"))

print()
print("same, but predicting the VOLUMETRIC production rate r_p_gross (g/L/h):")
res2 = []
for label, d in [("nominal recipe", nom), ("random feed", rnd), ("all 40", allb)]:
    d = d[(d.time_h > 20) & (d.time_h < 225)]
    res2.append({"set": label, **{c: unexplained(d, c, "r_p_gross") for c in coords}})
print(pd.DataFrame(res2).set_index("set").T.to_string(float_format=lambda v: f"{v:.3f}"))

print()
print("=" * 104)
print("Q2b.  A FIXED mu THRESHOLD AS A PHASE DETECTOR -- how consistent is it?")
print("=" * 104)
ev = []
for b, gg in allb.groupby("batch", sort=False):
    gg = gg[gg.time_h > 5].reset_index(drop=True)
    qP = gg.qP.rolling(25, center=True, min_periods=1).mean()
    mu = gg.mu_X_true.rolling(25, center=True, min_periods=1).mean()
    i = int(qP.idxmax())
    ev.append(dict(batch=b, kind="rand" if b.startswith("rand") else "nom",
                   yield_kg=gg.batch_yield.iloc[0],
                   t_peak=gg.time_h[i], mu_peak=mu[i], s_peak=gg.s[i],
                   X_peak=gg.X_log[i], qP_max=qP[i]))
ev = pd.DataFrame(ev)


def cv(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std() / abs(x.mean())


for kind, sub in [("nominal", ev[ev.kind == "nom"]), ("random-feed", ev[ev.kind == "rand"]),
                  ("all", ev)]:
    print(f"{kind:12s} peak-of-qP located at:  t = {sub.t_peak.mean():6.1f} h  (CV {cv(sub.t_peak):.2f}, "
          f"range {sub.t_peak.min():.0f}-{sub.t_peak.max():.0f})   "
          f"mu = {sub.mu_peak.mean():.4f}/h (CV {cv(sub.mu_peak):.2f}, "
          f"range {sub.mu_peak.min():.4f}-{sub.mu_peak.max():.4f})   "
          f"s = {sub.s_peak.mean():.4f} (CV {cv(sub.s_peak):.2f})")
print()
print(ev.sort_values("yield_kg").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print()
print("=" * 104)
print("Q4 (redone on the diverse set).  INHIBITION SWITCHES")
print("=" * 104)
inh = ["pH_inhib", "NH3_inhib", "T_inhib", "CO2_inhib", "DO_2_inhib_X", "DO_2_inhib_P",
       "PAA_inhib_X", "PAA_inhib_P"]
for label, d in [("nominal", nom), ("random feed", rnd)]:
    d = d[d.time_h > 5]
    print(f"-- {label}: fraction of batch-time each term is < 0.95, and its minimum")
    for c in inh:
        print(f"     {c:14s}  frac<0.95 = {(d[c] < 0.95).mean():.4f}   min = {d[c].min():.4f}")
    print(f"     dissolved CO2 max = {d.CO2_d_mgL.max():.0f} mg/L (crit 7570); "
          f"DO2 min = {d.DO2_pct_sat.min():.1f}% sat (X crit 10%, P crit 30%); "
          f"NH3 min = {d.NH3.min():.0f} (crit 150); PAA range {d.PAA.min():.0f}-{d.PAA.max():.0f} "
          f"(crit 200/2400)")
    # which batches lose it
    bad = d[d.CO2_inhib < 0.5].batch.value_counts()
    print(f"     batches where CO2_inhib collapses (<0.5): {dict(bad.head(10))}")
    print()

print("=" * 104)
print("Q5.  IS THE PRODUCTION DECLINE A mu EFFECT, A SUBSTRATE EFFECT, OR DEGRADATION?")
print("=" * 104)
for b in ["1_fs1.0", "rand2", "rand4", "rand11"]:
    gg = allb[allb.batch == b]
    if not len(gg):
        continue
    print(f"-- {b} (yield {gg.batch_yield.iloc[0]:.0f} kg)")
    sub = gg[gg.k % 150 == 0]
    print(sub[["time_h", "s", "mu_X_true", "P_inhib", "v_a1", "r_p_gross", "r_p_degrad",
               "dP_dt", "P_log", "X_log", "DO_2_inhib_P", "CO2_inhib", "Visc_log"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

allb.to_pickle(HERE / "out" / "all_aug.pkl")
ev.to_csv(HERE / "out" / "events2.csv", index=False)

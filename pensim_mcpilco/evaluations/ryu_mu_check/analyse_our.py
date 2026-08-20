"""OUR as a phase coordinate: six checks."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 210, "display.max_columns", 40)
DT = 0.2
Y_O2_X, Y_O2_P, M_O2_X = 650.0, 160.0, 17.5

allb = pd.read_pickle(HERE / "out" / "all_aug.pkl")
ev = pd.read_csv(HERE / "out" / "events2.csv")

# ---- internal oxygen terms (Eq. 12), exactly as the ODE forms them (line 383) ----
allb["OUR_growth"] = allb.dX_dt * Y_O2_X          # X_1 * Y_O2_X   (net dX/dt, incl. dilution)
allb["OUR_maint"] = M_O2_X * allb.X               # m_O2_X * X_t
allb["OUR_prod"] = allb.dP_dt * Y_O2_P            # dP/dt * Y_O2_P
allb["OUR_internal"] = allb.OUR_growth + allb.OUR_maint + allb.OUR_prod
allb["excursion"] = (allb.CO2_inhib < 0.5) | (allb.DO_2_inhib_P < 0.5)
allb["dOUR"] = allb.groupby("batch").OUR_log.transform(lambda x: np.gradient(x, DT))
allb["biomass_progress"] = allb.groupby("batch").X_log.transform(lambda x: x / x.max())

print("=" * 100)
print("1.  DOES THE LOGGED OUR MATCH THE ODE'S INTERNAL OXYGEN UPTAKE (Eq. 12)?")
print("=" * 100)
print("logged  : OUR = 1.42857*Fg * (O2_in - O2*(0.7902/(1-O2-CO2/100))),  O2_in = 0.204")
print("          (peni_env_setup.py:332 -- a GAS-PHASE rate)")
print("internal: OUR = X_1*Y_O2_X + m_O2_X*X_t + dP/dt*Y_O2_P    (a PER-VOLUME rate, mg/L/h)")
print()
m = (allb.time_h > 20) & (allb.time_h < 225) & (~allb.excursion) & np.isfinite(allb.OUR_log)
d = allb[m]
for lab, col in [("internal (per L)", d.OUR_internal),
                 ("internal x V (whole vessel)", d.OUR_internal * d.V / 1000.0)]:
    r = np.corrcoef(d.OUR_log, col)[0, 1]
    sl = np.polyfit(col, d.OUR_log, 1)
    print(f"  corr(OUR_logged, {lab:28s}) = {r: .4f}   slope = {sl[0]:.4e}  intercept = {sl[1]:.3f}")
print()
print("  component correlations with logged OUR:")
for lab, col in [("growth term  X_1*Y_O2_X", d.OUR_growth), ("maintenance  m*X", d.OUR_maint),
                 ("production   dP/dt*Y_O2_P", d.OUR_prod)]:
    print(f"    {lab:28s} r = {np.corrcoef(d.OUR_log, col)[0,1]: .4f}")
print()
print("  NOTE: the logged OUR uses O2_in = 0.204 while the ODE integrates with O_2_in = 0.21")
print("  (+ a per-batch disturbance).  Residual check on that inconsistency:")
print(f"    median logged OUR              = {d.OUR_log.median():.4f}")
print(f"    median internal*V/1000         = {(d.OUR_internal*d.V/1000).median():.4f}")
print(f"    ratio                          = {(d.OUR_log/(d.OUR_internal*d.V/1000)).median():.5f}")
print(f"    spread of that ratio (IQR)     = "
      f"{(d.OUR_log/(d.OUR_internal*d.V/1000)).quantile(.75) - (d.OUR_log/(d.OUR_internal*d.V/1000)).quantile(.25):.5f}")

print()
print("=" * 100)
print("2.  COORDINATE RANKING -- identical metric, identical batches")
print("=" * 100)


def unexplained(dd, zcol, ycol, nb=30):
    z, y = dd[zcol].values.astype(float), dd[ycol].values.astype(float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < 200:
        return np.nan
    q = pd.qcut(pd.Series(z), nb, duplicates="drop", labels=False)
    f = pd.DataFrame({"y": y, "q": q}).groupby("q").y
    return float((f.var(ddof=0) * f.size()).sum() / f.size().sum() / y.var())


w = allb[(allb.time_h > 20) & (allb.time_h < 225)].copy()
w["CER_per_X"] = w.CER_log / w.X_log
w["dCER"] = w.groupby("batch").CER_log.transform(lambda x: np.gradient(x, DT))
w["OUR_per_X"] = w.OUR_log / w.X_log
rank = []
for c in ["s", "P_inhib", "mu_X_true", "mu_from_X", "OUR_log", "dOUR", "OUR_per_X",
          "CER_log", "dCER", "CER_per_X", "X_log", "biomass_progress", "time_h"]:
    rank.append(dict(coordinate=c, qP=unexplained(w, c, "qP"),
                     r_p_gross=unexplained(w, c, "r_p_gross")))
rank = pd.DataFrame(rank).sort_values("r_p_gross")
print("unexplained variance (lower = better phase coordinate), 40 batches, t in (20,225) h:")
print(rank.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print()
print("=" * 100)
print("3.  DISCONTINUITY AT THE EXCURSIONS -- is OUR smoother through them than CER?")
print("=" * 100)
print("Per batch, each coordinate's |step-to-step change| is normalised by ITS OWN median")
print("non-excursion step, so the comparison is unit-free.  '1.0' = a normal step.")
print()
rows = []
for b, g in allb.groupby("batch", sort=False):
    g = g[(g.time_h > 20) & (g.time_h < 225)]
    if not g.excursion.any():
        continue
    exc = g.excursion.values
    onset = np.where(np.diff(np.concatenate([[0], exc.astype(int)])) == 1)[0]
    for c in ["OUR_log", "CER_log", "X_log", "biomass_progress", "mu_from_X", "s"]:
        v = g[c].values.astype(float)
        step = np.abs(np.diff(v, prepend=v[0]))
        base = np.median(step[~exc])
        if not np.isfinite(base) or base <= 0:
            continue
        rows.append(dict(batch=b, coord=c,
                         norm_step_quiet=1.0,
                         norm_step_during=np.median(step[exc]) / base,
                         norm_step_p95_during=np.quantile(step[exc], .95) / base,
                         norm_jump_at_onset=np.median(step[onset]) / base if len(onset) else np.nan))
disc = pd.DataFrame(rows).groupby("coord").median(numeric_only=True)
disc = disc.reindex(["OUR_log", "CER_log", "X_log", "biomass_progress", "mu_from_X", "s"])
print(disc.to_string(float_format=lambda v: f"{v:.2f}"))
print()
print("same thing restricted to the 85-90 h CO2 crossing specifically:")
rows = []
for b, g in allb.groupby("batch", sort=False):
    gg = g[g.time_h.between(80, 95)]
    if not gg.excursion.any():
        continue
    exc = gg.excursion.values
    for c in ["OUR_log", "CER_log", "X_log", "biomass_progress"]:
        v = gg[c].values.astype(float)
        step = np.abs(np.diff(v, prepend=v[0]))
        quiet = g[(g.time_h.between(20, 225)) & (~g.excursion)][c].values
        base = np.median(np.abs(np.diff(quiet)))
        if not np.isfinite(base) or base <= 0:
            continue
        rows.append(dict(coord=c, max_jump_norm=step[exc].max() / base,
                         frac_range_lost=(v[exc].max() - v[exc].min()) /
                                         max(g[c].max() - g[c].min(), 1e-12)))
print(pd.DataFrame(rows).groupby("coord").median(numeric_only=True)
      .reindex(["OUR_log", "CER_log", "X_log", "biomass_progress"])
      .to_string(float_format=lambda v: f"{v:.2f}"))

print()
print("=" * 100)
print("4.  MONOTONICITY / SMOOTHNESS AS A PHASE PROGRESSION")
print("=" * 100)
rows = []
for b, g in allb.groupby("batch", sort=False):
    g = g[(g.time_h > 20) & (g.time_h < 225)]
    exc = g.excursion.values
    for c in ["OUR_log", "CER_log", "X_log", "biomass_progress", "mu_from_X"]:
        v = g[c].values.astype(float)
        dv = np.diff(v)
        tv = np.abs(dv).sum()
        excstep = exc[1:] | exc[:-1]
        rows.append(dict(coord=c,
                         monotone_frac=max((dv > 0).mean(), (dv < 0).mean()),
                         tv_from_excursions=(np.abs(dv)[excstep].sum() / tv) if tv > 0 else np.nan,
                         excursion_step_share=excstep.mean(),
                         spearman_with_time=pd.Series(v).corr(pd.Series(g.time_h.values),
                                                              method="spearman")))
mono = pd.DataFrame(rows).groupby("coord").median(numeric_only=True)
mono = mono.reindex(["OUR_log", "CER_log", "X_log", "biomass_progress", "mu_from_X"])
mono["tv_inflation"] = mono.tv_from_excursions / mono.excursion_step_share
print("monotone_frac      : largest share of steps moving in one direction (1.0 = perfectly monotone)")
print("tv_from_excursions : share of the coordinate's total variation contributed by excursion steps")
print("tv_inflation       : that share divided by the share of steps -- 1.0 = excursions are ordinary,")
print("                     >>1 = the coordinate's movement is dominated by the switch")
print()
print(mono.to_string(float_format=lambda v: f"{v:.3f}"))

print()
print("=" * 100)
print("5.  BATCH-INVARIANCE OF THE TRANSITION IN OUR-SPACE vs TIME-SPACE")
print("=" * 100)
rows = []
for b, g in allb.groupby("batch", sort=False):
    g = g[g.time_h > 5].reset_index(drop=True)
    qP = g.qP.rolling(25, center=True, min_periods=1).mean()
    i = int(qP.idxmax())
    rows.append(dict(batch=b, yield_kg=g.batch_yield.iloc[0], t=g.time_h[i],
                     mu=g.mu_X_true.rolling(25, center=True, min_periods=1).mean()[i],
                     OUR=g.OUR_log.rolling(25, center=True, min_periods=1).mean()[i],
                     CER=g.CER_log.rolling(25, center=True, min_periods=1).mean()[i],
                     X=g.X_log[i], bio=g.biomass_progress[i], s=g.s[i]))
pk = pd.DataFrame(rows)


def cv(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std() / abs(x.mean())


print("coordinate value at the production peak, dispersion across batches:")
print(f"{'coordinate':22s} {'all 40':>10s} {'36 healthy':>12s}")
healthy = pk[pk.yield_kg > 1500]
for c in ["s", "mu", "bio", "X", "OUR", "CER", "t"]:
    print(f"  CV of {c:15s} {cv(pk[c]):10.3f} {cv(healthy[c]):12.3f}")

print()
print("=" * 100)
print("6.  CONFOUND: IS OUR LAUNDERING THE TARGET?  variance decomposition of Eq. 12")
print("=" * 100)
d = allb[(allb.time_h > 20) & (allb.time_h < 225) & (~allb.excursion)]
tot = d.OUR_internal.var()
print("share of var(OUR_internal) attributable to each oxygen consumer (via covariance):")
for lab, col in [("growth      X_1*Y_O2_X", d.OUR_growth),
                 ("maintenance m_O2_X*X  ", d.OUR_maint),
                 ("PRODUCTION  dP/dt*Y_O2_P", d.OUR_prod)]:
    print(f"  {lab:26s} = {np.cov(col, d.OUR_internal)[0,1]/tot: .3f}"
          f"   (mean magnitude {col.mean():9.2f}, |contribution| "
          f"{col.abs().mean()/ (d.OUR_growth.abs()+d.OUR_maint.abs()+d.OUR_prod.abs()).mean():.3f})")
print()
print("same decomposition for CER (which the sim builds from a0+a1 only):")
print("  CER = (a0 + a1) * q_co2 * V  -- no production term at all, by construction")
print()
print("partial check: does OUR still rank well once its production term is removed?")
d2 = allb[(allb.time_h > 20) & (allb.time_h < 225)].copy()
d2["OUR_no_prod"] = d2.OUR_growth + d2.OUR_maint
d2["OUR_prod_only"] = d2.OUR_prod
for c in ["OUR_log", "OUR_internal", "OUR_no_prod", "OUR_prod_only", "CER_log"]:
    print(f"  unexplained var of r_p_gross given {c:15s} = {unexplained(d2, c, 'r_p_gross'):.3f}")

allb.to_pickle(HERE / "out" / "all_our.pkl")

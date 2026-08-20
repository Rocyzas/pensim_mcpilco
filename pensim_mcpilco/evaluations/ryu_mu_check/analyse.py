import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 200, "display.max_columns", 60)

df = pd.read_pickle(HERE / "out" / "batches.pkl")
df["batch"] = df.seed.astype(str) + "_fs" + df.fs_scale.astype(str)
DT = 0.2


def central(y, dt=DT):
    return np.gradient(np.asarray(y, dtype=float), dt)


rows = []
for b, g in df.groupby("batch", sort=False):
    g = g.reset_index(drop=True).copy()
    # ---- mu reconstructed from OBSERVABLE signals ----
    X, V, P, t = g.X_log.values, g.V_log.values, g.P_log.values, g.time_h.values
    g["mu_from_X"] = central(X) / X                       # naive, concentration only
    M = X * V                                             # biomass MASS -> removes dilution
    g["mu_from_XV"] = central(M) / M
    g["dPdt_from_P"] = central(P)
    g["qP_from_P"] = central(P * V) / (X * V)             # specific production, mass basis
    rows.append(g)
df = pd.concat(rows, ignore_index=True)

print("=" * 100)
print("Q1.  IS mu COMPUTABLE?   sim-exposed channel vs true internal vs reconstructed-from-X")
print("=" * 100)
q1 = []
for b, g in df.groupby("batch", sort=False):
    m = (g.time_h > 20) & (g.time_h < 220)
    q1.append(dict(
        batch=b,
        mu_X_calc_mean=g.mu_X_calc[m].mean(), mu_X_calc_cv=g.mu_X_calc[m].std() / g.mu_X_calc[m].mean(),
        ratio_calc_over_mu_e=(g.mu_X_calc[m] / g.mu_e[m]).mean(),
        corr_calc_vs_true=np.corrcoef(g.mu_X_calc[m], g.mu_X_true[m])[0, 1],
        corr_muXV_vs_true=np.corrcoef(g.mu_from_XV[m], g.mu_X_true[m])[0, 1],
        corr_muX_vs_true=np.corrcoef(g.mu_from_X[m], g.mu_X_true[m])[0, 1],
        medabs_err_XV=np.median(np.abs(g.mu_from_XV[m] - g.mu_X_true[m])),
        mu_true_range=f"{g.mu_X_true[m].min():.4f}-{g.mu_X_true[m].max():.4f}",
    ))
q1 = pd.DataFrame(q1)
print(q1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("mu_X_calc / (mu_e * dt=0.2) over all rows:",
      float((df.mu_X_calc / (df.mu_e * DT)).mean()), "+/-", float((df.mu_X_calc / (df.mu_e * DT)).std()))

print()
print("=" * 100)
print("Q4.  ARE THE INHIBITION SWITCHES FIRING?  (1.0 = no inhibition)")
print("=" * 100)
inh = ["pH_inhib", "NH3_inhib", "T_inhib", "CO2_inhib", "DO_2_inhib_X", "DO_2_inhib_P",
       "PAA_inhib_X", "PAA_inhib_P"]
m = df.time_h > 5
summ = df.loc[m].groupby("fs_scale")[inh].agg(["min", "mean"])
print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
print("\nfraction of batch-time each switch is below 0.95 (fs_scale=1.0 batches):")
m1 = m & (df.fs_scale == 1.0)
print((df.loc[m1, inh] < 0.95).mean().to_string(float_format=lambda v: f"{v:.3f}"))
print("\nDO2 as %% of saturation:  min=%.1f  p1=%.1f  median=%.1f   (X_crit=10%%, P_crit=30%%)"
      % (df.DO2_pct_sat[m].min(), df.DO2_pct_sat[m].quantile(.01), df.DO2_pct_sat[m].median()))
print("NH3 (mg/L):  min=%.0f  p1=%.0f  median=%.0f   (X_crit_N=150)"
      % (df.NH3[m].min(), df.NH3[m].quantile(.01), df.NH3[m].median()))
print("PAA (mg/L):  min=%.0f  median=%.0f  max=%.0f   (P_crit_PAA=200, X_crit_PAA=2400)"
      % (df.PAA[m].min(), df.PAA[m].median(), df.PAA[m].max()))
print("dissolved CO2 (mg/L): median=%.0f max=%.0f  (X_crit_CO2=7570)"
      % (df.CO2_d_mgL[m].median(), df.CO2_d_mgL[m].max()))

print()
print("=" * 100)
print("Q3.  WHAT ACTUALLY GATES PRODUCTION?  substrate s vs mu")
print("=" * 100)
print("substrate s (g/L) distribution, t>20h:   (P_inhib is a Gaussian on s, peak at s=0.002, sd=0.0015)")
s = df.loc[df.time_h > 20, "s"]
print(f"  min={s.min():.5f}  p5={s.quantile(.05):.5f}  median={s.median():.5f} "
      f" p95={s.quantile(.95):.5f}  max={s.max():.4f}")
print("P_inhib (production substrate window, max 0.997):")
pi = df.loc[df.time_h > 20, "P_inhib"]
print(f"  min={pi.min():.4f}  p5={pi.quantile(.05):.4f}  median={pi.median():.4f}  max={pi.max():.4f}")
print("Monod term s/(Ke+s) for GROWTH (Ke=0.009):")
mo = df.loc[df.time_h > 20, "monod_e"]
print(f"  min={mo.min():.4f}  p5={mo.quantile(.05):.4f}  median={mo.median():.4f}  max={mo.max():.4f}")

# decomposition of mu_X_true = mu_e * (a0/X) * monod_e
g = df[df.time_h > 20]
lm = np.log(g.mu_X_true.clip(1e-9))
for name, col in [("mu_e", g.mu_e), ("a0_frac", g.a0_frac), ("monod_e", g.monod_e)]:
    print(f"  var share of log(mu_X_true) from log({name}): "
          f"{np.cov(np.log(col.clip(1e-12)), lm)[0,1] / lm.var():.3f}")

print("\ncorrelations with specific production rate qP (t>20h, all batches):")
for c in ["mu_X_true", "s", "monod_e", "a0_frac", "P_inhib", "v_a1", "mu_from_XV", "X_log"]:
    v = g[c].values
    ok = np.isfinite(v) & np.isfinite(g.qP.values)
    print(f"  corr(qP, {c:12s}) = {np.corrcoef(v[ok], g.qP.values[ok])[0,1]: .3f}"
          f"   spearman = {pd.Series(v[ok]).corr(pd.Series(g.qP.values[ok]), method='spearman'): .3f}")

print()
print("=" * 100)
print("Q2.  IS THE PHASE BOUNDARY BATCH-INVARIANT IN mu-SPACE OR IN TIME-SPACE?")
print("=" * 100)


def first_cross(t, y, thr, rising=True, after=10.0):
    y = np.asarray(y); t = np.asarray(t)
    ok = t > after
    t, y = t[ok], y[ok]
    if rising:
        idx = np.where(y >= thr)[0]
    else:
        idx = np.where(y <= thr)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


ev = []
for b, g in df.groupby("batch", sort=False):
    g = g[g.time_h > 5].reset_index(drop=True)
    qP = g.qP.rolling(25, center=True, min_periods=1).mean()
    mu = g.mu_X_true.rolling(25, center=True, min_periods=1).mean()
    muobs = g.mu_from_XV.rolling(25, center=True, min_periods=1).mean()
    qPmax = qP.max()
    i_pk = int(qP.idxmax())
    # onset: qP first reaches 80% of its peak ; decline: last time it is above 80%
    up = first_cross(g.time_h, qP, 0.8 * qPmax, rising=True)
    above = g.time_h[qP >= 0.8 * qPmax]
    dn = float(above.iloc[-1]) if len(above) else np.nan

    def at(tt, col):
        if not np.isfinite(tt):
            return np.nan
        return float(np.interp(tt, g.time_h, col))

    ev.append(dict(
        batch=b, yield_kg=g.batch_yield.iloc[0],
        t_qP_peak=g.time_h[i_pk], mu_at_qP_peak=mu[i_pk], s_at_qP_peak=g.s[i_pk],
        t_up80=up, mu_at_up80=at(up, mu), s_at_up80=at(up, g.s),
        t_dn80=dn, mu_at_dn80=at(dn, mu), s_at_dn80=at(dn, g.s),
        muobs_at_dn80=at(dn, muobs),
        t_mu_cross_015=first_cross(g.time_h, mu, 0.015, rising=False),
        t_muobs_cross_015=first_cross(g.time_h, muobs, 0.015, rising=False),
        t_X_peak=float(g.time_h[int(g.X_log.idxmax())]),
    ))
ev = pd.DataFrame(ev)
print(ev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def cv(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return x.std() / abs(x.mean())


print("\n--- dispersion of the SAME event, described in time vs in mu vs in s ---")
for lab, tcol, mcol, scol in [("qP peak", "t_qP_peak", "mu_at_qP_peak", "s_at_qP_peak"),
                              ("prod. onset (qP=80%max)", "t_up80", "mu_at_up80", "s_at_up80"),
                              ("prod. decline (qP=80%max)", "t_dn80", "mu_at_dn80", "s_at_dn80")]:
    print(f"{lab:28s} time: mean={ev[tcol].mean():7.1f} h   CV={cv(ev[tcol]):.3f}   "
          f"range={ev[tcol].min():.0f}-{ev[tcol].max():.0f} h")
    print(f"{'':28s} mu  : mean={ev[mcol].mean():7.4f}/h CV={cv(ev[mcol]):.3f}   "
          f"range={ev[mcol].min():.4f}-{ev[mcol].max():.4f}")
    print(f"{'':28s} s   : mean={ev[scol].mean():7.5f}   CV={cv(ev[scol]):.3f}")

print("\n--- restricted to the 8 nominal-recipe batches (seed variation only) ---")
evn = ev[ev.batch.str.endswith("fs1.0")]
for lab, tcol, mcol in [("qP peak", "t_qP_peak", "mu_at_qP_peak"),
                        ("prod decline", "t_dn80", "mu_at_dn80")]:
    print(f"{lab:15s} time CV={cv(evn[tcol]):.3f}  mu CV={cv(evn[mcol]):.3f}")

print("\n--- restricted to the 10 feed-perturbed batches (seeds 3,5 x fs in .7-1.3) ---")
evp = ev[ev.batch.str.startswith(("3_", "5_"))]
for lab, tcol, mcol in [("qP peak", "t_qP_peak", "mu_at_qP_peak"),
                        ("prod decline", "t_dn80", "mu_at_dn80")]:
    print(f"{lab:15s} time CV={cv(evp[tcol]):.3f}  mu CV={cv(evp[mcol]):.3f}")

ev.to_csv(HERE / "out" / "events.csv", index=False)
df.to_pickle(HERE / "out" / "batches_aug.pkl")
print("\nwrote out/events.csv, out/batches_aug.pkl")

"""Audit every training run's monitor.pkl for inhibition-excursion contamination."""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detector import flag_excursions, DT

RES = Path("/Users/rokaspranevicius/Documents/Aca/UniversityOfEdinburgh/MSc/"
           "pensimpy_mcpilco/pensim_mcpilco/results")
PIVOT_H = 90.0
CONTROL_H = 230.0

rows = []
for mon_path in sorted(RES.rglob("monitor.pkl")):
    rel = mon_path.relative_to(RES)
    family = rel.parts[0]
    run = str(rel.parent)
    try:
        mon = pickle.load(open(mon_path, "rb"))
    except Exception as e:                                        # noqa: BLE001
        continue
    # recover the decision interval from the paired log.pkl (states are (n_dec+1, D))
    dt_dec = np.nan
    log_path = mon_path.parent / "log.pkl"
    if log_path.exists():
        try:
            lg = pickle.load(open(log_path, "rb"))
            ss = lg.get("state_samples_history")
            if ss is not None and len(ss):
                n_dec = np.asarray(ss[0]).shape[0] - 1
                if n_dec > 0:
                    dt_dec = CONTROL_H / n_dec
        except Exception:                                          # noqa: BLE001
            pass

    for ep, m in enumerate(mon):
        t = np.asarray(m["t"], float)
        P = np.asarray(m["P"], float)
        if len(t) < 100:
            continue
        mask = flag_excursions(t, P)
        n = len(t)
        # map native steps -> GP decision transitions
        if np.isfinite(dt_dec):
            spd = int(round(dt_dec / DT))
            ntr = n // spd
            tr_bad = np.array([mask[i * spd:(i + 1) * spd].any() for i in range(ntr)], dtype=bool)
            tr_t = (np.arange(ntr) + 1) * dt_dec
        else:
            tr_bad, tr_t, ntr = np.zeros(0, bool), np.zeros(0), 0
        rows.append(dict(
            family=family, run=run, episode=ep, dt_dec=dt_dec,
            n_steps=n, n_bad_steps=int(mask.sum()),
            hours_bad=mask.sum() * DT,
            n_tr=ntr, n_bad_tr=int(tr_bad.sum()),
            bad_tr_phase1=int((tr_bad & (tr_t <= PIVOT_H)).sum()),
            bad_tr_phase2=int((tr_bad & (tr_t > PIVOT_H)).sum()),
            n_tr_phase1=int((tr_t <= PIVOT_H).sum()), n_tr_phase2=int((tr_t > PIVOT_H).sum()),
            first_bad_h=float(t[mask][0]) if mask.any() else np.nan,
            final_P=float(P[-1]),
        ))

df = pd.DataFrame(rows)
df.to_csv(HERE / "out" / "log_audit.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 40)

print(f"scanned {df.run.nunique()} runs / {len(df)} training episodes "
      f"across {df.family.nunique()} experiment families")
print(f"decision intervals found: {sorted(df.dt_dec.dropna().unique())}")
print()
print("=" * 96)
print("OVERALL CONTAMINATION")
print("=" * 96)
tot_tr, bad_tr = df.n_tr.sum(), df.n_bad_tr.sum()
print(f"  native timesteps   : {df.n_bad_steps.sum():>8,} / {df.n_steps.sum():>8,} "
      f"= {df.n_bad_steps.sum()/df.n_steps.sum():.2%}")
print(f"  GP TRANSITIONS     : {bad_tr:>8,} / {tot_tr:>8,} = {bad_tr/tot_tr:.2%}"
      "   <-- what the GP actually trains on")
print(f"  episodes with >=1 excursion : {(df.n_bad_steps>0).sum()} / {len(df)} "
      f"= {(df.n_bad_steps>0).mean():.1%}")
print(f"  episodes with >10 h contaminated : {(df.hours_bad>10).sum()} / {len(df)} "
      f"= {(df.hours_bad>10).mean():.1%}")
print()
print("distribution of contaminated hours per episode:")
q = df.hours_bad.describe(percentiles=[.5, .75, .9, .95, .99])
print("  " + "  ".join(f"{k}={v:.1f}" for k, v in q.items() if k != "count"))

print()
print("=" * 96)
print("WHERE IN THE BATCH  (phase 1 = t<=90 h, phase 2 = t>90 h)")
print("=" * 96)
p1b, p1n = df.bad_tr_phase1.sum(), df.n_tr_phase1.sum()
p2b, p2n = df.bad_tr_phase2.sum(), df.n_tr_phase2.sum()
print(f"  phase 1 (growth,     t<=90 h): {p1b:>7,} / {p1n:>7,} transitions = {p1b/p1n:.2%}")
print(f"  phase 2 (production, t> 90 h): {p2b:>7,} / {p2n:>7,} transitions = {p2b/p2n:.2%}")
print(f"  => {p2b/max(p1b+p2b,1):.1%} of ALL contaminated transitions fall in phase 2")

print()
print("=" * 96)
print("BY EXPERIMENT FAMILY")
print("=" * 96)
fam = df.groupby("family").agg(
    runs=("run", "nunique"), eps=("episode", "size"),
    pct_steps=("n_bad_steps", lambda s: 100 * s.sum() / df.loc[s.index, "n_steps"].sum()),
    pct_tr=("n_bad_tr", lambda s: 100 * s.sum() / max(df.loc[s.index, "n_tr"].sum(), 1)),
    eps_hit=("n_bad_steps", lambda s: 100 * (s > 0).mean()),
    med_h=("hours_bad", "median"), p95_h=("hours_bad", lambda s: s.quantile(.95)),
).sort_values("pct_tr", ascending=False)
print(fam.to_string(float_format=lambda v: f"{v:.1f}"))

print()
print("=" * 96)
print("WORST RUNS (by fraction of contaminated GP transitions)")
print("=" * 96)
r = df.groupby("run").agg(eps=("episode", "size"), bad_tr=("n_bad_tr", "sum"),
                          tr=("n_tr", "sum"), med_h=("hours_bad", "median"),
                          max_h=("hours_bad", "max"))
r["pct_tr"] = 100 * r.bad_tr / r.tr.clip(lower=1)
print(r[r.tr > 0].sort_values("pct_tr", ascending=False).head(15)
      .to_string(float_format=lambda v: f"{v:.1f}"))
print()
print("cleanest runs:")
print(r[r.tr > 0].sort_values("pct_tr").head(6).to_string(float_format=lambda v: f"{v:.1f}"))

print()
print("=" * 96)
print("DOES CONTAMINATION GROW AS THE POLICY LEARNS?  (episode index vs contaminated hours)")
print("=" * 96)
g = df.groupby("episode").hours_bad.agg(["mean", "median", "size"])
print(g.head(14).to_string(float_format=lambda v: f"{v:.1f}"))

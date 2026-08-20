"""Flag CO2/DO2 inhibition excursions from the P channel alone.

The training monitors log only [t, PAA, Viscosity, Wt, P, Fs, Fpaa, discharge], so the
inhibition terms are not directly available. But an excursion has a hard signature: when
CO2_inhib (or DO_2_inhib_P) collapses, r_p_gross goes to EXACTLY zero while the degradation
term mu_h*P keeps running, so dP/dt flips negative. Outside an excursion dP/dt is positive
throughout the production phase. Detector: dP/dt < 0 after production has started.
"""
import numpy as np

DT = 0.2


def flag_excursions(t, P, t_min=25.0, thresh=0.0):
    """Return a boolean mask over native timesteps where an inhibition excursion is active."""
    t = np.asarray(t, float)
    P = np.asarray(P, float)
    dP = np.gradient(P, DT)
    return (dP < thresh) & (t >= t_min) & np.isfinite(dP)


if __name__ == "__main__":
    # ---- validate against the 40 batches where the true CO2_inhib/DO2_inhib_P are known ----
    from pathlib import Path
    import pandas as pd
    HERE = Path(__file__).resolve().parent
    allb = pd.read_pickle(HERE / "out" / "all_aug.pkl")

    tp = fp = fn = tn = 0
    per_batch = []
    for b, g in allb.groupby("batch", sort=False):
        truth = ((g.CO2_inhib < 0.5) | (g.DO_2_inhib_P < 0.5)).values & (g.time_h.values >= 25)
        pred = flag_excursions(g.time_h.values, g.P_log.values)
        tp += (truth & pred).sum(); fp += (~truth & pred).sum()
        fn += (truth & ~pred).sum(); tn += (~truth & ~pred).sum()
        per_batch.append((b, truth.sum() * DT, pred.sum() * DT, g.batch_yield.iloc[0]))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"detector validation over 40 labelled batches (native 0.2 h steps)")
    print(f"  true excursion steps  : {tp+fn}")
    print(f"  flagged steps         : {tp+fp}")
    print(f"  recall    = {rec:.3f}   (fraction of real excursions caught)")
    print(f"  precision = {prec:.3f}   (fraction of flags that are real excursions)")
    print()
    print("per batch: true excursion hours vs detected hours")
    df = pd.DataFrame(per_batch, columns=["batch", "true_h", "flagged_h", "yield_kg"])
    print(df.sort_values("true_h", ascending=False).head(12)
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print("batches with zero true excursions -- what does the detector claim?")
    z = df[df.true_h == 0]
    print(f"  n={len(z)}, mean spurious flagged hours = {z.flagged_h.mean():.2f} "
          f"(max {z.flagged_h.max():.1f})")

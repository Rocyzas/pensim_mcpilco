"""
PYTHONPATH=.. python -m evaluations.action_deafness_margin_significance <diff_raw_csv> \
    [--split_hour 80]

Turns action_deafness_margin.py's --group_time/--group_no_time output
(action_deafness_margin_diff_raw.csv: one with_time-minus-no_time margin value per
pair/channel/hour) into a quantified per-channel significance test, instead of eyeballing the
diff plot's mean+band.

Design: within each seed-pair, the many hourly diff values are NOT independent samples (they're
correlated points along the same pair of batches) -- treating all of them as independent would
be pseudo-replication and overstate significance. So each pair is first collapsed to ONE mean
diff value (per channel, per time window), giving n_pairs independent observations, then a
one-sample (paired) t-test against 0 is run with df = n_pairs - 1. With only a handful of
with_time/no_time training-seed pairs available in this codebase, n is small (typically 3) --
the t-test (not a z/normal approximation) is used throughout for that reason, and the printed
p-value should be read as suggestive rather than conclusive at that sample size.

Reports three windows per channel: the full batch, and an early/late split at --split_hour
(default 80h, matching this codebase's own "does the deafness-margin improvement hold up past
~80h" question) -- so a channel whose improvement is real but concentrated early (or degrades
late) shows up as such, rather than being averaged away or hidden inside one full-batch number.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


def _paired_ttest(sub, window_label):
    per_pair = sub.groupby("pair")["diff_signal_minus_noise"].mean()
    n = len(per_pair)
    mean = float(per_pair.mean())
    sd = float(per_pair.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and sem > 0:
        t = mean / sem
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    else:
        t, p = float("nan"), float("nan")
    return {
        "window": window_label, "n_pairs": n, "mean_diff": mean, "sd": sd, "sem": sem,
        "t": t, "df": n - 1, "p_twotailed": p,
        "per_pair_values": [round(v, 6) for v in per_pair.values],
    }


def significance_table(raw_csv_path, split_hour=80.0):
    df = pd.read_csv(raw_csv_path)
    windows = {
        "full_batch": df,
        f"early_<=<{split_hour:g}h": df[df["hours"] <= split_hour],
        f"late_>{split_hour:g}h": df[df["hours"] > split_hour],
    }
    rows = []
    for channel in sorted(df["channel"].unique()):
        sub_ch = df[df["channel"] == channel]
        for wlabel, wdf in windows.items():
            sub = sub_ch[sub_ch["hours"].isin(wdf["hours"].unique())]
            r = _paired_ttest(sub, wlabel)
            r["channel"] = channel
            rows.append(r)
    cols = ["channel", "window", "n_pairs", "mean_diff", "sd", "sem", "t", "df",
           "p_twotailed", "per_pair_values"]
    return pd.DataFrame(rows)[cols]


def main(raw_csv_path, split_hour=80.0, out_csv=None):
    raw_csv_path = Path(raw_csv_path)
    table = significance_table(raw_csv_path, split_hour=split_hour)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 40)
    print(table.drop(columns=["per_pair_values"]).to_string(index=False,
         formatters={"mean_diff": "{:.5f}".format, "sd": "{:.5f}".format,
                    "sem": "{:.5f}".format, "t": "{:.3f}".format,
                    "p_twotailed": "{:.4f}".format}))
    out_csv = raw_csv_path.parent / "action_deafness_margin_significance.csv" if out_csv is None \
        else Path(out_csv)
    table.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    sig = table[table["p_twotailed"] < 0.05]
    if len(sig):
        print(f"\nsignificant at p<0.05 (n_pairs={table['n_pairs'].iloc[0]}, uncorrected for "
             f"multiple comparisons -- {len(table)} tests run):")
        for _, r in sig.iterrows():
            print(f"  {r['channel']:>10} / {r['window']:<14} mean_diff={r['mean_diff']:+.5f} "
                 f"t({r['df']:.0f})={r['t']:.2f} p={r['p_twotailed']:.4f}")
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_csv", type=str,
                   help="path to an action_deafness_margin_diff_raw.csv produced by "
                        "action_deafness_margin.py --group_time/--group_no_time")
    p.add_argument("--split_hour", type=float, default=80.0,
                   help="hour to split the early/late windows at (default 80)")
    p.add_argument("--out_csv", type=str, default=None,
                   help="where to save the table (default: alongside raw_csv)")
    args = p.parse_args()
    main(args.raw_csv, split_hour=args.split_hour, out_csv=args.out_csv)

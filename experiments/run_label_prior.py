"""
Label-prior (BBSE) drift signal over the monitoring period (t = 35..49), plus the
combined trigger D_t.  Calibration uses development timesteps (<= 34) only.

Trigger rule (fixed in advance):  D_t alarms when, at two consecutive timesteps,
the MMD signal OR the label-prior signal has fired.

    python experiments/run_label_prior.py --out results
Needs results/drift_mmd_scan.csv in --out (from run_drift_detector.py) to build D_t.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drift.label_prior import label_prior_scan
from protocol import TEST_T, VAL_T, build_protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    splits, _ = build_protocol(a.data)
    names = ("fit", "val", "test")
    x = np.concatenate([splits[s]["x"] for s in names])
    t = np.concatenate([splits[s]["timestep"] for s in names])
    y = np.concatenate([splits[s]["y"] for s in names])
    lab = np.concatenate([splits[s]["labelled"] for s in names])

    scan, null_df, info = label_prior_scan(
        x, t, y, lab, max(VAL_T), max(TEST_T), a.alpha, a.block, a.seed)
    print(f"\nFlag threshold={info['threshold']:.2f}  TPR={info['tpr']:.3f}  FPR={info['fpr']:.3f}  "
          f"usual fraud share (dev mean of pi_hat)={info['pi_bar']:.4f}")
    print(f"Null s over dev t=1-{max(VAL_T)} (n={len(null_df)}): mean={null_df['s'].mean():.3f} "
          f"max={null_df['s'].max():.3f}")
    cols = ["timestep", "flagged_share", "pi_hat", "ratio_vs_dev", "s", "p_value",
            "direction", "fired", "fired_2_in_row"]
    print(scan[cols].round(4).to_string(index=False))
    first = scan.loc[scan["fired"], "timestep"]
    print(f"\nLabel-prior first alarm: {first.min() if len(first) else 'none'}")
    scan.to_csv(os.path.join(a.out, "drift_label_prior_scan.csv"), index=False)
    null_df.to_csv(os.path.join(a.out, "drift_label_prior_null.csv"), index=False)

    mmd_path = os.path.join(a.out, "drift_mmd_scan.csv")
    if not os.path.exists(mmd_path):
        print(f"\n(No {mmd_path}; run experiments/run_drift_detector.py to get the combined D_t.)")
        return
    mmd = pd.read_csv(mmd_path)[["timestep", "fired"]].rename(columns={"fired": "mmd_fired"})
    d = scan[["timestep", "fired"]].rename(columns={"fired": "prior_fired"}).merge(mmd, on="timestep")
    d["either"] = d["mmd_fired"] | d["prior_fired"]
    d["D_t_alarm"] = d["either"] & d["either"].shift(1, fill_value=False)
    print("\n=== Combined trigger D_t (2 consecutive timesteps of MMD or label-prior alarm) ===")
    print(d.to_string(index=False))
    first = d.loc[d["D_t_alarm"], "timestep"]
    print(f"\nFirst D_t alarm: {first.min() if len(first) else 'none'}")
    d.to_csv(os.path.join(a.out, "drift_trigger_Dt.csv"), index=False)


if __name__ == "__main__":
    main()
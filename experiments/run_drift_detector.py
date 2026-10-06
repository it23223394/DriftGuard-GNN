"""
Run the MMD drift monitor over the monitoring period (t = 35..49).
Calibration uses development timesteps (<= 34) only; no labels are used anywhere.

    python experiments/run_drift_detector.py
    python experiments/run_drift_detector.py --window 5 --alpha 0.05 --out results
Writes results/drift_mmd_scan.csv and results/drift_mmd_null.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drift.detector import MMDMonitor
from protocol import TEST_T, VAL_T, build_protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--n_sub", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    splits, _ = build_protocol(a.data)
    x = np.concatenate([splits[s]["x"] for s in ("fit", "val", "test")])
    t = np.concatenate([splits[s]["timestep"] for s in ("fit", "val", "test")])

    last_dev = max(VAL_T)
    mon = MMDMonitor(x, t, a.window, a.n_sub, a.alpha, a.seed)
    mon.calibrate(last_dev)
    print(f"\nNull (dev timesteps {mon.null_ts[0]}-{mon.null_ts[-1]}, n={len(mon.null)}): "
          f"mean={mon.null.mean():.5f} std={mon.null.std():.5f} max={mon.null.max():.5f}")

    df = pd.DataFrame([mon.step(ts) for ts in range(last_dev + 1, max(TEST_T) + 1)])
    df["fired_2_in_row"] = df["fired"] & df["fired"].shift(1, fill_value=False)
    print(df.round(5).to_string(index=False))
    first = df.loc[df["fired"], "timestep"]
    first2 = df.loc[df["fired_2_in_row"], "timestep"]
    print(f"\nFirst alarm: {first.min() if len(first) else 'none'}   "
          f"first 2-in-a-row alarm: {first2.min() if len(first2) else 'none'}")

    df.to_csv(os.path.join(a.out, "drift_mmd_scan.csv"), index=False)
    pd.DataFrame({"timestep": mon.null_ts, "mmd2": mon.null}).to_csv(
        os.path.join(a.out, "drift_mmd_null.csv"), index=False)


if __name__ == "__main__":
    main()
"""
Combine every results/baselines_per_timestep*.csv (e.g. runs split by model or seed)
and print the summary tables. Duplicate (model, seed, policy, split, timestep) rows
keep the latest file.

    python experiments/summarize_results.py            # reads results/
    python experiments/summarize_results.py my_folder
"""
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate import summarize


def main(out="results"):
    files = sorted(glob.glob(os.path.join(out, "baselines_per_timestep*.csv")))
    if not files:
        sys.exit(f"no baselines_per_timestep*.csv files found in {out}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["model", "seed", "policy", "split", "timestep"]
    df = df.drop_duplicates(subset=key, keep="last")
    df.to_csv(os.path.join(out, "all_baselines_per_timestep.csv"), index=False)

    print(f"{len(files)} result files combined.")
    print("\nSeeds per model:")
    print(df[df["timestep"] == -1].groupby("model")["seed"].nunique().to_string())
    pooled, macro = summarize(df)
    print("\n=== Test, pooled (mean/std over seeds) ===")
    print(pooled.to_string())
    print("\n=== Test, macro per-timestep F1 by window (descriptive only) ===")
    print(macro.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:2])
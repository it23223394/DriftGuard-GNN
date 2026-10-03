"""
Train all baselines under the strict protocol and write:
  results/baselines_per_timestep.csv  (model, seed, policy, thr, split, timestep, metrics)
  results/baselines_summary.csv       (pooled test metrics, mean/std over seeds)
Each model is scored with two threshold policies: val_f1 (tuned on val) and fixed_0.5.

    python experiments/run_baselines.py                      # all models, 5 seeds
    python experiments/run_baselines.py --models rf_all xgb_all --seeds 42
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import baselines
from evaluate import per_timestep, pick_threshold, summarize
from protocol import build_protocol, set_seed

ALL_MODELS = baselines.SKLEARN_MODELS + baselines.GNN_MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 11])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)

    splits, _ = build_protocol(a.data)
    lab = {s: splits[s]["labelled"] for s in ("val", "test")}
    y = {s: splits[s]["y"][lab[s]] for s in lab}
    ts = {s: splits[s]["timestep"][lab[s]] for s in lab}

    frames = []
    for seed in a.seeds:
        for name in a.models:
            set_seed(seed)
            model, probs = baselines.fit(name, splits, seed, epochs=a.epochs)
            if not a.no_ckpt:
                baselines.save(name, model, os.path.join(a.ckpt, f"{name}_seed{seed}"))
            thr_val = pick_threshold(y["val"], probs["val"])
            for policy, thr in (("val_f1", thr_val), ("fixed_0.5", 0.5)):
                for s in ("val", "test"):
                    df = per_timestep(y[s], probs[s], ts[s], thr)
                    df.insert(0, "split", s)
                    df.insert(0, "thr", thr)
                    df.insert(0, "policy", policy)
                    df.insert(0, "seed", seed)
                    df.insert(0, "model", name)
                    frames.append(df)
                    if policy == "val_f1" and s == "test":
                        pooled = df[df.timestep == -1].iloc[0]
            print(f"[seed {seed}] {name:9s} thr={thr_val:.2f}  test pooled F1={pooled.f1:.3f} "
                  f"AUPRC={pooled.auprc:.3f}")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(os.path.join(a.out, "baselines_per_timestep.csv"), index=False)
    pooled, macro = summarize(results)
    pooled.to_csv(os.path.join(a.out, "baselines_summary.csv"))
    print("\n=== Test, pooled (mean/std over seeds) ===")
    print(pooled.to_string())
    print("\n=== Test, macro per-timestep F1 by window (descriptive only) ===")
    print(macro.to_string())


if __name__ == "__main__":
    main()
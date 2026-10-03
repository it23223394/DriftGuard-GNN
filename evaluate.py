"""
evaluate.py - metrics for the DriftGuard-GNN protocol. Illicit = positive class.

Per-timestep rows use timestep = -1 for the pooled (all timesteps in the split) row.
Undefined values (e.g. recall/F1/AUPRC at a timestep with no illicit nodes) are NaN.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def metrics(y, prob, thr):
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = prob >= thr
    n_pos = int((y == 1).sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    if n_pos == 0:
        return dict(n=len(y), n_illicit=0, precision=precision,
                    recall=np.nan, f1=np.nan, auprc=np.nan)
    fn = n_pos - tp
    return dict(
        n=len(y), n_illicit=n_pos, precision=precision, recall=tp / n_pos,
        f1=2 * tp / (2 * tp + fp + fn),
        auprc=float(average_precision_score(y, prob)),
    )


def pick_threshold(y_val, p_val):
    """Threshold maximising illicit-class F1 on the VALIDATION split only."""
    scores = [np.nan_to_num(metrics(y_val, p_val, t)["f1"]) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(scores))])


def per_timestep(y, prob, timestep, thr):
    """One row per timestep plus a pooled row (timestep = -1)."""
    y, prob, timestep = np.asarray(y), np.asarray(prob), np.asarray(timestep)
    rows = [dict(timestep=-1, **metrics(y, prob, thr))]
    for t in np.unique(timestep):
        m = timestep == t
        rows.append(dict(timestep=int(t), **metrics(y[m], prob[m], thr)))
    return pd.DataFrame(rows)


def summarize(results, split="test"):
    """Mean/std across seeds: pooled metrics, plus macro per-timestep F1 by window."""
    d = results[(results["split"] == split) & (results["timestep"] == -1)]
    pooled = (d.groupby(["model", "policy"])[["precision", "recall", "f1", "auprc"]]
                .agg(["mean", "std"]).round(3))

    t = results[(results["split"] == split) & (results["timestep"] > 0)].copy()
    t["window"] = np.where(t["timestep"] <= 40, "f1_t35_40", "f1_t41_49")
    macro = (t.groupby(["model", "policy", "seed", "window"])["f1"].mean()   # NaNs skipped
              .groupby(["model", "policy", "window"]).agg(["mean", "std"]).round(3)
              .unstack("window"))
    return pooled, macro
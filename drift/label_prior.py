"""
drift/label_prior.py - label-prior shift signal via BBSE (Lipton et al., 2018).

Label-free at monitoring time: it only needs the classifier's predictions on ALL nodes
of a timestep (labelled and unknown), never the timestep's labels.

  1. Cross-fit a random forest on the DEVELOPMENT period (t <= last_dev_t) by holding
     out blocks of consecutive timesteps, so every dev node gets an out-of-sample score.
  2. Pick a flag threshold and measure TPR / FPR on dev labelled nodes
     (leave-one-timestep-out for the null, pooled for monitoring).
  3. At timestep t: mu_t = share of nodes flagged;  pi_hat_t = (mu_t - FPR) / (TPR - FPR).
  4. Signal s_t = |log((pi_hat_t + eps) / (pi_bar_dev + eps))|  (how many times larger or
     smaller the fraud share is than its usual development level).
  5. Null = s_t over dev timesteps. Conformal p-value, alarm if p <= alpha.
No test labels or test features are used for calibration.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluate import pick_threshold

EPS = 1e-3


def _rf(seed):
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")


def cross_fit_probs(x, t, y, lab, last_dev_t, block=3, seed=42):
    """Out-of-fold fraud probability for every dev node (labelled or not)."""
    dev = t <= last_dev_t
    p = np.full(len(x), np.nan)
    blocks = (t - 1) // block
    for b in np.unique(blocks[dev]):
        hold = dev & (blocks == b)
        train = dev & lab & ~hold
        p[hold] = _rf(seed).fit(x[train], y[train]).predict_proba(x[hold])[:, 1]
    return p


def bbse_pi(mu, tpr, fpr):
    return float(np.clip((mu - fpr) / max(tpr - fpr, 1e-6), 0.0, 1.0))


def label_prior_scan(x, t, y, lab, last_dev_t, last_t, alpha=0.05, block=3, seed=42):
    dev = t <= last_dev_t
    p = cross_fit_probs(x, t, y, lab, last_dev_t, block, seed)
    final = _rf(seed).fit(x[dev & lab], y[dev & lab])          # model used while monitoring
    mon = (t > last_dev_t) & (t <= last_t)
    p[mon] = final.predict_proba(x[mon])[:, 1]

    dl = dev & lab
    thr = pick_threshold(y[dl], p[dl])
    flag = p >= thr

    dev_ts = np.arange(1, last_dev_t + 1)
    tp = np.array([(flag & lab & (y == 1) & (t == k)).sum() for k in dev_ts], float)
    pos = np.array([(lab & (y == 1) & (t == k)).sum() for k in dev_ts], float)
    fp = np.array([(flag & lab & (y == 0) & (t == k)).sum() for k in dev_ts], float)
    neg = np.array([(lab & (y == 0) & (t == k)).sum() for k in dev_ts], float)
    tpr_all, fpr_all = tp.sum() / pos.sum(), fp.sum() / neg.sum()

    rows = []
    for k in range(1, last_t + 1):
        mu = float(flag[t == k].mean())
        if k <= last_dev_t:                       # leave this timestep out of its own rates
            i = k - 1
            tpr = (tp.sum() - tp[i]) / max(pos.sum() - pos[i], 1)
            fpr = (fp.sum() - fp[i]) / max(neg.sum() - neg[i], 1)
        else:
            tpr, fpr = tpr_all, fpr_all
        rows.append(dict(timestep=k, flagged_share=mu, pi_hat=bbse_pi(mu, tpr, fpr)))
    df = pd.DataFrame(rows)

    pi_bar = df.loc[df.timestep <= last_dev_t, "pi_hat"].mean()
    df["ratio_vs_dev"] = (df["pi_hat"] + EPS) / (pi_bar + EPS)
    df["s"] = np.abs(np.log(df["ratio_vs_dev"]))
    null_df = df[df.timestep <= last_dev_t].copy()
    null = null_df["s"].values

    scan = df[df.timestep > last_dev_t].copy()
    scan["p_value"] = [(1 + (null >= s).sum()) / (len(null) + 1) for s in scan["s"]]
    scan["fired"] = scan["p_value"] <= alpha
    scan["direction"] = np.where(scan["ratio_vs_dev"] < 1, "down", "up")
    scan["fired_2_in_row"] = scan["fired"] & scan["fired"].shift(1, fill_value=False)
    info = dict(threshold=thr, tpr=tpr_all, fpr=fpr_all, pi_bar=pi_bar)
    return scan.reset_index(drop=True), null_df.reset_index(drop=True), info
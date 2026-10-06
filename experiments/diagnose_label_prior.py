"""
Explain WHY the label-prior signal did or did not move: put the true labelled fraud rate
per timestep next to the signal's estimate. Diagnostic only - labels are used here to
explain the result, never by the detector.

    python experiments/diagnose_label_prior.py --out results
Needs drift_label_prior_scan.csv and drift_label_prior_null.csv in --out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from protocol import VAL_T, build_protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    splits, _ = build_protocol(a.data)
    names = ("fit", "val", "test")
    t = np.concatenate([splits[s]["timestep"] for s in names])
    y = np.concatenate([splits[s]["y"] for s in names])
    lab = np.concatenate([splits[s]["labelled"] for s in names])

    rows = []
    for k in range(1, int(t.max()) + 1):
        m = t == k
        n_lab = int((m & lab).sum())
        n_ill = int((m & lab & (y == 1)).sum())
        rows.append(dict(timestep=k, n_nodes=int(m.sum()), n_labelled=n_lab, n_illicit=n_ill,
                         illicit_rate_of_labelled=n_ill / max(n_lab, 1),
                         illicit_share_of_all_nodes=n_ill / int(m.sum())))
    truth = pd.DataFrame(rows)

    scan = pd.read_csv(os.path.join(a.out, "drift_label_prior_scan.csv"))
    null = pd.read_csv(os.path.join(a.out, "drift_label_prior_null.csv"))
    est = pd.concat([null, scan])[["timestep", "flagged_share", "pi_hat"]]
    df = truth.merge(est, on="timestep").sort_values("timestep")
    print(df.round(4).to_string(index=False))

    last_dev = max(VAL_T)
    for name, d in (("dev (t<=%d)" % last_dev, df[df.timestep <= last_dev]),
                    ("monitoring (t>%d)" % last_dev, df[df.timestep > last_dev])):
        r = np.corrcoef(d["pi_hat"], d["illicit_rate_of_labelled"])[0, 1]
        print(f"\n{name}: mean pi_hat={d['pi_hat'].mean():.4f} | mean illicit share of ALL nodes="
              f"{d['illicit_share_of_all_nodes'].mean():.4f} | mean illicit rate of LABELLED nodes="
              f"{d['illicit_rate_of_labelled'].mean():.4f} | corr(pi_hat, labelled rate)={r:.2f}")


if __name__ == "__main__":
    main()
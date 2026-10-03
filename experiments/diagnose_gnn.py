"""
Diagnose GNN training: loss and AUPRC per epoch on FIT and VAL (test is never used).

    python experiments/diagnose_gnn.py --kind gcn --epochs 300
    python experiments/diagnose_gnn.py --kind gcn --epochs 300 --wd 0 --hidden 128 --no_rf
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from sklearn.metrics import average_precision_score

import baselines
from protocol import N_LOCAL, build_protocol, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gcn", choices=["gcn", "sage", "gat"])
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_rf", action="store_true", help="skip the RF reference (saves time)")
    a = ap.parse_args()

    splits, _ = build_protocol("data")
    n_feat = N_LOCAL if a.local else None

    if not a.no_rf:  # reference: what a random forest reaches on the same val split
        _, rf_probs = baselines.fit_sklearn("rf_local" if a.local else "rf_all", splits, a.seed)
        y_val = splits["val"]["y"][splits["val"]["labelled"]]
        print(f"\nRF reference val AUPRC: {average_precision_score(y_val, rf_probs['val']):.3f}")

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = {k: baselines._graph(splits[k], device, n_feat) for k in ("fit", "val")}
    xf, eif, yf, mf = g["fit"]
    model = baselines.make_gnn(a.kind, xf.shape[1], hidden=a.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n_pos = (yf[mf] == 1).sum().item()
    n_neg = (yf[mf] == 0).sum().item()
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device))

    def auprc(key):
        x, ei, y, m = g[key]
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(x, ei), dim=1)[:, 1]
        return average_precision_score(y[m].cpu().numpy(), p[m].cpu().numpy())

    print(f"\n{a.kind} ({'local' if a.local else 'all'} features), "
          f"lr={a.lr}, wd={a.wd}, hidden={a.hidden}")
    print("epoch    loss  fit_auprc  val_auprc")
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(xf, eif)[mf], yf[mf])
        loss.backward()
        opt.step()
        if ep < 10 or ep % 25 == 0 or ep == a.epochs - 1:
            print(f"{ep:5d}  {loss.item():.4f}  {auprc('fit'):9.3f}  {auprc('val'):9.3f}")


if __name__ == "__main__":
    main()
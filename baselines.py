"""
baselines.py - RF, XGBoost, GCN, GraphSAGE, GAT trained on the protocol splits.

Trees use only the labelled fit nodes. GNNs train on the FIT graph only, stop early
on VAL AUPRC (val graph is a separate graph), and are applied to the val/test graphs
separately (inductive). Nothing here touches the test split before final scoring.

Decisions to state in the methodology:
  * GNN graphs are symmetrised (UNDIRECTED = True).
  * *_local models use only the 93 local features (no pre-aggregated ones), so
    gcn_local vs rf_local isolates whether graph structure helps.
"""

import numpy as np
from protocol import set_seed, N_LOCAL

UNDIRECTED = True
SKLEARN_MODELS = ["rf_all", "rf_local", "xgb_all"]
GNN_MODELS = ["gcn", "sage", "gat", "gcn_local", "sage_local", "gat_local"]


def _xy(split, n_feat=None):
    x = split["x"] if n_feat is None else split["x"][:, :n_feat]
    m = split["labelled"]
    return x[m], split["y"][m]


def fit_sklearn(name, splits, seed):
    n_feat = N_LOCAL if name.endswith("_local") else None
    xf, yf = _xy(splits["fit"], n_feat)
    if name.startswith("rf"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                       class_weight="balanced_subsample")
    else:
        from xgboost import XGBClassifier
        spw = (yf == 0).sum() / max((yf == 1).sum(), 1)
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              tree_method="hist", random_state=seed, n_jobs=-1)
    model.fit(xf, yf)
    probs = {s: model.predict_proba(_xy(splits[s], n_feat)[0])[:, 1] for s in ("val", "test")}
    return model, probs


def make_gnn(kind, in_dim, hidden=64, dropout=0.5):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "gcn":
                self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, 2)
            elif kind == "sage":
                self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, 2)
            else:  # gat
                self.c1 = GATConv(in_dim, hidden // 4, heads=4)
                self.c2 = GATConv(hidden, 2, heads=1)

        def forward(self, x, ei):
            x = torch.relu(self.c1(x, ei))
            x = torch.nn.functional.dropout(x, dropout, self.training)
            return self.c2(x, ei)

    return Net()


def _graph(split, device, n_feat=None):
    import torch
    x_np = split["x"] if n_feat is None else split["x"][:, :n_feat]
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    ei = torch.tensor(split["edge_index"], dtype=torch.long, device=device)
    if UNDIRECTED:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    y = torch.tensor(split["y"], dtype=torch.long, device=device)
    m = torch.tensor(split["labelled"], dtype=torch.bool, device=device)
    return x, ei, y, m


def fit_gnn(name, splits, seed, epochs=300, patience=50, lr=0.01, wd=0.0):
    import torch
    from sklearn.metrics import average_precision_score
    set_seed(seed)
    kind = name.replace("_local", "")
    n_feat = N_LOCAL if name.endswith("_local") else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = {k: _graph(splits[k], device, n_feat) for k in ("fit", "val", "test")}
    xf, eif, yf, mf = g["fit"]
    model = make_gnn(kind, xf.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_pos = (yf[mf] == 1).sum().item()
    n_neg = (yf[mf] == 0).sum().item()
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device))

    def probs(key):
        x, ei, y, m = g[key]
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(x, ei), dim=1)[:, 1]
        return p[m].cpu().numpy(), y[m].cpu().numpy()

    best, bad, best_state, best_ep = -1.0, 0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(xf, eif)[mf], yf[mf]).backward()
        opt.step()
        pv, yv = probs("val")
        score = average_precision_score(yv, pv)
        if score > best:
            best, bad, best_ep = score, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.best_epoch = best_ep
    return model, {"val": probs("val")[0], "test": probs("test")[0]}


def fit(name, splits, seed, epochs=300):
    if name in SKLEARN_MODELS:
        return fit_sklearn(name, splits, seed)
    return fit_gnn(name, splits, seed, epochs=epochs)


def save(name, model, path_no_ext):
    if name in SKLEARN_MODELS:
        import joblib
        joblib.dump(model, path_no_ext + ".joblib", compress=3)
    else:
        import torch
        torch.save(model.state_dict(), path_no_ext + ".pt")
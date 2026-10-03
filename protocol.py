"""
protocol.py - strict, leak-free Elliptic protocol for DriftGuard-GNN.

Three disjoint, time-ordered splits, each with its OWN graph:
    fit   t=1-28   model training
    val   t=29-34  threshold selection / early stopping (never test)
    test  t=35-49  final evaluation only

Scaler is fit on the fit split only. Every split is checked by
check_no_leakage() before it is returned.
"""

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FIT_T = range(1, 29)
VAL_T = range(29, 35)
TEST_T = range(35, 50)
N_FEATURES = 165          # 93 local + 72 aggregated (aggregated = cols 93+)
N_LOCAL = 93


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_elliptic(data_dir="data"):
    feats = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")
    feats.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(N_FEATURES)]
    classes.columns = ["txId", "class"]
    nodes = feats.merge(classes, on="txId", how="left")
    nodes["label"] = nodes["class"].astype(str).map({"1": 1, "2": 0})  # unknown -> NaN
    return nodes, edges


def _make_split(nodes, edges, timesteps):
    """Nodes in `timesteps` plus ONLY edges with both endpoints inside them."""
    df = nodes[nodes["timestep"].isin(list(timesteps))].reset_index(drop=True)
    idx = pd.Series(np.arange(len(df)), index=df["txId"].values)
    keep = edges["txId1"].isin(idx.index) & edges["txId2"].isin(idx.index)
    e = edges[keep]
    edge_index = np.vstack([idx[e["txId1"]].values, idx[e["txId2"]].values]).astype(np.int64)
    cols = [f"feat_{i}" for i in range(N_FEATURES)]
    return {
        "txId": df["txId"].values,
        "timestep": df["timestep"].values,
        "x_raw": df[cols].values.astype(np.float32),
        "y": df["label"].fillna(-1).astype(int).values,   # -1 = unknown
        "labelled": df["label"].notna().values,
        "edge_index": edge_index,
    }


def check_no_leakage(splits, scaler, edges):
    names = ["fit", "val", "test"]
    ids = [set(splits[n]["txId"]) for n in names]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), \
        "node overlap between splits"
    assert splits["fit"]["timestep"].max() < splits["val"]["timestep"].min(), "fit/val not time-ordered"
    assert splits["val"]["timestep"].max() < splits["test"]["timestep"].min(), "val/test not time-ordered"
    for n in names:
        s = splits[n]
        if s["edge_index"].size:
            assert s["edge_index"].max() < len(s["txId"]), f"{n}: edge points outside its own split"
    assert np.allclose(scaler.mean_, splits["fit"]["x_raw"].mean(axis=0), atol=1e-4), \
        "scaler was not fit on fit split only"
    split_of = {t: n for n, i in zip(names, ids) for t in i}
    s1 = edges["txId1"].map(split_of)
    s2 = edges["txId2"].map(split_of)
    cross = int(((s1 != s2) & s1.notna() & s2.notna()).sum())
    print(f"[leakage check] passed. Edges crossing splits (dropped): {cross}")
    return cross


def build_protocol(data_dir="data"):
    nodes, edges = load_elliptic(data_dir)
    splits = {
        "fit": _make_split(nodes, edges, FIT_T),
        "val": _make_split(nodes, edges, VAL_T),
        "test": _make_split(nodes, edges, TEST_T),
    }
    scaler = StandardScaler().fit(splits["fit"]["x_raw"])   # fit split only
    for s in splits.values():
        s["x"] = scaler.transform(s["x_raw"]).astype(np.float32)
    check_no_leakage(splits, scaler, edges)
    for n, s in splits.items():
        lab = s["y"][s["labelled"]]
        print(f"{n}: nodes={len(s['txId'])}, edges={s['edge_index'].shape[1]}, "
              f"labelled={len(lab)}, illicit rate={lab.mean():.4f}")
    return splits, scaler

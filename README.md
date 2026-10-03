# DriftGuard-GNN

Drift-triggered edge-reliability reweighting for GNN-based Bitcoin fraud
detection under leakage-free evaluation (SE4012).

## Data
Download the Elliptic dataset and place these files in `data/` (not committed):
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

## Protocol (`protocol.py`)
| Split | Timesteps | Use |
|-------|-----------|-----|
| fit   | 1-28      | model training |
| val   | 29-34     | threshold selection / early stopping |
| test  | 35-49     | final evaluation only |

Each split has its own graph. The scaler is fit on the fit split only.
`check_no_leakage()` runs automatically inside `build_protocol()`.

## Quick check
    python experiments/check_protocol.py

## Layout
- `protocol.py` - strict leak-free splits + seed helper
- `drift/` - MMD and label-prior drift detectors
- `recal/` - edge-trust re-fit and Bayes threshold correction
- `experiments/` - baselines, DriftGuard, ablations, label-lag
- `results/`, `checkpoints/` - outputs

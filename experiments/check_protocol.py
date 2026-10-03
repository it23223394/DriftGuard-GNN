"""Run the protocol on the real data and print split stats + leakage check."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol import build_protocol, set_seed

if __name__ == "__main__":
    set_seed(42)
    splits, scaler = build_protocol("data")

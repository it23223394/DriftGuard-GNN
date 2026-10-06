"""
drift/detector.py - label-free MMD feature-drift monitor.

Statistic at timestep t: unbiased RBF-MMD^2 between the nodes of timestep t and a
reference sample drawn from the previous `window` timesteps.
Null distribution: the same statistic over the DEVELOPMENT period only (no test data).
Decision: conformal p-value  p_t = (1 + #{null >= s_t}) / (n_null + 1);  alarm if p_t <= alpha.
The RBF bandwidth is fixed once from development data (median heuristic), so null and
monitoring statistics are comparable.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mmd2_unbiased(X, Y, gamma):
    n, m = len(X), len(Y)
    kxx = rbf_kernel(X, X, gamma=gamma)
    kyy = rbf_kernel(Y, Y, gamma=gamma)
    kxy = rbf_kernel(X, Y, gamma=gamma)
    term_x = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(term_x + term_y - 2.0 * kxy.mean())


def median_gamma(X, rng, n=2000):
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    S = X[idx].astype(np.float64)
    sq = (S ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * S @ S.T
    d2 = d2[np.triu_indices_from(d2, k=1)]
    return float(1.0 / np.median(d2[d2 > 0]))


class MMDMonitor:
    def __init__(self, x, t, window=5, n_sub=1000, alpha=0.05, seed=42):
        self.x, self.t = x, t
        self.window, self.n_sub, self.alpha = window, n_sub, alpha
        self.rng = np.random.default_rng(seed)
        self.gamma = None
        self.null_ts, self.null = [], np.array([])

    def _sample(self, mask):
        idx = np.flatnonzero(mask)
        if len(idx) > self.n_sub:
            idx = self.rng.choice(idx, self.n_sub, replace=False)
        return self.x[idx]

    def stat(self, ts):
        cur = self._sample(self.t == ts)
        ref = self._sample((self.t >= ts - self.window) & (self.t < ts))
        return mmd2_unbiased(cur, ref, self.gamma)

    def calibrate(self, last_dev_t):
        """Fix the bandwidth and build the null from timesteps <= last_dev_t only."""
        dev = self.t <= last_dev_t
        self.gamma = median_gamma(self.x[dev], self.rng)
        self.null_ts = list(range(self.window + 1, last_dev_t + 1))
        self.null = np.array([self.stat(ts) for ts in self.null_ts])

    def p_value(self, s):
        return float((1 + (self.null >= s).sum()) / (len(self.null) + 1))

    def step(self, ts):
        s = self.stat(ts)
        p = self.p_value(s)
        return dict(timestep=ts, mmd2=s, p_value=p, fired=bool(p <= self.alpha))
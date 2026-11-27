"""Per-clip statistics: Cohen's d, bootstrap CIs, t-test, Wilcoxon, Holm."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for Cohen's d over clips."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    samples = diff[idx]
    d = samples.mean(1) / samples.std(1, ddof=1)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def one_sample_summary(diff: np.ndarray) -> dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    lo, hi = bootstrap_ci(diff)
    return {
        "n": len(diff),
        "pct_positive": float((diff > 0).mean() * 100),
        "mean_diff": float(diff.mean()),
        "d": cohens_d(diff),
        "d_lo": lo,
        "d_hi": hi,
        "p_t": float(stats.ttest_1samp(diff, 0.0).pvalue),
        "p_wilcoxon": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

"""Pixel baselines scored on the same windows: repeat-last-frame and constant velocity.

These are the sanity anchors for Table 1: the mirror test must give exactly 0 for
both by construction, which checks the analysis pipeline end to end.
"""

from __future__ import annotations

import numpy as np


def repeat_last(seen: np.ndarray, n_pred: int) -> np.ndarray:
    return np.repeat(seen[-1:], n_pred, axis=0)


def constant_velocity(seen: np.ndarray, n_pred: int) -> np.ndarray:
    last, prev = seen[-1].astype(float), seen[-2].astype(float)
    v = last - prev
    return np.stack([np.clip(last + (k + 1) * v, 0, 255) for k in range(n_pred)])


def pixel_surprise(window: np.ndarray, seen: int, rule: str = "repeat_last") -> float:
    """Mean absolute pixel error (0..1 scale) of a rule on the predicted frames."""
    s, target = window[:seen], window[seen:].astype(float)
    fn = repeat_last if rule == "repeat_last" else constant_velocity
    pred = fn(s, len(target)).astype(float)
    return float(np.abs(pred - target).mean() / 255.0)

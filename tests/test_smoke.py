"""Fast CPU checks with a tiny randomly initialised V-JEPA 2 (no downloads).

Run: python -m pytest -q tests
"""

import numpy as np
import torch
from transformers import VJEPA2Config, VJEPA2Model

from aot.baselines import pixel_surprise
from aot.control import attach_trainable_predictor
from aot.stats import holm, one_sample_summary
from aot.stimuli import make_versions, synthetic_clip
from aot.surprise import surprise_steps, to_pixels, window_surprise


def tiny_model():
    cfg = VJEPA2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                       pred_hidden_size=32, pred_num_hidden_layers=2, pred_num_attention_heads=2,
                       crop_size=64, frames_per_clip=16)
    return VJEPA2Model(cfg).eval()


def test_versions_share_frames():
    clip = synthetic_clip(33, 32)
    v = make_versions(clip, flip_frame=16)
    assert all(x.shape == clip.shape for x in v.values())
    assert np.array_equal(v["RR"][0], clip[-1])
    assert np.array_equal(v["FR"][16], clip[16]) and np.array_equal(v["FR"][17], clip[15])
    assert np.array_equal(v["RF"][16], clip[16]) and np.array_equal(v["RF"][17], clip[17])


def test_surprise_shapes_and_mirror():
    torch.manual_seed(0)
    m = tiny_model()
    x = to_pixels(synthetic_clip(16, 64), 64, torch.device("cpu"), torch.float32)
    fwd = window_surprise(m, x, seen=8)
    end = window_surprise(m, x, seen=8, seen_at_end=True)
    assert fwd.shape == (4,) and end.shape == (4,)
    assert np.isfinite(fwd).all() and np.isfinite(end).all()


def test_pixel_baseline_is_nonnegative():
    clip = synthetic_clip(16, 32)
    s = pixel_surprise(clip, seen=8)
    assert s >= 0.0


def test_lora_backward():
    torch.manual_seed(0)
    m = attach_trainable_predictor(tiny_model(), rank=4, alpha=8)
    x = to_pixels(synthetic_clip(16, 64), 64, torch.device("cpu"), torch.float32).repeat(2, 1, 1, 1, 1)
    loss = surprise_steps(m, x, seen=8).mean()
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_stats():
    rng = np.random.default_rng(0)
    s = one_sample_summary(rng.normal(0.5, 1.0, 292))
    assert s["d_lo"] < s["d"] < s["d_hi"]
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

"""Feature-prediction surprise for V-JEPA 2 (Hugging Face transformers).

Surprise = L1 distance between the predictor output for the predicted frames and
the layer-normalised target-encoder features of those frames, averaged over
tokens and feature dimensions (as in Garrido et al. 2025).

Leak-free context: the context encoder only sees the seen frames. V-JEPA 2 uses
rotary position embeddings, so encoding the seen frames on their own gives the
same relative geometry they have inside the full window. The full window is
encoded separately to produce targets (the "target encoder" view).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from .config import load_config, resolve_device, save_config, seed_everything
from .stimuli import VERSIONS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass
class Grid:
    tubelet: int
    side: int  # patches per spatial side

    @property
    def per_slot(self) -> int:
        return self.side * self.side

    def slot_indices(self, slots: range, device: torch.device) -> torch.Tensor:
        idx = [s * self.per_slot + torch.arange(self.per_slot) for s in slots]
        return torch.cat(idx).unsqueeze(0).to(device)


def load_model(name: str, device: torch.device, dtype: str = "float16"):
    torch_dtype = getattr(torch, dtype) if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(name, torch_dtype=torch_dtype).to(device).eval()
    return model


def to_pixels(clip: np.ndarray, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """uint8 (T, H, W, 3) to normalised (1, T, 3, size, size)."""
    x = torch.from_numpy(clip).float().div(255).permute(0, 3, 1, 2)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x.unsqueeze(0) - MEAN) / STD
    return x.to(device=device, dtype=dtype)


def grid_for(model) -> Grid:
    c = model.config
    return Grid(tubelet=c.tubelet_size, side=c.crop_size // c.patch_size)


def surprise_steps(model, window: torch.Tensor, seen: int, seen_at_end: bool = False) -> torch.Tensor:
    """Differentiable surprise per predicted tubelet step, shape (B, steps).

    window: (B, T, 3, H, W). Encoders run without gradients (they stay frozen);
    gradients flow only through the predictor, which is what the positive
    control fine-tunes.
    seen_at_end=False: forecast the frames after the seen ones.
    seen_at_end=True: fill a masked beginning from the last `seen` frames.
    """
    g = grid_for(model)
    b = window.shape[0]
    t_slots = window.shape[1] // g.tubelet
    s_slots = seen // g.tubelet
    ctx_slots = range(t_slots - s_slots, t_slots) if seen_at_end else range(0, s_slots)
    tgt_slots = range(0, t_slots - s_slots) if seen_at_end else range(s_slots, t_slots)
    dev = window.device

    with torch.no_grad():
        full = model(pixel_values_videos=window, skip_predictor=True).last_hidden_state
        seen_frames = window[:, -seen:] if seen_at_end else window[:, :seen]
        ctx = model(pixel_values_videos=seen_frames, skip_predictor=True).last_hidden_state
        hybrid = full.clone()
        ctx_idx = g.slot_indices(ctx_slots, dev).repeat(b, 1)
        hybrid[:, ctx_idx[0]] = ctx
        tgt_idx = g.slot_indices(tgt_slots, dev).repeat(b, 1)
        target = F.layer_norm(full[:, tgt_idx[0]].float(), (full.shape[-1],))

    pred = model.predictor(encoder_hidden_states=hybrid, context_mask=[ctx_idx], target_mask=[tgt_idx]).last_hidden_state
    err = (pred.float() - target).abs().mean(-1)  # (B, n_target_tokens)
    return err.view(b, len(tgt_slots), g.per_slot).mean(-1)


@torch.no_grad()
def window_surprise(model, window: torch.Tensor, seen: int, seen_at_end: bool = False) -> np.ndarray:
    """Surprise per predicted step for a single window (1, T, 3, H, W) as numpy."""
    return surprise_steps(model, window, seen, seen_at_end)[0].cpu().numpy()


def windows(clip: torch.Tensor, length: int, stride: int):
    for start in range(0, clip.shape[1] - length + 1, stride):
        yield start, clip[:, start : start + length]


def score_clip(model, clip: np.ndarray, cfg: dict, device: torch.device) -> list[dict]:
    sc = cfg["surprise"]
    dtype = next(model.parameters()).dtype
    x = to_pixels(clip, model.config.crop_size, device, dtype)
    rows = []
    for start, w in windows(x, sc["window"], sc["stride"]):
        steps = window_surprise(model, w, sc["seen"])
        rows.append({"start": start, "mode": "forecast", "mean": float(steps.mean()),
                     "steps": " ".join(f"{v:.6f}" for v in steps)})
    return rows


def mirror_rows(model, clip_ff: np.ndarray, cfg: dict, device: torch.device) -> list[dict]:
    """Content-fixed mirror test: each FF window vs the same window time-flipped."""
    sc = cfg["surprise"]
    dtype = next(model.parameters()).dtype
    x = to_pixels(clip_ff, model.config.crop_size, device, dtype)
    rows = []
    for start, w in windows(x, sc["window"], sc["stride"]):
        w_rev = torch.flip(w, dims=[1])
        for direction, ww in (("forward", w), ("reversed", w_rev)):
            for at_end in (False, True):
                s = window_surprise(model, ww, sc["seen"], seen_at_end=at_end)
                rows.append({"start": start, "mode": f"mirror_{direction}_{'end' if at_end else 'start'}",
                             "mean": float(s.mean()), "steps": " ".join(f"{v:.6f}" for v in s)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Score surprise for all stimulus versions")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    model = load_model(cfg["model"]["name"], device, cfg["model"]["dtype"])

    stim_dir = Path(cfg["stimuli"]["out_dir"])
    clip_ids = sorted({p.name.rsplit("_", 1)[0] for p in stim_dir.glob("*_FF.npy")})
    if args.limit:
        clip_ids = clip_ids[: args.limit]
    out = Path(cfg["surprise"]["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out.with_suffix(".config.yaml"))
    fields = ["clip_id", "version", "start", "mode", "mean", "steps"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in clip_ids:
            for v in cfg["surprise"]["versions"]:
                assert v in VERSIONS, v
                clip = np.load(stim_dir / f"{cid}_{v}.npy")
                for r in score_clip(model, clip, cfg, device):
                    w.writerow({"clip_id": cid, "version": v, **r})
            if cfg["surprise"]["mirror"]:
                for r in mirror_rows(model, np.load(stim_dir / f"{cid}_FF.npy"), cfg, device):
                    w.writerow({"clip_id": cid, "version": "FF", **r})
            print(f"scored {cid}")


if __name__ == "__main__":
    main()

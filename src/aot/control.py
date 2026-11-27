"""Positive control: fine-tune the V-JEPA 2 predictor on forecasting.

Encoders stay frozen. The predictor is trained with the surprise score itself
as the loss (see `seen` frames, predict the rest of the window) on forward-only,
reversed-only or mixed (50/50) footage. If the arrow-of-time score is sensitive,
a forward-trained predictor should find reversed clips more surprising.

lora_rank > 0 trains LoRA adapters on the predictor attention and MLP layers;
lora_rank = 0 fine-tunes the whole predictor.

Training clips: control.train_dir holds 33-frame uint8 .npy segments that never
overlap a test clip; control.folds_csv maps segment ids to half 0 or 1 (split
by source video). Train on one half, evaluate H1 on the other half's test clips.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model

from .config import load_config, resolve_device, save_config, seed_everything
from .surprise import load_model, surprise_steps, to_pixels

# Regex over module names: only predictor attention and MLP layers get adapters.
PREDICTOR_TARGETS = r"predictor\.layer\.\d+\.(attention\.(query|key|value|proj)|mlp\.(fc1|fc2))"


def attach_trainable_predictor(model, rank: int, alpha: int):
    for p in model.parameters():
        p.requires_grad_(False)
    if rank == 0:
        for p in model.predictor.parameters():
            p.requires_grad_(True)
        return model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=PREDICTOR_TARGETS)
    model = inject_adapter_in_model(cfg, model)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    return model


def sample_windows(clips: list[np.ndarray], window: int, direction: str, batch: int, rng: random.Random) -> list[np.ndarray]:
    out = []
    for _ in range(batch):
        c = rng.choice(clips)
        s = rng.randrange(0, len(c) - window + 1)
        w = c[s : s + window]
        flip = direction == "reversed" or (direction == "mixed" and rng.random() < 0.5)
        out.append(w[::-1].copy() if flip else w)
    return out


def load_half(train_dir: Path, folds_csv: Path, half: int) -> list[np.ndarray]:
    with open(folds_csv) as f:
        keep = {r["segment_id"] for r in csv.DictReader(f) if int(r["half"]) == half}
    return [np.load(p) for p in sorted(train_dir.glob("*.npy")) if p.stem in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the forecasting positive control")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    cc, sc = cfg["control"], cfg["surprise"]
    seed_everything(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = resolve_device(cfg["device"])

    model = load_model(cfg["model"]["name"], device, "float32")
    model = attach_trainable_predictor(model, cc["lora_rank"], cc["lora_alpha"])
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=cc["lr"], weight_decay=0.0)

    clips = load_half(Path(cc["train_dir"]), Path(cc["folds_csv"]), cc["half"])
    steps_per_epoch = max(1, len(clips) * 4 // cc["batch_size"])
    crop = model.config.crop_size
    model.train()
    for epoch in range(cc["epochs"]):
        total = 0.0
        for _ in range(steps_per_epoch):
            batch = sample_windows(clips, sc["window"], cc["direction"], cc["batch_size"], rng)
            x = torch.cat([to_pixels(w, crop, device, torch.float32) for w in batch])
            loss = surprise_steps(model, x, sc["seen"]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{cc['epochs']} loss {total / steps_per_epoch:.5f}")

    out = Path(cc["out_dir"]) / f"{cc['direction']}_r{cc['lora_rank']}_half{cc['half']}"
    out.mkdir(parents=True, exist_ok=True)
    trainable = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k or (cc["lora_rank"] == 0 and "predictor" in k)}
    torch.save(trainable, out / "predictor_weights.pt")
    save_config(cfg, out / "config.yaml")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

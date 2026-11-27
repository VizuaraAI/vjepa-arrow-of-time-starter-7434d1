"""Linear probe: is playback direction decodable from frozen encoder features?

Each FF window and its time-flipped copy become one positive and one negative
example. Cross-validation is grouped by clip so a clip never appears in both
training and test folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config, resolve_device, seed_everything
from .surprise import load_model, to_pixels, windows


@torch.no_grad()
def pooled_features(model, window: torch.Tensor, layer: str = "last") -> np.ndarray:
    out = model(pixel_values_videos=window, skip_predictor=True, output_hidden_states=True)
    if layer == "middle" and out.hidden_states is not None:
        h = out.hidden_states[len(out.hidden_states) // 2]
    else:
        h = out.last_hidden_state
    return h.float().mean(1).squeeze(0).cpu().numpy()


def grouped_probe_accuracy(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, C: float = 1.0) -> tuple[float, float]:
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(x[tr], y[tr])
        accs.append(clf.score(x[te], y[te]))
    return float(np.mean(accs) * 100), float(np.std(accs) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward vs reversed linear probe")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    model = load_model(cfg["model"]["name"], device, cfg["model"]["dtype"])
    dtype = next(model.parameters()).dtype

    stim_dir = Path(cfg["stimuli"]["out_dir"])
    feats, labels, groups = [], [], []
    for p in sorted(stim_dir.glob("*_FF.npy")):
        cid = p.name.rsplit("_", 1)[0]
        x = to_pixels(np.load(p), model.config.crop_size, device, dtype)
        for _, w in windows(x, cfg["surprise"]["window"], cfg["surprise"]["stride"]):
            for label, ww in ((1, w), (0, torch.flip(w, dims=[1]))):
                feats.append(pooled_features(model, ww, cfg["probe"]["layer"]))
                labels.append(label)
                groups.append(cid)
    acc, sd = grouped_probe_accuracy(np.stack(feats), np.array(labels), np.array(groups),
                                     cfg["probe"]["n_splits"], cfg["probe"]["C"])
    out = Path(cfg["probe"]["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": cfg["model"]["name"], "layer": cfg["probe"]["layer"],
                   "accuracy": acc, "sd": sd, "n_clips": len(set(groups))}]).to_csv(out, index=False)
    print(f"{cfg['model']['name']} [{cfg['probe']['layer']}]: {acc:.1f}% +/- {sd:.1f}")


if __name__ == "__main__":
    main()

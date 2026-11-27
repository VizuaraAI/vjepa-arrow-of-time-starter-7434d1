"""Build matched stimulus sets (FF, RR, FR, RF) from real video clips.

All four versions of a clip are built from the same decoded frames, so colour,
layout and motion magnitude are identical; only the playback order differs.
Clips are stored as uint8 arrays of shape (T, H, W, 3).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import av
import numpy as np
import requests

from .config import load_config

VERSIONS = ("FF", "RR", "FR", "RF")


def make_versions(frames: np.ndarray, flip_frame: int) -> dict[str, np.ndarray]:
    """Return the four playback orders of a clip of T frames.

    FR plays forward up to flip_frame, then retraces backwards.
    RF plays the reversed clip up to flip_frame, then retraces forwards.
    """
    t = len(frames)
    fwd = np.arange(t)
    rev = fwd[::-1]
    fr = np.concatenate([fwd[: flip_frame + 1], fwd[flip_frame - 1 :: -1]])[:t]
    rf = np.concatenate([rev[: flip_frame + 1], rev[flip_frame - 1 :: -1]])[:t]
    if len(fr) < t:  # pad by continuing the retrace from the start / end
        fr = np.concatenate([fr, np.full(t - len(fr), fr[-1])])
        rf = np.concatenate([rf, np.full(t - len(rf), rf[-1])])
    order = {"FF": fwd, "RR": rev, "FR": fr, "RF": rf}
    return {k: frames[idx] for k, idx in order.items()}


def frame_hashes(frames: np.ndarray) -> list[str]:
    return [hashlib.sha1(f.tobytes()).hexdigest()[:12] for f in frames]


def download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".mp4")
    if not path.exists():
        r = requests.get(url, timeout=60, headers={"User-Agent": "aot-research-starter"})
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def decode_clip(
    path: Path, start_s: float, n_frames: int, fps: float,
    crop: tuple[int, int, int] | None, size: int,
) -> np.ndarray:
    """Decode n_frames at a fixed fps from start_s, square crop, resize to size."""
    times = start_s + np.arange(n_frames) / fps
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        next_i = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < times[next_i]:
                continue
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            if crop is None:
                s = min(h, w)
                x, y = (w - s) // 2, (h - s) // 2
            else:
                x, y, s = crop
            img = img[y : y + s, x : x + s]
            img = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(width=size, height=size).to_ndarray(format="rgb24")
            while next_i < n_frames and frame.time + 1e-6 >= times[next_i]:
                out.append(img)
                next_i += 1
            if next_i >= n_frames:
                break
    if len(out) < n_frames:
        raise ValueError(f"{path}: only {len(out)} of {n_frames} frames from t={start_s}s")
    return np.stack(out)


def synthetic_clip(n_frames: int = 33, size: int = 64, seed: int = 0) -> np.ndarray:
    """An irreversible toy process (a blob that spreads and fades) for smoke tests."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = rng.uniform(0.3, 0.7, 2) * size
    frames = []
    for t in range(n_frames):
        sigma = 2 + 0.6 * t
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)) * (1 - t / (1.5 * n_frames))
        img = np.stack([blob, 0.6 * blob, 0.3 * blob], -1)
        frames.append((255 * np.clip(img, 0, 1)).astype(np.uint8))
    return np.stack(frames)


def read_sources(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_all(cfg: dict) -> None:
    sc = cfg["stimuli"]
    out_dir = Path(sc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in read_sources(Path(sc["sources_csv"])):
        crop = None
        if row.get("crop_size"):
            crop = (int(row["crop_x"]), int(row["crop_y"]), int(row["crop_size"]))
        frames = decode_clip(
            download(row["url"], Path(sc["cache_dir"])), float(row["start_s"]),
            sc["n_frames"], sc["fps"], crop, sc["size"],
        )
        for name, clip in make_versions(frames, sc["flip_frame"]).items():
            np.save(out_dir / f"{row['clip_id']}_{name}.npy", clip)
            manifest.append({"clip_id": row["clip_id"], "version": name,
                             "group": row.get("group", ""), "reversible": row.get("reversible", "0"),
                             "hashes": " ".join(frame_hashes(clip))})
        print(f"built {row['clip_id']}")
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FF/RR/FR/RF stimulus sets")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    build_all(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()

"""Turn results/surprise.csv into the H1, H2, H3 and Mirror-H1 rows of Table 1."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .stats import holm, one_sample_summary


def post_flip_mean(steps: str, n_skip: int) -> float:
    vals = np.array([float(v) for v in steps.split()])
    return float(vals[n_skip:].mean()) if len(vals) > n_skip else float("nan")


def clip_scores(df: pd.DataFrame) -> pd.DataFrame:
    fc = df[df["mode"] == "forecast"]
    per = fc.pivot_table(index="clip_id", columns="version", values="mean", aggfunc="mean")
    out = pd.DataFrame(index=per.index)
    if {"RR", "FF"} <= set(per.columns):
        out["H1"] = per["RR"] - per["FF"]
    if {"FR", "RR"} <= set(per.columns):
        out["H2"] = per["FR"] - per["RR"]
    if {"RF", "FF"} <= set(per.columns):
        out["H3"] = per["RF"] - per["FF"]
    mir = df[df["mode"].str.startswith("mirror_")]
    if len(mir):
        m = mir.pivot_table(index="clip_id", columns="mode", values="mean", aggfunc="mean")
        rev = (m["mirror_reversed_start"] + m["mirror_reversed_end"]) / 2
        fwd = (m["mirror_forward_start"] + m["mirror_forward_end"]) / 2
        out["MirrorH1"] = rev - fwd
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise surprise scores per clip")
    ap.add_argument("--surprise_csv", default="results/surprise.csv")
    ap.add_argument("--out_csv", default="results/table1.csv")
    args = ap.parse_args()
    scores = clip_scores(pd.read_csv(args.surprise_csv))
    rows = [{"test": c, **one_sample_summary(scores[c].dropna().to_numpy())} for c in scores.columns]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_t"].tolist())
    table.to_csv(args.out_csv, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()

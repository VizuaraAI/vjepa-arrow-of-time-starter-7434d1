#!/usr/bin/env bash
# Score surprise (plain H1/H2/H3 and mirror test) for one model, then summarise.
# Usage: scripts/run_surprise.sh [model=facebook/vjepa2-vith-fpc64-256] [extra overrides]
set -euo pipefail
export PYTHONPATH=src
MODEL="${1:-facebook/vjepa2-vitl-fpc64-256}"; shift || true
TAG="$(basename "$MODEL")"
python -m aot.surprise --config configs/default.yaml model.name="$MODEL" \
  surprise.out_csv="results/${TAG}/surprise.csv" "$@"
python -m aot.analyze --surprise_csv "results/${TAG}/surprise.csv" --out_csv "results/${TAG}/table1.csv"

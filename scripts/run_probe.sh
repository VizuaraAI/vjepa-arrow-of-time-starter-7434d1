#!/usr/bin/env bash
# Forward vs reversed linear probe on frozen features, last and middle layer.
set -euo pipefail
export PYTHONPATH=src
MODEL="${1:-facebook/vjepa2-vitl-fpc64-256}"
TAG="$(basename "$MODEL")"
for LAYER in last middle; do
  python -m aot.probes --config configs/default.yaml model.name="$MODEL" \
    probe.layer="$LAYER" probe.out_csv="results/${TAG}/probe_${LAYER}.csv"
done

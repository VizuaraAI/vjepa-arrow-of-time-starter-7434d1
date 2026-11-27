#!/usr/bin/env bash
# Positive control grid: three training directions x two halves (cross-fitting).
# Usage: scripts/train_control.sh [lora_rank=8] [epochs=2]   (rank 0 = full predictor fine-tune)
set -euo pipefail
export PYTHONPATH=src
RANK="${1:-8}"; EPOCHS="${2:-2}"
for DIR in forward reversed mixed; do
  for HALF in 0 1; do
    python -m aot.control --config configs/default.yaml control.direction="$DIR" \
      control.lora_rank="$RANK" control.epochs="$EPOCHS" control.half="$HALF"
  done
done

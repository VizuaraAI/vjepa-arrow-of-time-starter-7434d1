#!/usr/bin/env bash
# Download source clips listed in data/sources.csv and build FF/RR/FR/RF versions.
set -euo pipefail
export PYTHONPATH=src
python -m aot.stimuli --config configs/default.yaml "$@"

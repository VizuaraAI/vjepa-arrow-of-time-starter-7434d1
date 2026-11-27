#!/usr/bin/env bash
# Create a virtual environment and install dependencies.
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests

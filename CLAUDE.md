# CLAUDE.md: instructions for the coding agent

## Project goal

Research code for the paper "Does V-JEPA Have an Arrow of Time?" (author: Saurabh Tripathi, Vizuara Research). The question: does the prediction surprise of self-supervised video world models (V-JEPA, V-JEPA 2, compared with VideoMAE V2) prefer forward over reversed footage of irreversible real-world events? Earlier phases found a robust null in surprise (|d| <= 0.10 with content fixed) while frozen features decode direction at 63 to 70%. The current goal is to make the null interpretable: a stronger positive control, near-reversible control clips, causal targets for flip analyses, and human and supervised ceilings. Target venue: a workshop paper (4 pages) at ICLR 2027, with a fuller version later.

## Layout

- `configs/default.yaml`: single source of hyperparameters. Override with `section.key=value` on the CLI.
- `src/aot/stimuli.py`: builds FF/RR/FR/RF from identical decoded frames. Never change frame order logic without updating `tests/test_smoke.py::test_versions_share_frames`.
- `src/aot/surprise.py`: `surprise_steps` is the one core function. Context features come from encoding only the seen frames (no future leak; V-JEPA 2 uses RoPE so relative positions are preserved); targets come from encoding the full window.
- `src/aot/analyze.py`, `src/aot/stats.py`: per-clip scores and tests. n is the number of clips (292), never the number of windows.
- `src/aot/probes.py`: probes always use `GroupKFold` by clip id.
- `src/aot/control.py`: positive control. Encoders frozen; only predictor parameters (or LoRA adapters on them) train.
- `experiments/log_template.md`: copy a row per run.

## Conventions

- Python 3.10+, type hints, small pure functions, no notebooks in `src/`.
- Every result file has a sibling `.config.yaml`; never report a number without its config.
- Seeds fixed via `seed_everything`. Report Cohen's d with bootstrap 95% CI and Holm-corrected p.
- Do not commit videos, `.npy` stimuli, results or checkpoints (see `.gitignore`).
- Before any change to `surprise.py`, run `PYTHONPATH=src python -m pytest -q tests`.
- Pre-registered tests are listed in the paper draft; do not add post hoc tests to the Holm family without noting it in the log.

## Required sanity checks (re-run after any pipeline change)

1. Identical inputs give identical surprise (within 1e-3 in float16).
2. Mirror test on pixel baselines (`baselines.py`) is exactly 0.
3. Untrained network of the same architecture gives flat surprise across conditions.

## Next steps (in order)

1. Port `data/sources.csv` and `folds.csv` from the Phase 3 folder; build stimuli; reproduce the Phase 3 mirror-test numbers for V-JEPA 2-L (d about +0.04) with this code before running anything new. Check agreement with the original `jepa-intuitive-physics` surprise code on 10 clips.
2. Positive control: `scripts/train_control.sh 32 10` and `scripts/train_control.sh 0 10` (full predictor). Add an evaluation entry point that loads `predictor_weights.pt` and scores held-out clips.
3. Near-reversible clips (pendulums, ball arcs, spinning wheels): add rows with `reversible=1` to `sources.csv`; split all analyses by this flag.
4. Causal targets: add an option in `surprise_steps` that encodes frames 0..k for the target of step k, then recompute H2/H3.
5. Supervised R(2+1)D-18 ceiling with `torchvision.models.video.r2plus1d_18` (Kinetics weights) and a fixed training recipe (lower LR, longer warmup, check that loss leaves chance).
6. Frame-rate ablation at 4 fps.

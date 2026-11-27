# vjepa-arrow-of-time-starter

Starter code for **"Does V-JEPA Have an Arrow of Time? Prediction Surprise on Reversed and Direction-Flipped Real Videos"** by Saurabh Tripathi (Vizuara Research).

Saurabh, this repo is a clean, small re-implementation of the core of your Phase 1 to 3 pipeline, built around the Hugging Face V-JEPA 2 checkpoints so that the next experiments (stronger positive control, near-reversible clips, causal targets, human ceiling) can be run and logged in one place and mapped directly to the tables of the paper draft. It does not replace your `world-model-lab` code: use it as the paper-facing, reproducible version, and port your already-validated pieces into it where it helps.

Scope: this repo scores V-JEPA 2 (the checkpoints available in `transformers`). V-JEPA 1 and VideoMAE V2 rows of the paper come from your existing pipeline; keep their CSVs in the same format so `aot.analyze` can summarise them too.

## What it measures

- **Surprise**: L1 distance between the V-JEPA 2 predictor output and the layer-normalised target-encoder features of the predicted frames (Garrido et al. 2025), per 2-frame step.
- **H1 / H2 / H3**: RR minus FF, FR minus RR, RF minus FF per clip.
- **Mirror test**: each forward window against the same window time-flipped, with seen frames at the start (forecast) and at the end (masked beginning), so content is identical.
- **Linear probe**: forward vs reversed from frozen, mean-pooled features, 5-fold CV grouped by clip.
- **Positive control**: LoRA (or full) fine-tuning of the predictor on forecasting with forward-only, reversed-only or mixed data, encoders frozen.

## Setup

```bash
bash scripts/setup.sh          # creates .venv, installs requirements, runs CPU smoke tests
. .venv/bin/activate
```

A GPU with 24 GB is enough for ViT-L and ViT-H inference in float16. The smoke tests use a tiny random model and need no downloads.

## Data

Videos are not distributed (Pixabay licence). Put your clip list in `data/sources.csv` (see `data/sources_example.csv` for the columns: `clip_id,url,start_s,crop_x,crop_y,crop_size,group,reversible`). Your existing `sources.csv` from Phase 3 maps directly onto this format. Then:

```bash
bash scripts/build_stimuli.sh  # downloads clips, writes data/stimuli/<clip>_{FF,RR,FR,RF}.npy + manifest.csv with frame hashes
```

## Run

```bash
bash scripts/run_surprise.sh facebook/vjepa2-vitl-fpc64-256         # Table 1 rows, results/<model>/table1.csv
bash scripts/run_surprise.sh facebook/vjepa2-vitl-fpc64-256 surprise.window=32 surprise.seen=16
bash scripts/run_probe.sh facebook/vjepa2-vith-fpc64-256            # Table 2 rows
bash scripts/train_control.sh 32 10                                  # Table 3: LoRA rank 32, 10 epochs
```

Any config value can be overridden on the command line as `section.key=value`. Each run writes its resolved config next to its results.

## Layout

```
configs/default.yaml       all hyperparameters
src/aot/stimuli.py         download, decode, crop, build FF/RR/FR/RF, frame hashes
src/aot/surprise.py        leak-free surprise for V-JEPA 2, sliding windows, mirror test
src/aot/analyze.py         per-clip H1/H2/H3/Mirror-H1, Cohen's d, bootstrap CI, Holm
src/aot/probes.py          grouped-CV linear probe on frozen features
src/aot/control.py         positive control (LoRA or full predictor fine-tune)
src/aot/baselines.py       repeat-last-frame and constant-velocity pixel baselines
src/aot/stats.py           statistics helpers
experiments/log_template.md  one row per run, mapped to the paper tables
tests/test_smoke.py        CPU tests with a tiny random V-JEPA 2
```

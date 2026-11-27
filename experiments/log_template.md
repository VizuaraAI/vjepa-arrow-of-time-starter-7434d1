# Experiment log

One row per run. The `Paper cell` column says which table and row the run fills in the paper draft (`out/paper/main.tex`). Keep the config path so every number can be traced.

## Table 1: Arrow of time in prediction surprise

| Date | Run id | Test | Model | Window / seen | Clips (reversed more surprising, %) | Cohen's d [95% CI] | Holm p | Config | Paper cell | Notes |
|------|--------|------|-------|---------------|-------------------------------------|--------------------|--------|--------|------------|-------|
| | | Plain H1 | V-JEPA ViT-L | 16 / 8 | | | | | T1 row 1 | |
| | | Plain H1 | V-JEPA ViT-H | 16 / 8 | | | | | T1 row 2 | |
| | | Plain H1 | V-JEPA 2 ViT-L | 16 / 8 | | | | | T1 row 3 | |
| | | Plain H1 | V-JEPA 2 ViT-H | 16 / 8 | | | | | T1 row 4 | |
| | | Plain H1 | VideoMAE V2 ViT-g | 16 / 8 | | | | | T1 row 5 | |
| | | Mirror | V-JEPA 2 ViT-L | 16 / 8 | | | | | T1 row 6 | |
| | | Mirror | V-JEPA 2 ViT-H | 16 / 8 | | | | | T1 row 7 | |
| | | Mirror | V-JEPA 2 ViT-L | 32 / 16 | | | | | T1 row 8 | |
| | | Mirror | V-JEPA 2 ViT-H | 32 / 16 | | | | | T1 row 9 | |
| | | Mirror, near-reversible | V-JEPA 2 ViT-H | 16 / 8 | | | | | T1 row 10 | |

## Table 2: Direction decoding from frozen features

| Date | Run id | Representation | Clip set | Linear last (%) | Linear middle (%) | Attentive (%) | Config | Paper cell | Notes |
|------|--------|----------------|----------|-----------------|-------------------|---------------|--------|------------|-------|
| | | Pixel statistics | irreversible | | n/a | n/a | | T2 row 1 | |
| | | Untrained ViT-L | irreversible | | | | | T2 row 2 | |
| | | VideoMAE V2 ViT-g | irreversible | | | | | T2 row 3 | |
| | | V-JEPA ViT-L | irreversible | | | | | T2 row 4 | |
| | | V-JEPA 2 ViT-L | irreversible | | | | | T2 row 5 | |
| | | V-JEPA 2 ViT-H | irreversible | | | | | T2 row 6 | |
| | | Supervised R(2+1)D-18 | irreversible | | | | | T2 row 7 | |
| | | Humans (40 clips) | irreversible | | | | | T2 row 8 | |
| | | (repeat each row) | near-reversible | | | | | T2 right block | |

## Table 3: Positive control (V-JEPA 2-L)

| Date | Run id | Fine-tuning | Epochs | GPU-hours | Forward-only d | Reversed-only d | 50/50 d | Halves | Config | Paper cell | Notes |
|------|--------|-------------|--------|-----------|----------------|-----------------|---------|--------|--------|------------|-------|
| | | None (pretrained) | 0 | 0 | | | | n/a | | T3 row 1 | |
| | | LoRA r8 | 2 | | | | | 0 and 1 | | T3 row 2 | |
| | | LoRA r32 | 10 | | | | | 0 and 1 | | T3 row 3 | |
| | | Full predictor | 10 | | | | | 0 and 1 | | T3 row 4 | |

## Sanity checks (tick per pipeline version)

| Date | Code version | Identical inputs agree | Pixel mirror = 0 | Untrained flat | Notes |
|------|--------------|------------------------|------------------|----------------|-------|
| | | | | | |

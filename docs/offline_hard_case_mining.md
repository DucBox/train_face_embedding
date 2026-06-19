# Offline Hard-Case Mining — Run Commands

3-stage pipeline to find, at a target FMR, the images/identity-pairs a checkpoint
is currently getting wrong on the training set itself. See
[hard_negative_sampling.md](hard_negative_sampling.md) for the design rationale.
Each stage is a separate script - re-run any one without redoing the others as
long as its input files already exist.

Set `OUT` once and reuse it for all 3 stages so each step reads the previous
step's output automatically:

```bash
export OUT=/path/to/hard_case_out
```

## 1) Embed the dataset → Parquet

`--weight` defaults to `DEFAULT_WEIGHT` at the top of `embed_dataset.py`
(currently a placeholder `/path/to/model.pt` - edit that constant, or pass
`--weight` explicitly to override per-run without touching the file).

```bash
torchrun --nproc_per_node=4 embed_dataset.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
    --output-dir "$OUT"
```

Tune if needed:
- `--flush-rows` (default 500,000): lower it if host RAM is tight - each rank
  only ever holds this many images' embeddings in memory at once.
- `--batch-size` / `--num-workers`: inference batch size and DataLoader workers.

## 2) Find the FMR threshold + hard cases

```bash
python3 find_hard_thresholds.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
    --embeddings-dir "$OUT" --output-dir "$OUT" --fmr 1e-6
```

Tune if needed:
- `--topk` (default 50): nearest-centroid candidates kept per identity for the
  impostor search - raise it if the script warns the threshold is only a lower
  bound (not enough candidates collected to reach the target FMR).
- `--device`: defaults to `cuda` if available.

## 3) Generate the hard-case artifacts

```bash
python3 generate_hard_cases.py --input-dir "$OUT" --output-dir "$OUT"
```

## Output files in `$OUT` after all 3 stages

- `*.parquet` (partitioned by `file_prefix`) - the raw embeddings, from stage 1
- `false_accept.csv`, `false_reject.csv` - from stage 2, threshold + score included
- `hard_class_neighbors.csv` - per-class hard-negative neighbor list, from stage 3
- `hard_images_review.csv` - genuine images below threshold, worst-first, from stage 3

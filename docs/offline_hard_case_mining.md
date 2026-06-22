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

`config` and `--embeddings-dir` default to `DEFAULT_CONFIG` / `DEFAULT_EMBEDDINGS_DIR`
at the top of `find_hard_thresholds.py` - edit those constants, or override per-run.
`--output-dir` defaults to `--embeddings-dir` if not given.

```bash
python3 find_hard_thresholds.py --embeddings-dir "$OUT" --fmr 1e-6
```

Tune if needed:
- `--topk` (default 50) / `--centroid-chunk` (default 1024): see "What do topk
  and centroid-chunk do?" below.
- `--device`: defaults to `cuda` if available.

### What do `--topk` and `--centroid-chunk` do?

Both only affect Pass 2 (finding impostor identity pairs) - neither changes how
the deployed model is used (1:1 verification is unaffected; these only exist
because *calibrating* the FMR=1e-6 threshold requires looking across the whole
identity population, not just one pair).

- **`--centroid-chunk`**: pure performance/memory knob, no effect on the result.
  Comparing every identity's centroid against every other identity's centroid
  at once would need a `num_classes x num_classes` similarity matrix (infeasible
  at millions of identities). The script instead processes `--centroid-chunk`
  identities at a time against all centroids. Raise it if you have GPU memory
  to spare (fewer, bigger matmuls = faster); lower it if you hit OOM.

- **`--topk`**: affects correctness of the threshold estimate. For each identity,
  only the `--topk` nearest other identities are kept as impostor candidates
  (keeping all of them is what's infeasible in the first place). If the true
  pool of pairs needed to reach FMR=1e-6 is larger than what `--topk` collected,
  the script prints a warning and the threshold is only a *lower bound*. Raise
  `--topk` if you see that warning.

## 3) Generate the hard-case artifacts

```bash
python3 generate_hard_cases.py --input-dir "$OUT" --output-dir "$OUT"
```

## Output files in `$OUT` after all 3 stages

- `*.parquet` (partitioned by `file_prefix`) - the raw embeddings, from stage 1
- `false_accept.csv`, `false_reject.csv` - from stage 2, threshold + score included
- `hard_class_neighbors.csv` - per-class hard-negative neighbor list, from stage 3
- `hard_images_review.csv` - genuine images below threshold, worst-first, from stage 3

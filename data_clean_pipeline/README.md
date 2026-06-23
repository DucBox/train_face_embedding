# Dataset self-cleaning pipeline

Re-embed the whole dataset with a freshly trained model, clean it (per-id DBSCAN
→ symmetric IVF merge), and emit a cleaned dataset as **3 `.rec` files sharing one
contiguous global id range**. Retrain on it, repeat. Weight picking is manual.

Every knob lives in [`config.py`](config.py). Edit there; all stages read `CFG`.

## Run

```bash
# local logic test on tiny synthetic fixtures (no GPU/S3/mxnet/faiss needed)
TEST_PIPELINE=1 WORK_DIR=/tmp/clean_loop_test ./run_pipeline.sh

# real run (set paths/creds/model in config.py first)
TEST_PIPELINE=0 NPROC=8 ./run_pipeline.sh
```

`test_pipeline=True` swaps the embed + rec-write stages for synthetic fixtures /
manifest-assertions so the **whole control flow and every transform is verified
locally**. `False` runs the real cluster path.

## Flow (input → output per stage)

| # | Stage | Script | Input | Output |
|---|---|---|---|---|
| 01a | embed rec | `embed_rec.py` | webface/public `.rec` | `01_embed/{src}/*.parquet` `{person_id,src,img_key,embedding}` |
| 01b | embed S3 | `embed_s3.py` | crawl 197M crops on S3 | `01_embed/crawl/*.parquet` (per-id checkpoint) |
| 02 | normalize | `normalize.py` | 01 | `02_norm/{src}` `embedding_normalized` |
| 03 | dbscan | `dbscan.py` | 02 | `03_dbscan/{src}` (per-id intra-clean; drops outliers + ids < min_samples) |
| 04 | template | `template.py --src` | 03 | `04_template/{src}/centers.parquet` `{person_id,img_count,embedding_center}` |
| 05 | merge internal | `merge.py --src` | 04 | `05_merge_internal/{src}/{map.csv,drop.txt}` |
| 06 | template global | `template.py --global` | 03+05 | `06_template_global/centers.parquet` (+ dynamic crawl offset → `meta.json`) |
| 07 | merge global | `merge.py --global` | 06 | `07_merge_global/{map.csv,drop.txt}` (symmetric, no anchor) |
| 08 | reindex | `reindex.py` | 04+05+07 | `08_reindex/label_map.parquet` `{src,orig_id,final_id}` + `meta.json {num_classes,num_image}` |
| 09 | write rec | `write_rec.py` | 03+08 + bytes | `rec_out/train_synthetic_clean.rec`, `train_public_clean.rec`, `train_crawl_*.rec` |

`verify.py --stage X` runs after each stage and **halts the flow** on any failure
(e.g. `normalize` asserts `‖v‖≈1`, `reindex` asserts ids are contiguous `0..K-1`).

## Key design decisions (per discussion)

- **Symmetric global merge, no anchor**: leader = max image count (tie → smaller
  id); uncertain `(lower,upper]` pairs drop the weaker side — even if it's a
  webface id. (`merge_lower_thr`/`merge_upper_thr` in config.)
- **Crawl offset is dynamic** = `max(webface∪public effective id)+1`, not a
  hardcoded 3M. Public gets no offset (already contiguous after webface).
- **Synthetic** never enters embed/dbscan/merge; `write_rec` re-attaches it by
  remapping each synthetic image's parent webface id through `label_map`
  (dropped if the parent was dropped). Synthetic-only indices of
  `train_synthetic.rec` are those after the pure `train.rec` image count.
- **3 output files, one id range**: mirrors `docs/data_sources.md`, so the
  trainer loads them unchanged — only `num_classes`/`num_image` (in `meta.json`)
  change. After a cross-source merge a class may have images in two files; that
  is fine (trainer concatenates + groups by label).

## RAM safety / resume

- Embed shards parquet by id-range; every later stage streams **one shard at a
  time → flush → gc**. The only full-RAM object is the template table
  (~3.6M×512 f32 ≈ 7.5GB) in the merge stage.
- CPU stages (normalize/dbscan/template) use `CFG.cpu_workers` (32–64).
- Resume: embed checkpoints per-id (`_checkpoints/`); other stages skip output
  shards / files that already exist.

## Verify the synthetic-rec layout (one-off, on the cluster)

`write_rec.py` assumes `train_synthetic.rec` == pure `train.rec` + synthetic
appended at the end. Prove it AND count the synthetic images vs your ground truth:

```bash
python verify_synthetic_layout.py \
    --pure /data/train.rec --synthetic /data/train_synthetic.rec \
    --expected-synthetic 6000000
```

Prints record counts, a label+bytes prefix check, and `SYNTHETIC IMAGE COUNT`
(PASS/MISMATCH vs `--expected-synthetic`).

## Dependencies

`pip install faiss-cpu` works for the merge stage; `faiss-gpu` is faster at
3.6M-template scale (no bare `faiss` on PyPI — it's `faiss-cpu`/`faiss-gpu`).
Real run also needs `mxnet`, `boto3`, `torch`. Test mode needs none of these
(falls back to sklearn for range-search).

## After a run

Read `meta.json` → set `config.num_classes` and `config.num_image` in the
training config, point `config.rec` at `rec_out/`, then run `train_v2.py`.

# Training Data Sources

The training set (`config.rec` root dir) is 3 separate `.rec`/`.idx` pairs,
loaded and concatenated by `get_dataloader()` in [dataset.py](../dataset.py).
Identity (class) IDs are allocated as one global, contiguous range across all
3 sources — by construction there are no ID collisions between sources.

| Source | File prefix | Has header? | ID range (this run) | Count | Origin |
|---|---|---|---|---|---|
| webface | `train_synthetic.rec`/`.idx` | yes (`header.flag > 0`) | ~0 – 2.1M | ~2.1M ids | WebFace42M + per-id 3DDFA-generated synthetic images appended at the end of each id's image list |
| public | `train_public.rec`/`.idx` | yes (`header.flag > 0`) | ~2.1M – 2.6M | ~500K ids | public face dataset |
| crawl | `train_1.rec` … `train_{num_rec_files-1}.rec` | no (`header.flag <= 0`) | ~2.6M – 3.6M | ~1M ids | internally crawled data, split across `num_rec_files - 1` shards |

`config.num_classes` (e.g. `3666172` in
[wf42m_pfc03_40epoch_64gpu_vit_l.py](../configs/wf42m_pfc03_40epoch_64gpu_vit_l.py))
is the size of the PartialFC classifier weight matrix, i.e. an upper bound on
the global ID range above — it is not required to be a tight/dense count.

## Header vs no-header loading

`MXFaceDataset.__init__` ([dataset.py:254-262](../dataset.py)) branches on the
RecordIO header of index 0:

```python
if header.flag > 0:
    # webface / public: header.label = (num_images, ...) -> dense range
    self.imgidx = np.array(range(1, int(header.label[0])))
else:
    # crawl shards: no usable header -> must enumerate actual keys
    self.imgidx = np.array(list(self.imgrec.keys))
```

`webface` and `public` were built with a real RecordIO header (`flag > 0`), so
their image index is just `range(1, num_images)`. The `crawl` shards
(`train_1` … `train_N`) were not, so the dataset enumerates `imgrec.keys()`
directly to discover which indices exist. `embed_dataset.py`'s
`MXEvalDataset` mirrors this exact same branch for offline analysis/embedding,
so the two stay consistent.

Per-image labels are read from the record body itself (`header.label` in
`__getitem__`, [dataset.py:292](../dataset.py)), not from the file-level
header — the file-level header only tells you how many images / how to find
them.

## Known data issue: duplicate identities in `webface`

Some person in `webface` has been assigned 2 different global IDs (i.e. the
same identity appears twice under unrelated IDs, rather than being one
contiguous block of images under a single ID). Because IDs are allocated
contiguously across all 3 sources (webface → public → crawl, with no gaps by
design), naively deleting/merging an ID and re-numbering everything after it
would require re-indexing all of `public` and `crawl` too.

This does not need a re-index. The fix is a remap applied purely at the
data-loading layer, never touching the `.rec`/`.idx` binaries:

1. A small lookup CSV, `duplicate_id,canonical_id`, one row per duplicate ID
   to be merged away. (`false_accept.csv` from the
   [offline hard-case mining pipeline](offline_hard_case_mining.md) — very
   high-similarity "impostor" pairs — is a plausible source for discovering
   candidate duplicate-ID pairs.)
2. In `MXFaceDataset.__getitem__` (and the equivalent in `embed_dataset.py`'s
   `MXEvalDataset.__getitem__`), right after reading the raw label, remap it:
   `label = remap.get(label, label)`.
3. In PartialFC's per-step negative sampling (`partial_fc_v2.py`'s `sample()`),
   exclude the now-dead `duplicate_id`s from the random negative pool with a
   `dead_mask` buffer, so sampling budget (relevant when `sample_rate < 1`,
   i.e. hard-negative mining is on) is never wasted on IDs that no longer have
   any images pointing at them.

`config.num_classes` / the classifier weight matrix size is left untouched —
the now-unused duplicate IDs just become inert rows.

## CFP — pose-split DBSCAN test (not training data)

`cfp-dataset/` (gitignored, see [prepare_custom_dataset.md](prepare_custom_dataset.md)
for the convention) is the *Celebrities in Frontal-Profile* dataset, used only
to sanity-check whether DBSCAN(eps=0.3, min_samples, cosine) on this model's
embeddings incorrectly splits one identity's frontal and profile photos into
separate clusters. 500 identities × (10 frontal + 4 profile) images. Pipeline:

1. `align_cfp.py` — detect + align every raw CFP image to 112×112 via
   `yolov6n_face.onnx`, writes `cfp-dataset/processed_data/`.
2. `embed_cfpw.py` — embed the aligned crops, writes `cfp_embeddings.parquet`.
3. `run_dbscan_cfp.py` — DBSCAN + per-identity retention/pose-split stats +
   visualization grids.

This is a held-out diagnostic set, not used for training or the hard-case
mining pipeline above.

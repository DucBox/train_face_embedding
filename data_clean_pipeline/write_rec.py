"""
Stage 09 — WRITE REC.  Single responsibility: materialize the cleaned dataset as
THREE .rec/.idx pairs sharing ONE contiguous global id range, so the existing
trainer (get_dataloader concatenates sources) needs no change beyond num_classes.

The set of images to write = the post-DBSCAN images (stage 03 output) whose final
label != -1, with the label replaced by label_map (stage 08). Bytes come from:
  - webface/public/synthetic  -> source .rec via read_idx (img_key="prefix:ridx")
  - crawl                     -> S3 byte fetch (img_key=aligned_s3_path)

Outputs:
  train_synthetic_clean.rec  : webface-pure imgs + synthetic imgs (remapped)
  train_public_clean.rec     : public imgs
  train_crawl_{i}.rec        : crawl imgs, sharded

test_pipeline=True: no mxnet/boto3 — writes a MANIFEST parquet per output and
asserts id contiguity / coverage / counts against meta instead of packing bytes.
"""
from __future__ import annotations

import os

import polars as pl
from tqdm import tqdm

from common import (COL_ID, COL_SRC, COL_KEY, list_parquet, meta_read, log, die)
from config import CFG, SOURCES, WEBFACE, PUBLIC, CRAWL


def _label_maps():
    """{src: {orig_id -> final_id}} for surviving ids only."""
    lm = pl.read_parquet(os.path.join(CFG.dir_reindex(), "label_map.parquet"))
    out = {s: {} for s in SOURCES}
    for s, o, f in zip(lm["src"].to_list(), lm["orig_id"].to_list(), lm["final_id"].to_list()):
        if f >= 0:
            out[s][int(o)] = int(f)
    return out


def _iter_surviving(src, lmap):
    """Yield (final_id, img_key) for every surviving post-DBSCAN image of `src`."""
    for fp in list_parquet(CFG.dir_dbscan(src)):
        df = pl.read_parquet(fp, columns=[COL_ID, COL_KEY])
        for pid, key in zip(df[COL_ID].to_list(), df[COL_KEY].to_list()):
            f = lmap.get(int(pid))
            if f is not None:
                yield f, key


# --------------------------------------------------------------------------- #
# TEST MODE — manifests + assertions (no mxnet / boto3)
# --------------------------------------------------------------------------- #
def _run_test():
    meta = meta_read(CFG.path_meta())
    K = meta["num_classes"]
    maps = _label_maps()
    os.makedirs(CFG.rec_out_dir, exist_ok=True)

    plans = {
        "train_synthetic_clean": [(WEBFACE, maps[WEBFACE])],
        "train_public_clean": [(PUBLIC, maps[PUBLIC])],
        "train_crawl": [(CRAWL, maps[CRAWL])],
    }
    all_finals, total = set(), 0
    for name, parts in plans.items():
        rows_f, rows_k = [], []
        for src, lmap in parts:
            for f, key in _iter_surviving(src, lmap):
                rows_f.append(f); rows_k.append(key)
        # synthetic: simulate one synthetic image per surviving webface id
        if name == "train_synthetic_clean":
            for o, f in maps[WEBFACE].items():
                rows_f.append(f); rows_k.append(f"{CFG.synthetic_prefix}:{o}")
        pl.DataFrame({"final_id": rows_f, "img_key": rows_k}).write_parquet(
            os.path.join(CFG.rec_out_dir, f"{name}.manifest.parquet"))
        all_finals.update(rows_f); total += len(rows_f)
        log(f"[write_rec:test] {name}: {len(rows_f):,} imgs, "
            f"ids [{min(rows_f) if rows_f else '-'}..{max(rows_f) if rows_f else '-'}]")

    expect = set(range(K))
    if all_finals != expect:
        die(f"id coverage broken: |finals|={len(all_finals)} K={K} "
            f"missing={len(expect - all_finals)} extra={len(all_finals - expect)}")
    log(f"[write_rec:test] OK contiguous 0..{K - 1}, total imgs (incl synthetic)={total:,}")


# --------------------------------------------------------------------------- #
# REAL MODE — pack .rec via mxnet (+ S3 for crawl)
# --------------------------------------------------------------------------- #
def _run_real():
    import mxnet as mx

    maps = _label_maps()
    os.makedirs(CFG.rec_out_dir, exist_ok=True)
    rec_cache = {}

    def rec_bytes(img_key):
        prefix, ridx = img_key.rsplit(":", 1)
        if prefix not in rec_cache:
            rec_cache[prefix] = mx.recordio.MXIndexedRecordIO(
                os.path.join(CFG.rec_root, f"{prefix}.idx"),
                os.path.join(CFG.rec_root, f"{prefix}.rec"), "r")
        _, img = mx.recordio.unpack(rec_cache[prefix].read_idx(int(ridx)))
        return img

    # ---- train_synthetic_clean: webface-pure (from rec) + synthetic ----
    def write_rec_source(out_name, src):
        out = os.path.join(CFG.rec_out_dir, out_name)
        w = mx.recordio.MXIndexedRecordIO(out + ".idx", out + ".rec", "w")
        i = 0
        for final, key in tqdm(_iter_surviving(src, maps[src]),
                               desc=f"write {out_name}", unit="img"):
            hdr = mx.recordio.IRHeader(0, float(final), i, 0)
            w.write_idx(i, mx.recordio.pack(hdr, rec_bytes(key)))
            i += 1
        return w, i

    w_syn, i = write_rec_source("train_synthetic_clean", WEBFACE)
    # append synthetic-only indices of train_synthetic.rec, remapped by webface map
    syn = mx.recordio.MXIndexedRecordIO(
        os.path.join(CFG.rec_root, f"{CFG.synthetic_prefix}.idx"),
        os.path.join(CFG.rec_root, f"{CFG.synthetic_prefix}.rec"), "r")
    pure = mx.recordio.MXIndexedRecordIO(
        os.path.join(CFG.rec_root, f"{CFG.rec_prefix[WEBFACE][0]}.idx"),
        os.path.join(CFG.rec_root, f"{CFG.rec_prefix[WEBFACE][0]}.rec"), "r")
    # synthetic-only records = keys above the pure rec's max key (verified by
    # verify_synthetic_layout.py: train_synthetic == pure prefix + appended tail).
    max_pure = max(pure.keys)
    syn_keys = [k for k in syn.keys if k > max_pure]
    import numbers
    for k in tqdm(syn_keys, desc="write synthetic", unit="img"):
        h, img = mx.recordio.unpack(syn.read_idx(k))
        parent = int(h.label if isinstance(h.label, numbers.Number) else h.label[0])
        final = maps[WEBFACE].get(parent)
        if final is None:
            continue
        hdr = mx.recordio.IRHeader(0, float(final), i, 0)
        w_syn.write_idx(i, mx.recordio.pack(hdr, img)); i += 1
    w_syn.close()
    log(f"[write_rec] train_synthetic_clean: {i:,} imgs")

    # ---- public ----
    w_pub, n_pub = write_rec_source("train_public_clean", PUBLIC)
    w_pub.close()
    log(f"[write_rec] train_public_clean: {n_pub:,} imgs")

    # ---- crawl (S3), sharded by tar ----
    _write_crawl(maps[CRAWL])


def _write_crawl(lmap):
    import mxnet as mx
    import boto3

    om = pl.read_parquet(CFG.offset_map_path).with_columns(
        (pl.col("tar_path") + "/" + pl.col("member_name")).alias("img_key"))
    om = dict(zip(om["img_key"].to_list(),
                  zip(om["tar_path"].to_list(), om["start_byte"].to_list(),
                      om["length"].to_list())))
    s3 = boto3.client("s3", endpoint_url=CFG.s3_endpoint,
                      aws_access_key_id=CFG.s3_access_key,
                      aws_secret_access_key=CFG.s3_secret_key)
    log("[write_rec] scanning surviving crawl images...")
    rows = list(_iter_surviving(CRAWL, lmap))
    shards = [rows[i::CFG.crawl_out_shards] for i in range(CFG.crawl_out_shards)]
    for sid, srows in enumerate(tqdm(shards, desc="crawl shards", unit="shard")):
        out = os.path.join(CFG.rec_out_dir, f"train_crawl_{sid:03d}")
        w = mx.recordio.MXIndexedRecordIO(out + ".idx", out + ".rec", "w")
        for i, (final, key) in enumerate(
                tqdm(srows, desc=f"  shard {sid:03d}", unit="img", leave=False)):
            meta = om.get(key)
            if meta is None:
                continue
            tar, start, length = meta
            body = s3.get_object(Bucket=CFG.s3_bucket, Key=tar,
                                 Range=f"bytes={start}-{start + length - 1}")["Body"].read()
            hdr = mx.recordio.IRHeader(0, float(final), i, 0)
            w.write_idx(i, mx.recordio.pack(hdr, body))
        w.close()
    log(f"[write_rec] crawl: {len(rows):,} imgs over {CFG.crawl_out_shards} shards")


def run():
    _run_test() if CFG.test_pipeline else _run_real()


if __name__ == "__main__":
    run()

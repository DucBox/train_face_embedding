"""
Stage 04 / 06 — TEMPLATE (centroid per id).  Single responsibility: aggregate
L2-normalized image embeddings into one centroid per id (sum -> L2-normalize).

Two modes:
  --src <name>   per-source template on the post-DBSCAN images (native ids).
  --global       cross-source template on post-INTERNAL-MERGE ids: applies each
                 source's internal merge map + drop, adds the dynamic crawl
                 offset, and writes the combined centers + the offset into meta.

RAM note: holds running float64 sum vectors keyed by id (~id_count x 512 x 8B;
~15GB at 3.6M ids — fits in the 300GB budget). Images are streamed shard-by-shard.

Output centers.parquet: person_id (Int64), img_count (Int64), embedding_center (List f32[512]).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

from common import (COL_ID, COL_EMBN, list_parquet, emb_matrix, log,
                    read_merge_artifacts, meta_write)
from config import CFG, SOURCES, WEBFACE, PUBLIC, CRAWL


def _aggregate(in_dir, id_map=None, drop=None, offset=0, sums=None, counts=None):
    """Stream shards in in_dir, fold into sums/counts dicts keyed by EFFECTIVE id."""
    sums = sums if sums is not None else {}
    counts = counts if counts is not None else defaultdict(int)
    drop = drop or set()
    for fp in tqdm(list_parquet(in_dir), desc=f"template:{os.path.basename(in_dir)}"):
        df = pl.read_parquet(fp, columns=[COL_ID, COL_EMBN])
        if df.height == 0:
            continue
        ids = df[COL_ID].to_numpy()
        m = emb_matrix(df, COL_EMBN)
        for pid, vec in zip(ids, m):
            pid = int(pid)
            if pid in drop:
                continue
            eff = id_map.get(pid, pid) if id_map else pid
            eff += offset
            if eff in sums:
                sums[eff] += vec
            else:
                sums[eff] = vec.astype(np.float64).copy()
            counts[eff] += 1
    return sums, counts


def _write_centers(sums, counts, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ids = sorted(sums.keys())
    if not ids:
        pl.DataFrame({"person_id": [], "img_count": [], "embedding_center": []},
                     schema={"person_id": pl.Int64, "img_count": pl.Int64,
                             "embedding_center": pl.List(pl.Float32)}).write_parquet(out_path)
        return 0
    mat = np.stack([sums[i] for i in ids])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    pl.DataFrame({
        "person_id": ids,
        "img_count": [int(counts[i]) for i in ids],
        "embedding_center": [r for r in mat.astype(np.float32)],
    }).write_parquet(out_path)
    return len(ids)


def run_src(src: str):
    sums, counts = _aggregate(CFG.dir_dbscan(src))
    n = _write_centers(sums, counts, os.path.join(CFG.dir_template(src), "centers.parquet"))
    log(f"[template:{src}] {n:,} ids")


def run_global():
    sums, counts = {}, defaultdict(int)
    # webface + public keep native ids; compute their max effective id for offset.
    for src in (WEBFACE, PUBLIC):
        mmap, drop = read_merge_artifacts(CFG.dir_merge_int(src))
        sums, counts = _aggregate(CFG.dir_dbscan(src), mmap, drop, 0, sums, counts)
    max_anchor = max(sums.keys()) if sums else 0
    offset = max(max_anchor + 1, CFG.crawl_offset_floor)
    log(f"[template:global] crawl offset = {offset:,} (max webface/public id = {max_anchor:,})")
    # crawl gets the dynamic offset on top of its internal-merge effective id.
    cmap, cdrop = read_merge_artifacts(CFG.dir_merge_int(CRAWL))
    sums, counts = _aggregate(CFG.dir_dbscan(CRAWL), cmap, cdrop, offset, sums, counts)
    n = _write_centers(sums, counts,
                       os.path.join(CFG.dir_template_global(), "centers.parquet"))
    meta_write(CFG.path_meta(), crawl_offset=offset, global_template_ids=n)
    log(f"[template:global] {n:,} ids total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", choices=SOURCES)
    g.add_argument("--global", dest="globalmode", action="store_true")
    a = ap.parse_args()
    run_global() if a.globalmode else run_src(a.src)

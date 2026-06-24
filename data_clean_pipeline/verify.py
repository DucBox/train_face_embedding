"""
VERIFY — per-stage correctness checks. Single responsibility: assert a stage's
output is complete and well-formed BEFORE the next stage runs. Used after every
stage by run_pipeline.sh (essential in test mode, cheap sanity in real mode).

    python verify.py --stage normalize --src webface
    python verify.py --stage reindex

Each check raises SystemExit(non-zero) on failure so the .sh halts the flow.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl
from tqdm import tqdm

from common import (COL_ID, COL_EMBN, list_parquet, emb_matrix, meta_read,
                    read_merge_artifacts, log, die)
from config import CFG, SOURCES, CRAWL


def _ok(msg): log(f"[verify] OK  {msg}")


def v_normalize(src):
    ins, outs = list_parquet(CFG.dir_norm(src)), CFG.dir_norm(src)
    if not ins:
        die(f"normalize:{src} produced no shards")
    worst = 0.0
    for fp in tqdm(ins, desc=f"verify normalize:{src}", unit="shard"):
        df = pl.read_parquet(fp)
        if df.height == 0:
            continue
        norms = np.linalg.norm(emb_matrix(df, COL_EMBN), axis=1)
        worst = max(worst, float(np.abs(norms - 1.0).max()))
    # 1e-3 tolerance: embeddings are stored as float16 (rounding gives ‖v‖ dev
    # ~1e-4), while a real bug (unnormalized / BGR / wrong dim) gives ~O(0.1+).
    if worst > 1e-3:
        die(f"normalize:{src} ‖v‖ deviates from 1.0 by {worst:.2e}")
    _ok(f"normalize:{src} ‖v‖≈1 (max dev {worst:.2e}, fp16 tol 1e-3)")


def v_dbscan(src):
    in_ids, out_ids = set(), set()
    for fp in tqdm(list_parquet(CFG.dir_norm(src)), desc=f"verify dbscan in:{src}", unit="shard"):
        in_ids.update(pl.read_parquet(fp, columns=[COL_ID])[COL_ID].to_list())
    bad = 0
    for fp in tqdm(list_parquet(CFG.dir_dbscan(src)), desc=f"verify dbscan out:{src}", unit="shard"):
        df = pl.read_parquet(fp, columns=[COL_ID])
        out_ids.update(df[COL_ID].to_list())
        vc = df[COL_ID].value_counts()
        bad += vc.filter(pl.col("count") < CFG.dbscan_min_samples).height
    if not out_ids <= in_ids:
        die(f"dbscan:{src} invented ids not present in input")
    if bad:
        die(f"dbscan:{src} left {bad} ids below min_samples")
    _ok(f"dbscan:{src} ids {len(out_ids):,}⊆{len(in_ids):,}, none < min_samples")


def v_template(src):
    df = pl.read_parquet(os.path.join(CFG.dir_template(src), "centers.parquet"))
    if df["person_id"].n_unique() != df.height:
        die(f"template:{src} has duplicate ids")
    if df.height:
        norms = np.linalg.norm(emb_matrix(df, "embedding_center"), axis=1)
        if np.abs(norms - 1.0).max() > 1e-4:
            die(f"template:{src} centroids not unit-norm")
    _ok(f"template:{src} {df.height:,} unit-norm centroids, unique ids")


def v_merge(src=None):
    d = CFG.dir_merge_global() if src is None else CFG.dir_merge_int(src)
    mmap, drop = read_merge_artifacts(d)
    # a merged id must not point at a dropped id, and leaders must not be dropped
    for og, leader in mmap.items():
        if leader in drop:
            die(f"merge{'/'+src if src else '/global'}: {og} merged into dropped {leader}")
    _ok(f"merge{'/'+src if src else '/global'}: merged={len(mmap):,} dropped={len(drop):,}, consistent")


def v_reindex():
    meta = meta_read(CFG.path_meta())
    K = meta["num_classes"]
    lm = pl.read_parquet(os.path.join(CFG.dir_reindex(), "label_map.parquet"))
    finals = set(lm.filter(pl.col("final_id") >= 0)["final_id"].to_list())
    if finals != set(range(K)):
        die(f"reindex: final ids not contiguous 0..{K-1} (got {len(finals)} distinct)")
    _ok(f"reindex: contiguous 0..{K-1}, num_image(real)={meta['num_image']:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["normalize", "dbscan", "template", "merge", "reindex"])
    ap.add_argument("--src", choices=SOURCES)
    ap.add_argument("--global", dest="g", action="store_true")
    a = ap.parse_args()
    if a.stage == "normalize": v_normalize(a.src)
    elif a.stage == "dbscan": v_dbscan(a.src)
    elif a.stage == "template": v_template(a.src)
    elif a.stage == "merge": v_merge(None if a.g else a.src)
    elif a.stage == "reindex": v_reindex()


if __name__ == "__main__":
    main()

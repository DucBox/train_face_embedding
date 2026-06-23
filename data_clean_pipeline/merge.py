"""
Stage 05 / 07 — MERGE (IVF range-search + symmetric graph merge).  Single
responsibility: given a centers.parquet (one centroid per id), decide which ids
to merge (> upper_thr) and which to drop (lower_thr, upper_thr]. SYMMETRIC — no
source is an anchor (per user). Writes map.csv (og_id->new_id) + drop.txt.

  --src <name>   internal merge of one source   (centers from template.py --src)
  --global       merge across all 3 sources      (centers from template.py --global)

Only the centers table is held in RAM (~3.6M x 512 f32 ~= 7.5GB at full scale).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from common import symmetric_merge, write_merge_artifacts, emb_matrix, log
from config import CFG, SOURCES


def _merge(centers_path: str, out_dir: str, tag: str):
    df = pl.read_parquet(centers_path)
    if df.height == 0:
        write_merge_artifacts(out_dir, {}, set())
        log(f"[merge:{tag}] empty centers")
        return
    ids = df["person_id"].to_numpy().astype(np.int64)
    counts = df["img_count"].to_numpy().astype(np.int64)
    vecs = emb_matrix(df, "embedding_center")
    merge_map, drop_ids = symmetric_merge(
        ids, counts, vecs,
        CFG.merge_lower_thr, CFG.merge_upper_thr, CFG.ivf_n_clusters, CFG.ivf_nprobe)
    write_merge_artifacts(out_dir, merge_map, drop_ids)
    log(f"[merge:{tag}] ids={len(ids):,} merged={len(merge_map):,} dropped={len(drop_ids):,}")


def run_src(src):
    _merge(os.path.join(CFG.dir_template(src), "centers.parquet"),
           CFG.dir_merge_int(src), src)


def run_global():
    _merge(os.path.join(CFG.dir_template_global(), "centers.parquet"),
           CFG.dir_merge_global(), "global")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", choices=SOURCES)
    g.add_argument("--global", dest="globalmode", action="store_true")
    a = ap.parse_args()
    run_global() if a.globalmode else run_src(a.src)

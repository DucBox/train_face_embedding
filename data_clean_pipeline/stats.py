"""
STATS — consolidated per-source report for the clean stages. Reads whatever
stage outputs exist for a source and prints image/id counts + norm check +
merge drop/merge numbers, so you can eyeball one source's whole flow.

    python stats.py --src crawl
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import polars as pl
from tqdm import tqdm

from common import (COL_ID, COL_EMBN, list_parquet, emb_matrix,
                    read_merge_artifacts)
from config import CFG, SOURCES


def _imgs(d):
    return sum(pl.read_parquet(fp, columns=[COL_ID]).height
               for fp in list_parquet(d)) if os.path.isdir(d) else None


def _ids(d):
    s = set()
    for fp in list_parquet(d):
        s.update(pl.read_parquet(fp, columns=[COL_ID])[COL_ID].to_list())
    return len(s)


def _imgs_per_id(d):
    c = Counter()
    for fp in list_parquet(d):
        vc = pl.read_parquet(fp, columns=[COL_ID])[COL_ID].value_counts()
        for pid, n in zip(vc[COL_ID].to_list(), vc["count"].to_list()):
            c[int(pid)] += int(n)
    return c


def _norm_dev(d):
    worst, sample = 0.0, []
    for fp in tqdm(list_parquet(d), desc="norm check", unit="shard", leave=False):
        m = emb_matrix(pl.read_parquet(fp, columns=[COL_EMBN]), COL_EMBN)
        if len(m) == 0:
            continue
        norms = np.linalg.norm(m, axis=1)
        worst = max(worst, float(np.abs(norms - 1.0).max()))
        if len(sample) < 3:
            sample.append(float(norms[0]))
    return worst, sample


def run(src):
    line = "=" * 64
    print(f"\n{line}\nSTATS — source: {src}\n{line}")

    d_embed, d_norm, d_db = CFG.dir_embed(src), CFG.dir_norm(src), CFG.dir_dbscan(src)

    # 1) EMBED
    n_embed = _imgs(d_embed)
    print(f"[embed]      images = {n_embed:,}" if n_embed is not None else "[embed]      (none)")

    # 2) NORMALIZE
    if os.path.isdir(d_norm) and list_parquet(d_norm):
        n_norm = _imgs(d_norm)
        worst, sample = _norm_dev(d_norm)
        print(f"[normalize]  images = {n_norm:,}")
        print(f"[normalize]  ‖v‖ max|dev from 1.0| = {worst:.2e}  "
              f"(samples {[f'{s:.7f}' for s in sample]})  -> {'OK ~1.0' if worst < 1e-4 else 'BAD'}")

    # 3) DBSCAN (before vs after)
    if os.path.isdir(d_db) and list_parquet(d_db):
        before_i = _imgs(d_norm)
        after_i = _imgs(d_db)
        before_id = _ids(d_norm)
        after_id = _ids(d_db)
        di, dd = before_i - after_i, before_id - after_id
        print(f"[dbscan]     images : {before_i:,} -> {after_i:,}  "
              f"(removed {di:,}, {100*di/max(before_i,1):.2f}% outliers)")
        print(f"[dbscan]     ids    : {before_id:,} -> {after_id:,}  "
              f"(removed {dd:,} ids with <min_samples / no cluster)")

    # 4) MERGE (internal)
    md = CFG.dir_merge_int(src)
    if os.path.exists(os.path.join(md, "map.csv")):
        mmap, drop = read_merge_artifacts(md)
        per_id = _imgs_per_id(d_db)
        before_i = sum(per_id.values())
        dropped_imgs = sum(per_id.get(i, 0) for i in drop)
        after_i = before_i - dropped_imgs
        before_id = len(per_id)
        # ids after merge = ids not dropped and not merged-away (merged keep leader id)
        after_id = len({mmap.get(i, i) for i in per_id if i not in drop})
        print(f"[merge]      merged {len(mmap):,} ids away, dropped {len(drop):,} ids")
        print(f"[merge]      images : {before_i:,} -> {after_i:,}  "
              f"(dropped {dropped_imgs:,} imgs from dropped ids; merged ids keep their imgs)")
        print(f"[merge]      ids    : {before_id:,} -> {after_id:,}")
    print(line + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=SOURCES, required=True)
    run(ap.parse_args().src)

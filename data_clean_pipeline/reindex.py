"""
Stage 08 — REINDEX (compose all maps -> final contiguous labels).  Single
responsibility: fold the three remapping steps into ONE lookup table that the
writer applies to both real images and synthetic.

Chain per source, for each native (post-DBSCAN) id `o`:
    drop if o in internal-drop(src)
    e1 = internal_map(src)[o]                 (+crawl offset for crawl)
    drop if e1 in global-drop
    e2 = global_map[e1]                       (symmetric leader)
    final = contiguous_index[e2]              (0..K-1, sorted ascending)

Output:
    label_map.parquet : src, orig_id, final_id   (final_id == -1 means DROP)
    meta.json         : num_classes (K), num_image (real, excl. synthetic),
                        id range per source block
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from tqdm import tqdm

from common import read_merge_artifacts, meta_write, meta_read, log
from config import CFG, SOURCES, WEBFACE, PUBLIC, CRAWL


def run():
    meta = meta_read(CFG.path_meta())
    offset = int(meta["crawl_offset"])
    gmap, gdrop = read_merge_artifacts(CFG.dir_merge_global())

    # surviving leader ids in e1-space (post-internal+offset, post-global)
    gcenters = pl.read_parquet(
        os.path.join(CFG.dir_template_global(), "centers.parquet"))["person_id"].to_list()
    surviving = sorted({gmap.get(x, x) for x in gcenters if x not in gdrop})
    contiguous = {e2: i for i, e2 in enumerate(surviving)}
    K = len(surviving)
    log(f"[reindex] surviving classes K = {K:,}")

    rows_src, rows_orig, rows_final = [], [], []
    block_ranges = {}
    num_image = 0
    for src in SOURCES:
        imap, idrop = read_merge_artifacts(CFG.dir_merge_int(src))
        off = offset if src == CRAWL else 0
        cpath = os.path.join(CFG.dir_template(src), "centers.parquet")
        cdf = pl.read_parquet(cpath)
        native_ids = cdf["person_id"].to_list()
        counts = dict(zip(native_ids, cdf["img_count"].to_list()))
        finals = []
        for o in tqdm(native_ids, desc=f"reindex:{src}", unit="id"):
            if o in idrop:
                final = -1
            else:
                e1 = imap.get(o, o) + off
                if e1 in gdrop:
                    final = -1
                else:
                    final = contiguous.get(gmap.get(e1, e1), -1)
            rows_src.append(src); rows_orig.append(int(o)); rows_final.append(int(final))
            if final >= 0:
                finals.append(final)
                num_image += int(counts[o])
        if finals:
            block_ranges[src] = [min(finals), max(finals)]
        log(f"[reindex] {src}: {len(native_ids):,} native ids, "
            f"{sum(1 for f in finals):,} kept")

    os.makedirs(CFG.dir_reindex(), exist_ok=True)
    pl.DataFrame({"src": rows_src, "orig_id": rows_orig, "final_id": rows_final}) \
        .write_parquet(os.path.join(CFG.dir_reindex(), "label_map.parquet"))
    meta_write(CFG.path_meta(), num_classes=K, num_image=num_image,
               block_ranges=block_ranges)
    log(f"[reindex] num_classes={K:,} num_image(real)={num_image:,} "
        f"blocks={block_ranges} -> {CFG.dir_reindex()}")


if __name__ == "__main__":
    run()

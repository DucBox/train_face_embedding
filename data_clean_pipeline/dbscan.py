"""
Stage 03 — DBSCAN per-id intra-clean.  Single responsibility: within EACH
person_id, DBSCAN(eps, min_samples, cosine) and keep only the largest cluster;
drop outlier images, and drop the whole id if it has < min_samples images or no
valid cluster. eps is per-source (CFG.dbscan_eps[src]).

RAM-safe: processes one shard file at a time; within a shard, partitions by
person_id and farms ids to workers. Resume = skip existing output shard files.

    python dbscan.py --src crawl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from common import COL_ID, COL_EMBN, list_parquet, emb_matrix, log
from config import CFG, SOURCES

_EPS = None
_MIN = None


def _init(eps, min_samples):
    global _EPS, _MIN
    _EPS, _MIN = eps, min_samples
    # keep BLAS single-threaded inside workers (we parallelize across ids)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _clean_id(df: pl.DataFrame):
    if df.height < _MIN:
        return None
    X = emb_matrix(df, COL_EMBN)
    labels = DBSCAN(eps=_EPS, min_samples=_MIN, metric="cosine", n_jobs=1).fit_predict(X)
    valid = np.where(labels != -1)[0]
    if len(valid) == 0:
        return None
    uniq, cnt = np.unique(labels[valid], return_counts=True)
    best = uniq[np.argmax(cnt)]
    return df.with_columns(pl.Series("_c", labels)).filter(pl.col("_c") == best).drop("_c")


def run(src: str):
    in_dir, out_dir = CFG.dir_norm(src), CFG.dir_dbscan(src)
    os.makedirs(out_dir, exist_ok=True)
    eps, mins = CFG.dbscan_eps[src], CFG.dbscan_min_samples
    workers = max(1, CFG.cpu_workers)
    kept = dropped = 0
    ctx = mp.get_context("spawn")
    for fp in tqdm(list_parquet(in_dir), desc=f"dbscan:{src}"):
        out_fp = os.path.join(out_dir, os.path.basename(fp))
        if os.path.exists(out_fp):
            continue
        df = pl.read_parquet(fp)
        if df.height == 0:
            df.write_parquet(out_fp)
            continue
        groups = df.partition_by(COL_ID, maintain_order=False)
        results = []
        with ctx.Pool(min(workers, len(groups)), initializer=_init,
                      initargs=(eps, mins)) as pool:
            for res in pool.imap_unordered(_clean_id, groups, chunksize=64):
                if res is not None:
                    results.append(res)
        n_in = len(groups)
        kept += len(results)
        dropped += n_in - len(results)
        out = pl.concat(results) if results else df.head(0)
        out.write_parquet(out_fp)
    log(f"[dbscan:{src}] ids kept={kept:,} dropped(<min/no-cluster)={dropped:,} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=SOURCES, required=True)
    run(ap.parse_args().src)

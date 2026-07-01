"""
Stage 02 — NORMALIZE.  Single responsibility: L2-normalize the raw `embedding`
column into `embedding_normalized`. Per-file, parallel, RAM-safe (one shard in
flight per worker). Resume = skip output files that already exist.

    python normalize.py --src webface
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import polars as pl
from tqdm import tqdm

from common import COL_EMB, COL_EMBN, list_parquet, emb_matrix, log
from config import CFG, SOURCES


def _norm_one(args):
    in_fp, out_fp = args
    if os.path.exists(out_fp):
        return 0
    df = pl.read_parquet(in_fp)
    if df.height == 0:
        df.write_parquet(out_fp)
        return 0
    m = emb_matrix(df, COL_EMB)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    df = df.drop(COL_EMB).with_columns(
        pl.Series(COL_EMBN, [r for r in m.astype(np.float32)]))
    df.write_parquet(out_fp)
    return df.height


def run(src: str):
    in_dir, out_dir = CFG.dir_embed(src), CFG.dir_norm(src)
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(fp, os.path.join(out_dir, os.path.basename(fp))) for fp in list_parquet(in_dir)]
    if not tasks:
        log(f"[normalize:{src}] no input shards in {in_dir}")
        return
    workers = min(CFG.cpu_workers, len(tasks))
    total = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for n in tqdm(pool.imap_unordered(_norm_one, tasks), total=len(tasks),
                      desc=f"normalize:{src}"):
            total += n
    log(f"[normalize:{src}] {total:,} imgs -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=SOURCES, required=True)
    run(ap.parse_args().src)

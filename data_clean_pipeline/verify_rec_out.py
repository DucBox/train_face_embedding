"""
Stage 10 — VERIFY REC OUT.  Single responsibility: sanity-check the FINAL
rec_out/ files actually written by write_rec.py (not the intermediate stages):
  - total image count, per-output-file breakdown
  - id coverage: every id in [0, num_classes) appears at least once; no label
    falls outside that range (would mean a remap bug)
  - per-id image-count distribution (min/max/mean/median/percentiles),
    overall and per source block (webface / public / crawl)

Two modes, same convention as every other stage:
  test_pipeline=True  -> reads the *.manifest.parquet files _run_test() wrote
  test_pipeline=False -> reads the real .rec files written by mxnet, via a
                          plain SEQUENTIAL forward scan (MXRecordIO.read(),
                          not read_idx) so nothing needs the .idx and no full
                          file is ever loaded into RAM. One worker per file,
                          pooled across CFG.cpu_workers, so the ~95M-image
                          crawl shards finish in parallel.

Run AFTER write_rec.py has produced every train_*_clean.rec / train_crawl_*.rec
file. Writes a JSON report to WORK_DIR/verify_rec_out_report.json and prints a
PASS/FAIL line (PASS = no missing ids, no out-of-range labels).
"""
from __future__ import annotations

import glob
import json
import numbers
import os
from multiprocessing import get_context

import numpy as np
import polars as pl
from tqdm import tqdm

from common import log, die, meta_read
from config import CFG, SOURCES


def _count_manifest(args):
    path, K = args
    df = pl.read_parquet(path, columns=["final_id"])
    ids = df["final_id"].to_numpy()
    total = len(ids)
    in_range = (ids >= 0) & (ids < K)
    oor = int((~in_range).sum())
    counts = np.bincount(ids[in_range].astype(np.int64), minlength=K).astype(np.int64)
    return os.path.basename(path), total, counts, oor


def _count_rec(args):
    """Sequential forward scan of one .rec file (no .idx / random access needed)."""
    path, K = args
    import mxnet as mx
    rec = mx.recordio.MXRecordIO(path, "r")
    counts = np.zeros(K, dtype=np.int64)
    total = 0
    oor = 0
    while True:
        item = rec.read()
        if item is None:
            break
        header, _ = mx.recordio.unpack(item)
        lab = int(header.label if isinstance(header.label, numbers.Number) else header.label[0])
        total += 1
        if 0 <= lab < K:
            counts[lab] += 1
        else:
            oor += 1
    rec.close()
    return os.path.basename(path), total, counts, oor


def _percentiles(counts_present):
    p = np.percentile(counts_present, [1, 25, 50, 75, 99])
    return {"p1": float(p[0]), "p25": float(p[1]), "median": float(p[2]),
            "p75": float(p[3]), "p99": float(p[4])}


def run():
    meta = meta_read(CFG.path_meta())
    if "num_classes" not in meta:
        die("meta.json has no num_classes -- run reindex.py first")
    K = int(meta["num_classes"])
    expected_real = int(meta.get("num_image", -1))
    block_ranges = meta.get("block_ranges", {})

    if CFG.test_pipeline:
        files = sorted(glob.glob(os.path.join(CFG.rec_out_dir, "*.manifest.parquet")))
        worker = _count_manifest
    else:
        files = sorted(glob.glob(os.path.join(CFG.rec_out_dir, "*.rec")))
        worker = _count_rec
    if not files:
        die(f"no output files found in {CFG.rec_out_dir}")
    jobs = [(f, K) for f in files]
    log(f"[verify_rec_out] scanning {len(files)} files (K={K:,} classes)...")

    counts = np.zeros(K, dtype=np.int64)
    total_images = 0
    total_oor = 0
    per_file = {}
    nproc = max(1, min(CFG.cpu_workers, len(jobs)))
    ctx = get_context("spawn")
    with ctx.Pool(nproc) as pool:
        for name, total, c, oor in tqdm(pool.imap_unordered(worker, jobs),
                                        total=len(jobs), desc="verify", unit="file"):
            counts += c
            total_images += total
            total_oor += oor
            per_file[name] = total
            log(f"  {name}: {total:,} imgs" + (f"  [{oor} OUT-OF-RANGE!]" if oor else ""))

    log(f"[verify_rec_out] TOTAL images written = {total_images:,}"
        + (f"  ({total_oor} out-of-range labels!)" if total_oor else ""))

    present = counts > 0
    missing = np.where(~present)[0]
    log(f"[verify_rec_out] id coverage: {int(present.sum()):,}/{K:,} ids present, "
        f"{len(missing):,} MISSING")
    if len(missing):
        sample = missing[:20].tolist()
        log(f"  missing id sample: {sample}{' ...' if len(missing) > 20 else ''}")

    cp = counts[present]
    dist = {"min": int(cp.min()), "max": int(cp.max()), "mean": float(cp.mean())}
    dist.update(_percentiles(cp))
    log(f"[verify_rec_out] per-id image count (overall, {len(cp):,} ids): "
        f"min={dist['min']} max={dist['max']} mean={dist['mean']:.2f} "
        f"median={dist['median']:.0f} p1={dist['p1']:.0f} p99={dist['p99']:.0f}")

    block_stats = {}
    for src in SOURCES:
        if src not in block_ranges:
            continue
        lo, hi = block_ranges[src]
        c = counts[lo:hi + 1]
        p = c[c > 0]
        if len(p) == 0:
            continue
        bd = {"ids": int(len(c)), "ids_present": int(len(p)),
              "imgs": int(c.sum()), "min": int(p.min()), "max": int(p.max()),
              "mean": float(p.mean()), "median": float(np.median(p))}
        block_stats[src] = bd
        log(f"[verify_rec_out]   {src} [{lo}..{hi}]: {bd['ids_present']:,}/{bd['ids']:,} "
            f"ids present, {bd['imgs']:,} imgs, min={bd['min']} max={bd['max']} "
            f"mean={bd['mean']:.2f} median={bd['median']:.0f}")

    if expected_real >= 0:
        log(f"[verify_rec_out] meta num_image(real, excl. synthetic) = {expected_real:,} "
            f"-- written total above includes synthetic too, not directly comparable 1:1")

    report = {"total_images": int(total_images), "out_of_range": int(total_oor),
              "num_classes": K, "ids_present": int(present.sum()),
              "ids_missing": int(len(missing)), "missing_sample": missing[:200].tolist(),
              "distribution_overall": dist, "distribution_per_source": block_stats,
              "per_file_counts": per_file}
    out_path = os.path.join(CFG.work_dir, "verify_rec_out_report.json")
    json.dump(report, open(out_path, "w"), indent=2)
    log(f"[verify_rec_out] report written -> {out_path}")

    ok = total_oor == 0 and len(missing) == 0
    log(f"[verify_rec_out] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    run()

"""
Post-process tool: drop one or more final_ids from the finished rec_out/ files
and re-assign labels to fill the gaps — WITHOUT re-running any upstream stage
or re-fetching from S3.

How it works
------------
For each rec_out/*.rec file (in parallel, one worker per file):
  - Sequential forward scan (MXRecordIO, no .idx needed for reading).
  - Records whose label is in DROP_IDS are skipped entirely.
  - Surviving records get a remapped label:
      new_label = old_label - (number of dropped ids strictly below old_label)
    This fills every gap left by the removed ids so the final range is
    contiguous again [0 .. K - len(DROP_IDS) - 1].
  - Written to a temporary .new.rec/.new.idx, then atomically renamed over
    the original once the file is done — so a crash leaves the originals
    intact and the run is safely re-startable.

After all files are rewritten, meta.json is updated:
  num_classes -= len(DROP_IDS)
  num_image   -= total images dropped (counted during rewrite)

Usage
-----
  DROP_IDS=<id1>[,<id2>,...] TEST_PIPELINE=0 WORK_DIR=... python3 drop_reindex_rec.py

  DROP_IDS: comma-separated list of final_ids to remove (from large_ids.json).
  To find the ids: look at WORK_DIR/large_ids.json (sorted desc by img_count).

Example (drop the single 980k-image id whose final_id is e.g. 1234567):
  DROP_IDS=1234567 TEST_PIPELINE=0 WORK_DIR=/workspace/.../model_viettelai005 python3 drop_reindex_rec.py
"""
from __future__ import annotations

import bisect
import glob
import json
import numbers
import os
from multiprocessing import get_context

from tqdm import tqdm

from common import log, die, meta_read, meta_write
from config import CFG

NPROC = int(os.environ.get("NPROC", str(CFG.cpu_workers)))


# --------------------------------------------------------------------------- #
# worker — runs in subprocess, rewrites one .rec file
# --------------------------------------------------------------------------- #
def _rewrite_file(args):
    path, drop_set, sorted_drops = args
    import mxnet as mx

    base = path[:-4]  # strip .rec
    tmp_rec = base + ".new.rec"
    tmp_idx = base + ".new.idx"

    r = mx.recordio.MXRecordIO(path, "r")
    w = mx.recordio.MXIndexedRecordIO(tmp_idx, tmp_rec, "w")

    new_i = 0
    n_dropped = 0
    while True:
        item = r.read()
        if item is None:
            break
        header, img = mx.recordio.unpack(item)
        lab = int(header.label if isinstance(header.label, numbers.Number)
                  else header.label[0])
        if lab in drop_set:
            n_dropped += 1
            continue
        new_lab = lab - bisect.bisect_left(sorted_drops, lab)
        hdr = mx.recordio.IRHeader(0, float(new_lab), new_i, 0)
        w.write_idx(new_i, mx.recordio.pack(hdr, img))
        new_i += 1

    r.close()
    w.close()

    # atomic rename: original is replaced only when new file is fully written
    os.rename(tmp_rec, path)
    os.rename(tmp_idx, base + ".idx")

    return os.path.basename(path), n_dropped, new_i


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run():
    raw = os.environ.get("DROP_IDS", "").strip()
    if not raw:
        # show top-10 candidates from large_ids.json to help user pick
        lp = os.path.join(CFG.work_dir, "large_ids.json")
        if os.path.exists(lp):
            top = json.load(open(lp))[:10]
            log("[drop_reindex] DROP_IDS not set. Top ids by image count:")
            for x in top:
                log(f"  final_id={x['final_id']}  img_count={x['img_count']:,}")
        die("Set DROP_IDS=<id1>[,<id2>,...] and re-run.")

    drop_ids = sorted({int(x) for x in raw.split(",") if x.strip()})
    drop_set = frozenset(drop_ids)
    log(f"[drop_reindex] will DROP {len(drop_ids)} id(s): {drop_ids}")

    # sanity: check ids exist in meta
    meta = meta_read(CFG.path_meta())
    K = int(meta.get("num_classes", 0))
    if not K:
        die("meta.json has no num_classes — something is wrong")
    for d in drop_ids:
        if d < 0 or d >= K:
            die(f"DROP_ID {d} is out of range [0, {K})")

    rec_files = sorted(glob.glob(os.path.join(CFG.rec_out_dir, "*.rec")))
    if not rec_files:
        die(f"no .rec files found in {CFG.rec_out_dir}")

    # skip files already rewritten in a previous partial run
    # (tmp files renamed over originals, so if .new.rec doesn't exist the file
    #  is either untouched-original or already-done; we re-run untouched ones)
    jobs = [(f, drop_set, drop_ids) for f in rec_files
            if not os.path.exists(f[:-4] + ".new.rec")]
    log(f"[drop_reindex] {len(rec_files)} rec files, {len(jobs)} need rewriting "
        f"({len(rec_files) - len(jobs)} already done / skipped)")

    total_dropped = 0
    total_kept = 0
    nproc = max(1, min(NPROC, len(jobs)))
    ctx = get_context("spawn")
    with ctx.Pool(nproc) as pool:
        for name, n_dropped, n_kept in tqdm(
                pool.imap_unordered(_rewrite_file, jobs),
                total=len(jobs), desc="rewrite", unit="file"):
            total_dropped += n_dropped
            total_kept += n_kept
            log(f"  {name}: dropped={n_dropped:,}  kept={n_kept:,}")

    new_K = K - len(drop_ids)
    old_num_image = int(meta.get("num_image", 0))
    new_num_image = max(0, old_num_image - total_dropped)
    meta_write(CFG.path_meta(),
               num_classes=new_K,
               num_image=new_num_image,
               dropped_ids=drop_ids)

    log(f"[drop_reindex] DONE")
    log(f"  images dropped : {total_dropped:,}")
    log(f"  images kept    : {total_kept:,}")
    log(f"  num_classes    : {K:,} -> {new_K:,}")
    log(f"  num_image(real): {old_num_image:,} -> {new_num_image:,}  "
        f"(approx — synthetic not counted in meta.num_image)")
    log(f"  meta.json updated -> {CFG.path_meta()}")


if __name__ == "__main__":
    run()

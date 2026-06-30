"""
Inspection tool: visualise randomly-sampled images for every id that has
more than THRESHOLD images in the final rec_out/ files.

Steps
-----
1. Read WORK_DIR/large_ids.json (written by verify_rec_out.py) and keep only
   ids with img_count > THRESHOLD (default 2000).
2. Scan every rec_out/*.rec sequentially in parallel (one worker per file) to
   collect (file_path, sequential_idx) for each target id.
3. For each target id, draw N_SAMPLE random (file, idx) pairs, fetch the raw
   JPEG bytes via MXIndexedRecordIO.read_idx(), decode to RGB, tile into a
   grid of GRID_COLS columns with an ID / count title bar, and save to
   WORK_DIR/large_id_grids/grid_{final_id}_{count}imgs.jpg.

Env vars (all optional, fall back to config defaults):
  WORK_DIR      — same as the rest of the pipeline
  THRESHOLD     — min img_count to inspect (default 2000)
  N_SAMPLE      — images to sample per id (default 200)
  GRID_COLS     — images per row (default 10)
  GRID_IMG_SIZE — each thumbnail side in px (default 112)
  NPROC         — parallel scan workers (default CFG.cpu_workers)
"""
from __future__ import annotations

import glob
import json
import numbers
import os
import random
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from common import log, die
from config import CFG

THRESHOLD = int(os.environ.get("THRESHOLD", "2000"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "200"))
GRID_COLS = int(os.environ.get("GRID_COLS", "10"))
GRID_IMG_SIZE = int(os.environ.get("GRID_IMG_SIZE", "112"))
NPROC = int(os.environ.get("NPROC", str(CFG.cpu_workers)))
TITLE_H = 32   # px height for the text bar above each grid


# --------------------------------------------------------------------------- #
# worker: sequential scan of one .rec file, collect (seq_idx) per target id
# --------------------------------------------------------------------------- #
def _scan_file(args):
    path, target_ids_frozen = args
    import mxnet as mx
    result = defaultdict(list)
    i = 0
    rec = mx.recordio.MXRecordIO(path, "r")
    while True:
        item = rec.read()
        if item is None:
            break
        header, _ = mx.recordio.unpack(item)
        lab = int(header.label if isinstance(header.label, numbers.Number)
                  else header.label[0])
        if lab in target_ids_frozen:
            result[lab].append(i)
        i += 1
    rec.close()
    return path, dict(result)


# --------------------------------------------------------------------------- #
# fetch one image bytes via indexed random access
# --------------------------------------------------------------------------- #
_rec_cache: dict = {}


def _fetch_bytes(path: str, idx: int) -> bytes:
    import mxnet as mx
    if path not in _rec_cache:
        idx_path = path[:-4] + ".idx"
        _rec_cache[path] = mx.recordio.MXIndexedRecordIO(idx_path, path, "r")
    _, img_bytes = mx.recordio.unpack(_rec_cache[path].read_idx(idx))
    return img_bytes


# --------------------------------------------------------------------------- #
# build grid image for one id
# --------------------------------------------------------------------------- #
def _make_grid(samples: list[tuple[str, int]], final_id: int, count: int,
               img_size: int, cols: int) -> Image.Image:
    import cv2
    thumbs = []
    for path, idx in samples:
        raw = _fetch_bytes(path, idx)
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((img_size, img_size, 3), np.uint8)
        rgb = cv2.cvtColor(cv2.resize(bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        thumbs.append(Image.fromarray(rgb))

    rows = (len(thumbs) + cols - 1) // cols
    grid_w = cols * img_size
    grid_h = rows * img_size + TITLE_H
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

    draw = ImageDraw.Draw(grid)
    title = f"final_id={final_id}  total_imgs={count:,}  sampled={len(thumbs)}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(255, 220, 60), font=font)

    for k, thumb in enumerate(thumbs):
        r, c = divmod(k, cols)
        grid.paste(thumb, (c * img_size, TITLE_H + r * img_size))
    return grid


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run():
    large_path = os.path.join(CFG.work_dir, "large_ids.json")
    if not os.path.exists(large_path):
        die(f"large_ids.json not found at {large_path} — run verify_rec_out.py first")
    all_large = json.load(open(large_path))
    targets = [x for x in all_large if x["img_count"] > THRESHOLD]
    if not targets:
        log(f"[inspect] no ids with img_count > {THRESHOLD}")
        return
    log(f"[inspect] {len(targets)} ids with >{THRESHOLD} imgs (threshold={THRESHOLD})")

    target_ids = frozenset(x["final_id"] for x in targets)
    count_map = {x["final_id"]: x["img_count"] for x in targets}

    rec_files = sorted(glob.glob(os.path.join(CFG.rec_out_dir, "*.rec")))
    if not rec_files:
        die(f"no .rec files found in {CFG.rec_out_dir}")
    log(f"[inspect] scanning {len(rec_files)} rec files for target ids...")

    # parallel scan
    jobs = [(f, target_ids) for f in rec_files]
    nproc = max(1, min(NPROC, len(jobs)))
    global_map: dict[int, list[tuple[str, int]]] = defaultdict(list)
    ctx = get_context("spawn")
    with ctx.Pool(nproc) as pool:
        for path, result in tqdm(pool.imap_unordered(_scan_file, jobs),
                                 total=len(jobs), desc="scan", unit="file"):
            for fid, idxs in result.items():
                global_map[fid].extend((path, i) for i in idxs)

    out_dir = os.path.join(CFG.work_dir, "large_id_grids")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(42)
    for entry in tqdm(sorted(targets, key=lambda x: -x["img_count"]),
                      desc="grids", unit="id"):
        fid = entry["final_id"]
        total = count_map[fid]
        pool_list = global_map.get(fid, [])
        if not pool_list:
            log(f"  [warn] id {fid}: found 0 records in scan, skipping")
            continue
        n = min(N_SAMPLE, len(pool_list))
        samples = rng.sample(pool_list, n)
        grid = _make_grid(samples, fid, total, GRID_IMG_SIZE, GRID_COLS)
        fname = f"grid_{fid:07d}_{total}imgs.jpg"
        grid.save(os.path.join(out_dir, fname), quality=90)
        log(f"  saved {fname}  ({n} sampled from {len(pool_list):,} found)")

    log(f"[inspect] done — {len(targets)} grids saved to {out_dir}")


if __name__ == "__main__":
    run()

"""
Shared, dependency-light primitives used by every stage.

Design rules honored here:
- RAM-safe: helpers iterate parquet SHARD FILES one at a time; nothing loads the
  whole dataset. The only full-RAM object anywhere is the per-id TEMPLATE table
  (~3.6M x 512 f32 ~= 7.5GB), handled in the merge stage.
- single-responsibility: this module has no stage logic, only reusable pieces.
- test/real parity: `range_search` picks faiss (real) or sklearn (test) so the
  exact same merge logic runs in both modes.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import polars as pl

from config import WEBFACE, PUBLIC, CRAWL, SOURCES

# Canonical image-level parquet schema (one row per image):
#   person_id : Int64      global-ish id (native; crawl gets +offset later)
#   src       : Utf8       'webface' | 'public' | 'crawl'
#   img_key   : Utf8       rec sources: f"{prefix}:{rec_idx}"; crawl: aligned_s3_path
#   embedding / embedding_normalized : List(Float32)[512]
COL_ID, COL_SRC, COL_KEY = "person_id", "src", "img_key"
COL_EMB, COL_EMBN = "embedding", "embedding_normalized"


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 1):
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# parquet shard IO (RAM-safe iteration)
# --------------------------------------------------------------------------- #
def list_parquet(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "*.parquet")))


def iter_parquet(dir_path: str, columns=None) -> Iterable[Tuple[str, pl.DataFrame]]:
    """Yield (file_path, df) one shard file at a time — never holds two at once."""
    for fp in list_parquet(dir_path):
        yield fp, pl.read_parquet(fp, columns=columns)


def emb_matrix(df: pl.DataFrame, col: str) -> np.ndarray:
    """Embedding column (float16 fixed-size-list OR float list) -> float32 (N, D)."""
    if df.height == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(df[col].to_list(), dtype=np.float32)


def write_emb_parquet(path: str, meta_df: pl.DataFrame, emb: np.ndarray, emb_col: str):
    """Write meta columns + the embedding as a fixed_size_list<float16> column.
    float16 is ~4x smaller on disk than the default python-float (Float64) list
    polars infers, with negligible effect on cosine DBSCAN / centroids. Read back
    via emb_matrix() which upcasts to float32. emb: (N, D) array."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float16))
    n, d = emb.shape
    tbl = meta_df.to_arrow().append_column(
        emb_col,
        pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), type=pa.float16()), d))
    pq.write_table(tbl, path)


# --------------------------------------------------------------------------- #
# per-id checkpoint (resume after crash)
# --------------------------------------------------------------------------- #
def ckpt_load(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {int(x) for x in f.read().split() if x.strip()}


def ckpt_append(path: str, ids: Iterable[int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for i in ids:
            f.write(f"{int(i)}\n")


# --------------------------------------------------------------------------- #
# union-find (replaces igraph connected_components, zero deps)
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

    def components(self) -> Dict[int, List[int]]:
        comp: Dict[int, List[int]] = {}
        for i in range(len(self.p)):
            comp.setdefault(self.find(i), []).append(i)
        return comp


# --------------------------------------------------------------------------- #
# range search backend: faiss (real) | sklearn (test) — same output contract
# --------------------------------------------------------------------------- #
def range_search(vecs: np.ndarray, lower_thr: float, n_clusters: int, nprobe: int):
    """Return (src_idx, tgt_idx, sim) arrays for all pairs with cosine sim >= lower_thr,
    de-duplicated to i < j. Vectors MUST be L2-normalized (cosine == inner product)."""
    n = len(vecs)
    if n < 2:
        return (np.array([], int), np.array([], int), np.array([], np.float32))
    try:
        import faiss  # real path
        dim = vecs.shape[1]
        ncl = max(1, min(n_clusters, n // 30))
        log(f"  [range_search] faiss IVF{ncl} nprobe={nprobe} on {n:,} vecs")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, ncl, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = nprobe
        lims, D, I = index.range_search(vecs, float(lower_thr))
        lims, D, I = lims.astype("int64"), D.astype("float32"), I.astype("int64")
        src = np.repeat(np.arange(n), np.diff(lims))
        tgt, sim = I, D
    except ImportError:  # test path: exact, small N
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", radius=1.0 - lower_thr).fit(vecs)
        dist, idx = nn.radius_neighbors(vecs)
        src = np.repeat(np.arange(n), [len(a) for a in idx])
        tgt = np.concatenate(idx) if n else np.array([], int)
        sim = 1.0 - np.concatenate(dist) if n else np.array([], np.float32)
    keep = src < tgt
    return src[keep].astype(int), tgt[keep].astype(int), sim[keep].astype(np.float32)


# --------------------------------------------------------------------------- #
# core symmetric merge — shared by internal & global merge stages
# --------------------------------------------------------------------------- #
def symmetric_merge(ids: np.ndarray, counts: np.ndarray, vecs: np.ndarray,
                    lower_thr: float, upper_thr: float, n_clusters: int, nprobe: int):
    """SYMMETRIC, no anchor (per user). Returns (merge_map, drop_ids):
       merge_map: {orig_id -> leader_id} for non-leader, non-dropped ids
       drop_ids : set of orig ids removed in the (lower, upper] uncertain zone
    Leader of a merged component = max image count (tie -> smaller id).
    Drop zone: weaker (fewer-image) side of an uncertain pair is dropped."""
    n = len(ids)
    log(f"  [merge] range-search over {n:,} centroids (lower={lower_thr}, upper={upper_thr})...")
    s, t, sim = range_search(vecs, lower_thr, n_clusters, nprobe)
    merge_pair = sim > upper_thr
    drop_pair = (sim > lower_thr) & (sim <= upper_thr)
    log(f"  [merge] {len(s):,} candidate pairs -> {int(merge_pair.sum()):,} merge / "
        f"{int(drop_pair.sum()):,} drop-zone")

    uf = UnionFind(n)
    for a, b in zip(s[merge_pair], t[merge_pair]):
        uf.union(int(a), int(b))

    # leader (by count) per component; component total count for drop resolution
    comps = uf.components()
    multi = sum(1 for m in comps.values() if len(m) > 1)
    log(f"  [merge] union-find: {multi:,} multi-id components to collapse")
    leader_idx = {i: i for i in range(n)}
    comp_count = counts.astype(np.int64).copy()
    for root, members in comps.items():
        if len(members) < 2:
            continue
        leader = max(members, key=lambda x: (counts[x], -ids[x]))
        total = int(sum(counts[m] for m in members))
        for m in members:
            leader_idx[m] = leader
            comp_count[m] = total

    drop_idx: Set[int] = set()
    for u, v in zip(s[drop_pair], t[drop_pair]):
        lu, lv = leader_idx[int(u)], leader_idx[int(v)]
        if lu == lv:
            continue
        if comp_count[lu] > comp_count[lv]:
            drop_idx.add(lv)
        elif comp_count[lv] > comp_count[lu]:
            drop_idx.add(lu)
        else:
            drop_idx.add(max(lu, lv))

    drop_ids: Set[int] = set()
    merge_map: Dict[int, int] = {}
    for i in range(n):
        leader = leader_idx[i]
        if leader in drop_idx:
            drop_ids.add(int(ids[i]))
    for i in range(n):
        if int(ids[i]) in drop_ids:
            continue
        leader = leader_idx[i]
        if i != leader:
            merge_map[int(ids[i])] = int(ids[leader])
    return merge_map, drop_ids


def write_merge_artifacts(out_dir: str, merge_map: Dict[int, int], drop_ids: Set[int]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "drop.txt"), "w") as f:
        for pid in sorted(drop_ids):
            f.write(f"{pid}\n")
    if merge_map:
        pl.DataFrame({"og_id": list(merge_map.keys()),
                      "new_id": list(merge_map.values())}).write_csv(
            os.path.join(out_dir, "map.csv"))
    else:  # always leave a (possibly empty) map so downstream can read unconditionally
        pl.DataFrame({"og_id": [], "new_id": []},
                     schema={"og_id": pl.Int64, "new_id": pl.Int64}).write_csv(
            os.path.join(out_dir, "map.csv"))


def read_merge_artifacts(in_dir: str):
    drop_ids: Set[int] = set()
    dp = os.path.join(in_dir, "drop.txt")
    if os.path.exists(dp):
        drop_ids = ckpt_load(dp)
    merge_map: Dict[int, int] = {}
    mp = os.path.join(in_dir, "map.csv")
    if os.path.exists(mp):
        df = pl.read_csv(mp)
        if df.height:
            merge_map = dict(zip(df["og_id"].to_list(), df["new_id"].to_list()))
    return merge_map, drop_ids


# --------------------------------------------------------------------------- #
# meta.json
# --------------------------------------------------------------------------- #
def meta_write(path: str, **kv):
    cur = {}
    if os.path.exists(path):
        cur = json.load(open(path))
    cur.update(kv)
    json.dump(cur, open(path, "w"), indent=2)


def meta_read(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


# --------------------------------------------------------------------------- #
# TEST FIXTURES: tiny synthetic per-source embeddings with KNOWN structure so
# every stage can be asserted locally (no GPU/S3/mxnet/faiss).
# --------------------------------------------------------------------------- #
def _unit(v):
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def make_fixtures(cfg):
    """Build small per-source embed parquet shards encoding test cases:
      - clean ids with >=min_samples images       (should survive)
      - an id with a noise outlier image           (dbscan drops the outlier)
      - an id with < min_samples images            (dbscan drops the whole id)
      - two near-duplicate ids in the SAME source  (internal merge collapses)
      - the SAME person present in webface AND crawl (global merge collapses)
    """
    rng = np.random.default_rng(0)
    D = cfg.embedding_size

    def cluster(center, k, jitter=0.02):
        return _unit(center + rng.normal(0, jitter, size=(k, D)))

    centers = _unit(rng.normal(0, 1, size=(40, D)))
    shared_person = _unit(rng.normal(0, 1, size=(1, D)))[0]  # appears in webface & crawl

    rows = {WEBFACE: [], PUBLIC: [], CRAWL: []}

    def add(src, pid, embs, keyfn):
        for j, e in enumerate(embs):
            rows[src].append({COL_ID: int(pid), COL_SRC: src,
                              COL_KEY: keyfn(pid, j), COL_EMB: e.tolist()})

    reck = lambda pre: (lambda pid, j: f"{pre}:{pid*100+j}")
    s3k = lambda pid, j: f"{cfg.crawl_s3_prefix}/person_0_9999/person_{pid:07d}.tar/{j}.jpg"

    # webface ids 0..9 ; id 5 has an outlier; id 9 has only 2 imgs (dropped)
    for pid in range(10):
        e = cluster(centers[pid], 5)
        if pid == 5:
            e[0] = _unit(rng.normal(0, 1, size=(1, D)))[0]  # outlier image
        if pid == 9:
            e = e[:2]                                        # < min_samples
        add(WEBFACE, pid, e, reck("train"))
    # webface ids 10 & 11 are the SAME person -> internal merge
    dup = centers[10]
    add(WEBFACE, 10, cluster(dup, 6), reck("train"))
    add(WEBFACE, 11, cluster(dup, 4), reck("train"))
    # webface id 12 == shared_person (also appears in crawl below)
    add(WEBFACE, 12, cluster(shared_person, 5), reck("train"))

    # public ids 100..104 (already offset after webface in real data; here just disjoint)
    for pid in range(100, 105):
        add(PUBLIC, pid, cluster(centers[20 + (pid - 100)], 5), reck("train_public"))

    # crawl ids 1..5 (native, from 1); id 3 == shared_person -> global merge w/ webface 12
    for pid in range(1, 6):
        c = shared_person if pid == 3 else centers[30 + pid]
        add(CRAWL, pid, cluster(c, 5), s3k)

    for src in SOURCES:
        d = cfg.dir_embed(src)
        os.makedirs(d, exist_ok=True)
        pl.DataFrame(rows[src]).write_parquet(os.path.join(d, "part-0000.parquet"))
        log(f"[fixtures] {src}: {len(rows[src])} imgs -> {d}")

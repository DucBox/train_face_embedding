"""
Test: double-hop merge correctness.

Scenario:
  webface: id 50 = shared identity S
  crawl:   id 1 = S, id 2 = S  (-> internal merge {1,2}; then global merge with webface 50)
  crawl:   id 3 = clean distinct
  public:  id 200, 201 = clean distinct

Expected: webface:50, crawl:1, crawl:2 all map to the SAME final_id
(crawl 2 goes through double-hop: crawl2->internal leader crawl1->global->webface50)

Run:
    cd data_clean_pipeline
    WORK_DIR=/tmp/dh_test python3 test_double_hop.py
"""
import glob
import os
import shutil
import sys

import numpy as np
import polars as pl

os.environ.setdefault("WORK_DIR", "/tmp/dh_test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import COL_ID, COL_SRC, COL_KEY, COL_EMB
from config import CFG, WEBFACE, PUBLIC, CRAWL

import normalize, dbscan, template, merge, reindex

# ------------------------------------------------------------------ helpers --
rng = np.random.default_rng(1)
D = CFG.embedding_size


def unit(v):
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def clust(center, k, noise=0.02):
    return unit(center + rng.normal(0, noise, size=(k, D)))


def write_fixtures(src, items):
    """Write embed-stage parquet for a source. items = [(pid, emb_array), ...]"""
    rows = []
    for pid, embs in items:
        for j, e in enumerate(embs):
            rows.append({COL_ID: pid, COL_SRC: src,
                         COL_KEY: f"{src}:{pid}:{j}", COL_EMB: e.tolist()})
    d = CFG.dir_embed(src)
    os.makedirs(d, exist_ok=True)
    pl.DataFrame(rows).write_parquet(os.path.join(d, "p0.parquet"))


# ------------------------------------------------------------------ setup ---
WORK_DIR = CFG.work_dir
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

S = unit(rng.normal(0, 1, size=(1, D)))[0]       # shared identity across sources
cen = unit(rng.normal(0, 1, size=(30, D)))

# webface: 50=S, 100/101 distinct
write_fixtures(WEBFACE, [(50, clust(S, 5)), (100, clust(cen[0], 5)), (101, clust(cen[1], 5))])
# public:  200/201 distinct
write_fixtures(PUBLIC, [(200, clust(cen[2], 5)), (201, clust(cen[3], 5))])
# crawl:   1=S, 2=S (internal merge), 3 distinct
write_fixtures(CRAWL, [(1, clust(S, 6)), (2, clust(S, 4)), (3, clust(cen[4], 5))])

# ------------------------------------------------------------------ pipeline -
for src in (WEBFACE, PUBLIC, CRAWL):
    normalize.run(src)
    dbscan.run(src)
    template.run_src(src)
    merge.run_src(src)

template.run_global()
merge.run_global()
reindex.run()

# ------------------------------------------------------------------ check ---
lm = pl.read_parquet(os.path.join(CFG.dir_reindex(), "label_map.parquet"))
print("\n================ DOUBLE-HOP CHECK ================")
print("label_map:\n", lm.sort(["src", "orig_id"]))


def fin(src, orig):
    r = lm.filter((pl.col("src") == src) & (pl.col("orig_id") == orig))
    return None if r.height == 0 else int(r["final_id"][0])


f_wf50 = fin(WEBFACE, 50)
f_c1   = fin(CRAWL, 1)
f_c2   = fin(CRAWL, 2)
print(f"\nwebface 50  -> {f_wf50}")
print(f"crawl 1     -> {f_c1}   (internal leader of crawl {{1,2}})")
print(f"crawl 2     -> {f_c2}   (double-hop: crawl2->crawl1->global)")
ok_dh = (f_wf50 is not None and f_wf50 == f_c1 == f_c2 and f_wf50 >= 0)
print(f"\n[DOUBLE-HOP] webface50 == crawl1 == crawl2 == {f_wf50} ? "
      f"{'PASS ✓' if ok_dh else 'FAIL ✗'}")

# coverage: every dbscan id has a label_map entry
cov_ok = True
for src in (WEBFACE, PUBLIC, CRAWL):
    ids = set()
    for fp in glob.glob(os.path.join(CFG.dir_dbscan(src), "*.parquet")):
        ids |= set(pl.read_parquet(fp, columns=[COL_ID])[COL_ID].to_list())
    lm_ids = set(lm.filter(pl.col("src") == src)["orig_id"].to_list())
    miss = ids - lm_ids
    print(f"[COVERAGE] {src}: dbscan_ids={len(ids)} missing_in_label_map={len(miss)}")
    cov_ok &= (len(miss) == 0)
print(f"[COVERAGE] all present? {'PASS ✓' if cov_ok else 'FAIL ✗'}")

if not (ok_dh and cov_ok):
    sys.exit(1)
print("\nAll checks PASSED")

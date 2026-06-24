"""
GLOBAL CONFIG for the iterative dataset self-cleaning pipeline.

This is the ONE place to edit knobs. Every stage script imports `CFG` from here,
so changing a threshold / path / worker count here propagates to the whole flow.

Flow (see README.md):
  per-source: embed -> normalize -> dbscan -> template -> merge(internal)
  global:     offset(crawl) -> template -> merge(global, symmetric) -> reindex
  output:     write 3 .rec files with one contiguous global id range

Run `python -c "from config import CFG; CFG.dump()"` to print the resolved config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# Source names used everywhere as dict keys / parquet `src` column values.
WEBFACE = "webface"
PUBLIC = "public"
CRAWL = "crawl"
SOURCES = (WEBFACE, PUBLIC, CRAWL)


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # MASTER SWITCH
    # ------------------------------------------------------------------ #
    # True  -> run on tiny self-generated synthetic fixtures, asserting every
    #          stage's logic locally (no GPU / S3 / mxnet / faiss needed).
    # False -> run the real pipeline on the cluster.
    test_pipeline: bool = bool(int(os.environ.get("TEST_PIPELINE", "1")))

    # ------------------------------------------------------------------ #
    # PATHS  (override per-run with env vars; real-run values below)
    # ------------------------------------------------------------------ #
    # Working root: every stage writes a sub-folder under here.
    work_dir: str = os.environ.get(
        "WORK_DIR",
        "/workspace/FaceNist/raw_data_processing/clean_loop/run01",
    )

    # New-model backbone weights used to RE-EMBED everything this round.
    model_weight: str = os.environ.get("MODEL_WEIGHT", "/path/to/new_model.pt")
    network: str = os.environ.get("NETWORK", "vit_l_dinov3")
    embedding_size: int = 512

    # --- rec sources (webface pure + public) ---
    # Folder holding the source .rec/.idx pairs.
    rec_root: str = os.environ.get("REC_ROOT", "/workspace/data/face_embedding/data")
    # rec prefixes (without extension) per source. webface uses the PURE rec
    # (no synthetic); synthetic is only re-attached at write time.
    rec_prefix = {
        WEBFACE: ["train"],          # pure webface
        PUBLIC: ["train_public"],
    }
    # synthetic rec prefix (label == parent webface id). Embed/clean never touch
    # it; write_rec remaps it by the final webface id-map.
    synthetic_prefix: str = "train_synthetic"

    # --- crawl source (197M aligned crops on S3, 1 tar / person) ---
    s3_endpoint: str = os.environ.get("S3_ENDPOINT", "http://s3-data.cyberspace.vn")
    s3_bucket: str = os.environ.get("S3_BUCKET", "ttnt-data")   # confirmed via download_person.py
    s3_access_key: str = os.environ.get("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.environ.get("S3_SECRET_KEY", "")
    # prefix containing per-person tars: {prefix}/person_{start}_{end}/person_{pid:07d}.tar
    # confirmed: aligned 112x112 crawl crops live here, ids from 0, shard folders person_0_999.
    crawl_s3_prefix: str = os.environ.get(
        "CRAWL_S3_PREFIX", "cv/processed-datasets/aligned_face_112_112")
    crawl_shard_size: int = 1000           # folder person_{start}_{start+999}
    # offset_map parquet: tar_path, member_name, start_byte, length (create_offset_map.py)
    offset_map_path: str = os.environ.get("OFFSET_MAP", "/path/to/offset_table.parquet")

    # ------------------------------------------------------------------ #
    # STAGE KNOBS
    # ------------------------------------------------------------------ #
    # --- embed ---
    embed_batch_size: int = 256
    embed_num_workers: int = 4            # DataLoader workers (rec embed)
    embed_flush_rows: int = 500_000       # RAM cap: rows held before a parquet flush

    # --- normalize / dbscan / template / merge: CPU parallelism ---
    cpu_workers: int = int(os.environ.get("CPU_WORKERS", "48"))

    # --- dbscan (per-id intra-clean), PER SOURCE ---
    # eps in COSINE distance (1 - cosine_sim). min_samples drops any id below it.
    dbscan_eps = {WEBFACE: 0.40, PUBLIC: 0.40, CRAWL: 0.30}
    dbscan_min_samples: int = 3

    # --- merge IVF (symmetric, used for BOTH internal and global) ---
    # cosine-similarity thresholds. > upper => merge; (lower, upper] => drop weaker side.
    merge_lower_thr: float = 0.43
    merge_upper_thr: float = 0.56
    # faiss IVF params (real run). n_clusters auto-shrinks for tiny inputs.
    ivf_n_clusters: int = 4096
    ivf_nprobe: int = 64

    # --- offset: crawl id offset is DYNAMIC = max(webface,public eff id)+1.
    # Set a hard floor only as a sanity assert (must stay > webface+public range).
    crawl_offset_floor: int = 0

    # ------------------------------------------------------------------ #
    # OUTPUT
    # ------------------------------------------------------------------ #
    rec_out_dir: str = ""                  # filled in __post_init__ (= work_dir/rec_out)
    crawl_out_shards: int = 32             # number of train_crawl_*.rec shards

    # ------------------------------------------------------------------ #
    # DERIVED / HELPERS
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if not self.rec_out_dir:
            self.rec_out_dir = os.path.join(self.work_dir, "rec_out")

    # canonical sub-dirs (single source of truth for inter-stage data contract)
    def dir_embed(self, src):     return os.path.join(self.work_dir, "01_embed", src)
    def dir_norm(self, src):      return os.path.join(self.work_dir, "02_norm", src)
    def dir_dbscan(self, src):    return os.path.join(self.work_dir, "03_dbscan", src)
    def dir_template(self, src):  return os.path.join(self.work_dir, "04_template", src)
    def dir_merge_int(self, src): return os.path.join(self.work_dir, "05_merge_internal", src)
    def dir_template_global(self): return os.path.join(self.work_dir, "06_template_global")
    def dir_merge_global(self):   return os.path.join(self.work_dir, "07_merge_global")
    def dir_reindex(self):        return os.path.join(self.work_dir, "08_reindex")
    def path_meta(self):          return os.path.join(self.work_dir, "meta.json")

    def ckpt(self, stage, src=""):
        d = os.path.join(self.work_dir, "_checkpoints")
        os.makedirs(d, exist_ok=True)
        name = f"{stage}_{src}.txt" if src else f"{stage}.txt"
        return os.path.join(d, name)

    def dump(self):
        import json
        print(json.dumps({k: v for k, v in asdict(self).items()}, indent=2, default=str))


CFG = Config()

if __name__ == "__main__":
    CFG.dump()

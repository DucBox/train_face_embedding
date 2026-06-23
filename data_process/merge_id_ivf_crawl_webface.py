import polars as pl
import numpy as np
import faiss
import igraph as ig
import os
import glob
import multiprocessing
from tqdm import tqdm
import gc
import psutil
import time
from datetime import datetime

# CONFIG
WEBFACE_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_webface42m/centers.parquet"  # IDs < 3M
CRAWL_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_v1/centers_v2.parquet" # IDs > 3M
INPUT_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/v1_ivf_real"
OUTPUT_FINAL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real"
INDEX_SAVE_PATH = os.path.join(OUTPUT_FINAL_DIR, "faiss_index.bin")
LOWER_THR = 0.43194
UPPER_THR = 0.56023
DIMENSION = 512
N_CLUSTERS = 4096
N_PROBE = 64
NUM_WORKERS_IO = 32
OFFSET_THRESHOLD = 3000000

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def process_file_apply(args):
    file_path, merge_map_dict = args
    try:
        df = pl.read_parquet(file_path)
        if df.height == 0:
            return

        if merge_map_dict:
            df = df.with_columns(
                pl.col("person_id").replace(merge_map_dict, default=pl.col("person_id"))
            )

        df = df.unique()

        out_name = os.path.basename(file_path)
        df.write_parquet(os.path.join(OUTPUT_FINAL_DIR, out_name))
        del df
    except Exception as e:
        print(f"[ERR] File {file_path}: {e}")

def run_pipeline():
    t_total = time.time()
    log(f"[START] PID={os.getpid()}")

    # STEP 1 LOAD
    t0 = time.time()
    log(f"[STEP 1] Loading WebFace & Crawl data...")
    try:
        df_w = pl.read_parquet(WEBFACE_PATH).select(["person_id", "img_count", "embedding_center"]).with_columns(pl.col("person_id").cast(pl.Int64), pl.col("img_count").cast(pl.Int64))
        df_c = pl.read_parquet(CRAWL_PATH).select(["person_id", "img_count", "embedding_center"]).with_columns(pl.col("person_id").cast(pl.Int64), pl.col("img_count").cast(pl.Int64))
        df_c = df_c.filter(pl.col("person_id") >= OFFSET_THRESHOLD)
        df_c = df_c.unique(subset=["person_id"], keep="first")

        df_centers = pl.concat([df_w, df_c])
        ids = df_centers["person_id"].to_list()
        counts = df_centers["img_count"].to_numpy()
        vecs = np.stack(df_centers["embedding_center"].to_numpy()).astype('float32')

        del df_w, df_c, df_centers
        idx_to_id = {i: pid for i, pid in enumerate(ids)}
        log(f" -> Total Loaded: {len(ids)} centers [T={time.time()-t0:.1f}s]")
    except Exception as e:
        log(f"[FATAL STEP1] {e}")
        return

    # STEP 2 FAISS
    t1 = time.time()
    log(f"[STEP 2] Building IVF{N_CLUSTERS}...")
    quantizer = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)

    index.train(vecs)
    index.add(vecs)
    index.nprobe = N_PROBE

    os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_SAVE_PATH)

    log(f" -> Range Search (> {LOWER_THR})...")
    lims, D, I = index.range_search(vecs, LOWER_THR)
    lims, D, I = lims.astype('int64'), D.astype('float32'), I.astype('int64')

    del quantizer, index, vecs
    gc.collect()
    log(f"[TIME] STEP2: {time.time()-t1:.1f}s")

    # STEP 3 EDGES
    t2 = time.time()
    log("[STEP 3] Converting Search Results to Edges...")
    sources = np.repeat(np.arange(len(ids)), np.diff(lims))
    targets = I
    scores = D

    mask_valid = sources < targets
    sources = sources[mask_valid]
    targets = targets[mask_valid]
    scores = scores[mask_valid]

    mask_merge = scores > UPPER_THR
    mask_drop = (scores > LOWER_THR) & (scores <= UPPER_THR)

    merge_edges = np.column_stack((sources[mask_merge], targets[mask_merge]))
    drop_src = sources[mask_drop]
    drop_tgt = targets[mask_drop]

    del sources, targets, scores, mask_valid, mask_merge, mask_drop, D, I
    gc.collect()
    log(f" -> {len(merge_edges)} merge | {len(drop_src)} drop candidates [T={time.time()-t2:.1f}s]")

    # STEP 4 GRAPH
    t3 = time.time()
    log(f"[STEP 4] iGraph processing...")
    if len(merge_edges) > 0:
        g = ig.Graph(len(ids), edges=merge_edges, directed=False)
        components = g.connected_components(mode="weak")
        del g
    else:
        components = []
    gc.collect()
    log(f"[TIME] STEP4: {time.time()-t3:.1f}s")

    # STEP 5 RESOLVE
    t4 = time.time()
    log("[STEP 5] Resolving (Crawl → WebFace Merge)...")
    raw_merge_map = {}

    for comp in components:
        if len(comp) < 2: continue

        wf_nodes = [n for n in comp if idx_to_id[n] < OFFSET_THRESHOLD]
        crawl_nodes = [n for n in comp if idx_to_id[n] >= OFFSET_THRESHOLD]

        if wf_nodes:
            leader = max(wf_nodes, key=lambda x: (counts[x], -x))
            leader_id = idx_to_id[leader]
            for n in crawl_nodes:
                raw_merge_map[idx_to_id[n]] = leader_id
            for n in wf_nodes:
                node_id = idx_to_id[n]
                if node_id != leader_id:
                    raw_merge_map[node_id] = leader_id
        else:
            if len(crawl_nodes) > 1:
                leader = max(crawl_nodes, key=lambda x: (counts[x], -x))
                leader_id = idx_to_id[leader]
                for n in crawl_nodes:
                    node_id = idx_to_id[n]
                    if node_id != leader_id:
                        raw_merge_map[node_id] = leader_id

    for u, v in zip(drop_src, drop_tgt):
        id_u, id_v = idx_to_id[u], idx_to_id[v]
        is_u_wf, is_v_wf = id_u < OFFSET_THRESHOLD, id_v < OFFSET_THRESHOLD

        if is_u_wf != is_v_wf:
            if is_u_wf: raw_merge_map[id_v] = id_u
            else: raw_merge_map[id_u] = id_v
        else:
            if counts[u] >= counts[v]:
                raw_merge_map[id_v] = id_u
            else:
                raw_merge_map[id_u] = id_v

    del drop_src, drop_tgt, components, counts
    gc.collect()

    # Flatten chains: A → B → C becomes A → C
    log(" -> Flattening map chains...")
    def find_root(pid):
        path = []
        while pid in raw_merge_map:
            path.append(pid)
            pid = raw_merge_map[pid]
        for node in path:
            raw_merge_map[node] = pid
        return pid

    final_merge_map = {}
    for original_id in list(raw_merge_map.keys()):
        root_id = find_root(original_id)
        if original_id != root_id:
            final_merge_map[original_id] = root_id

    del raw_merge_map
    gc.collect()

    log(f"[STATS] FINAL MERGE={len(final_merge_map)}")

    if final_merge_map:
        pl.DataFrame({
            "person_id": list(final_merge_map.keys()),
            "new_id": list(final_merge_map.values())
        }).write_csv(os.path.join(OUTPUT_FINAL_DIR, "merge_mapping.csv"))

    del idx_to_id, ids
    gc.collect()
    log(f"[TIME] STEP5: {time.time()-t4:.1f}s")

    # STEP 6 MP
    t5 = time.time()
    log(f"[STEP 6] Applying to Parquet files...")
    files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.parquet"))
    tasks = [(f, final_merge_map) for f in files]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(NUM_WORKERS_IO) as pool:
        list(tqdm(pool.imap_unordered(process_file_apply, tasks), total=len(tasks)))

    log(f"[TOTAL TIME] {time.time()-t_total:.1f}s")
    log(f"[DONE] Output dir: {OUTPUT_FINAL_DIR}")

if __name__ == "__main__":
    run_pipeline()

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

# --- CONFIGURATION ---
WEBFACE_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_public5m/centers.parquet"
CRAWL_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_after_merge_webface/cleaned_crawl_centers.parquet" 
INPUT_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/v1_ivf_real" 
OUTPUT_FINAL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real_only_new_id_after_merge_webface_public"


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

# def process_file_apply(args):
#     file_path, drop_set, map_df = args
#     try:
#         df = pl.read_parquet(file_path)
#         if df.height == 0: 
#             del df; return
        
#         # Apply Offset
#         df = df.with_columns(pl.col("person_id") + OFFSET_THRESHOLD)

#         # Drop bad IDs
#         if drop_set:
#             df = df.filter(pl.col("person_id").is_in(valid_ids_set))
#         if df.height == 0: 
#             del df; return

#         # Remap IDs (Internal Merge)
#         if map_df is not None:
#             df = (
#                 df.join(map_df, on="person_id", how="left")
#                 .with_columns(pl.col("new_id").fill_null(pl.col("person_id")).alias("person_id"))
#                 .drop("new_id")
#             )
        
#         out_name = os.path.basename(file_path)
#         df.write_parquet(os.path.join(OUTPUT_FINAL_DIR, out_name))
#         del df
#     except Exception as e:
#         print(f"[ERR] File {file_path}: {e}")


def process_file_apply(args):
    # args: file_path, valid_set (Thay vì drop_set), map_df
    file_path, valid_set, map_df = args 
    try:
        df = pl.read_parquet(file_path)
        if df.height == 0: 
            del df; return
        
        # 2. [WHITELIST] CHỈ GIỮ LẠI ID CÓ TRONG VALID_SET
        # (Loại bỏ ID rác, ID nhiễu, ID bị drop)
        if valid_set is not None:
            df = df.filter(pl.col("person_id").is_in(valid_set))
        
        if df.height == 0: 
            del df; return

        # 3. Remap ID (Internal Merge)
        if map_df is not None:
            df = (
                df.join(map_df, on="person_id", how="left")
                .with_columns(pl.col("new_id").fill_null(pl.col("person_id")).alias("person_id"))
                .drop("new_id")
            )
        
        # 4. [THÊM] Unique để tránh trùng lặp đường dẫn ảnh (do merge ID)
        # Giả sử cột đường dẫn ảnh là 'aligned_s3_path', nếu tên khác bạn sửa lại nhé
        # df = df.unique(subset=["aligned_s3_path"]) 
        # Nếu không nhớ tên cột ảnh thì dùng unique() toàn bộ dòng:
        df = df.unique()

        out_name = os.path.basename(file_path)
        df.write_parquet(os.path.join(OUTPUT_FINAL_DIR, out_name))
        del df
    except Exception as e:
        print(f"[ERR] File {file_path}: {e}")

def run_pipeline():
    t_total = time.time()
    log(f"[START] PID={os.getpid()}")
    
    # --- STEP 1: Load Data ---
    log(f"[STEP 1] Loading WebFace & Crawl data...")
    df_w = pl.read_parquet(WEBFACE_PATH).select(["person_id", "img_count", "embedding_center"])
    print(len(df_w))
    print(len(df_w['person_id'].unique()))
    df_c = pl.read_parquet(CRAWL_PATH).select(["person_id", "img_count", "embedding_center"])
    df_c = df_c.unique(subset=["person_id"], keep="first")
    print(len(df_c))
    print(len(df_c['person_id'].unique()))
    # Lưu bản copy của df_c để dùng cho Step 5.5
    df_c_orig = df_c.clone() 
    
    df_centers = pl.concat([df_w, df_c])
    ids = df_centers["person_id"].to_list()
    counts = df_centers["img_count"].to_numpy()
    vecs = np.stack(df_centers["embedding_center"].to_numpy()).astype('float32')
    
    del df_w, df_c, df_centers
    idx_to_id = {i: pid for i, pid in enumerate(ids)}
    
    # --- STEP 2: FAISS Search ---
    log(f"[STEP 2] Building IVF Index & Range Search...")
    quantizer = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = N_PROBE
    
    os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_SAVE_PATH)
    
    lims, D, I = index.range_search(vecs, LOWER_THR)
    lims, D, I = lims.astype('int64'), D.astype('float32'), I.astype('int64')
    
    del quantizer, index, vecs
    gc.collect()

    # --- STEP 3: Edge Processing ---
    log("[STEP 3] Processing Edges...")
    sources = np.repeat(np.arange(len(ids)), np.diff(lims))
    targets = I
    scores = D
    
    mask_valid = sources < targets 
    sources, targets, scores = sources[mask_valid], targets[mask_valid], scores[mask_valid]
    
    mask_merge = scores > UPPER_THR
    mask_drop = (scores > LOWER_THR) & (scores <= UPPER_THR)
    
    merge_edges = np.column_stack((sources[mask_merge], targets[mask_merge]))
    drop_src, drop_tgt = sources[mask_drop], targets[mask_drop]
    
    del sources, targets, scores, mask_valid, mask_merge, mask_drop, D, I
    gc.collect()

    # --- STEP 4: Graph Clustering ---
    log(f"[STEP 4] iGraph processing...")
    if len(merge_edges) > 0:
        g = ig.Graph(n=len(ids), edges=merge_edges, directed=False)
        components = g.connected_components(mode="weak")
        del g
    else:
        components = []
    gc.collect()

    # --- STEP 5: Resolve Conflicts ---
    log("[STEP 5] Resolving (Crawl vs WebFace Deduplication)...")
    real_drop_ids = set()
    real_merge_map = {}

    for comp in components:
        if len(comp) < 1: continue
        
        wf_nodes = [n for n in comp if idx_to_id[n] < OFFSET_THRESHOLD]
        crawl_nodes = [n for n in comp if idx_to_id[n] >= OFFSET_THRESHOLD]

        if wf_nodes:
            for n in crawl_nodes:
                real_drop_ids.add(idx_to_id[n])
        else:
            if len(crawl_nodes) > 1:
                leader = max(crawl_nodes, key=lambda x: (counts[x], -x))
                leader_id = idx_to_id[leader]
                for n in crawl_nodes:
                    node_id = idx_to_id[n]
                    if node_id != leader_id:
                        real_merge_map[node_id] = leader_id

    for u, v in zip(drop_src, drop_tgt):
        id_u, id_v = idx_to_id[u], idx_to_id[v]
        is_u_wf, is_v_wf = id_u < OFFSET_THRESHOLD, id_v < OFFSET_THRESHOLD
        
        if is_u_wf != is_v_wf:
            if is_u_wf: real_drop_ids.add(id_v)
            else: real_drop_ids.add(id_u)

    log(f"[STATS] DROP={len(real_drop_ids)} | INTERNAL MERGE={len(real_merge_map)}")

    # Save metadata
    with open(os.path.join(OUTPUT_FINAL_DIR, "drop_list_id.txt"), "w") as f:
        for pid in real_drop_ids: f.write(f"{pid}\n")
    
    map_df = None
    if real_merge_map:
        map_df = pl.DataFrame({
            "person_id": list(real_merge_map.keys()), 
            "new_id": list(real_merge_map.values())
        })
        map_df.write_csv(os.path.join(OUTPUT_FINAL_DIR, "merge_mapping.csv"))

    # --- STEP 5.5: Save Cleaned Crawl Center (NEW) ---
    log("[STEP 5.5] Saving Cleaned Crawl Centers...")
    is_already_offset = df_c_orig["person_id"][0] >= OFFSET_THRESHOLD
    is_valid = df_c_orig["person_id"][0] <= OFFSET_THRESHOLD*2
    
    if is_valid == False:
        print("Person ID > 2*OFFSET => Invalid => Stop")
    else:
        print("Person ID < 2* OFFSET => Valid => Continue")
        
    if not is_already_offset:
        print("Need + Offset")
        df_c_clean = df_c_orig.with_columns(pl.col("person_id") + OFFSET_THRESHOLD) 
    else:
        print("Dont need + offset")
        df_c_clean = df_c_orig.clone()
    
    df_c_clean = df_c_clean.filter(~pl.col("person_id").is_in(real_drop_ids))

    if map_df is not None:
        df_c_clean = (
            df_c_clean.join(map_df, on="person_id", how="left")
            .with_columns(pl.col("new_id").fill_null(pl.col("person_id")).alias("person_id"))
            .drop("new_id")
        )

    df_c_clean = df_c_clean.group_by("person_id").agg([
        pl.col("img_count").sum(),
        pl.col("embedding_center").first() 
    ])

    log(f"Original Crawl: {df_c_orig.height} | After Clean: {df_c_clean.height}")
    # clean_crawl_path = os.path.join(OUTPUT_FINAL_DIR, "cleaned_crawl_centers.parquet")
    df_c_clean.write_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_after_merge_webface_public/cleaned_crawl_centers.parquet")
    # log(f"Saved cleaned crawl centers to: {clean_crawl_path}")

    valid_ids_set = set(df_c_clean["person_id"].to_list())

    del df_c_orig, df_c_clean, components, counts, idx_to_id, ids
    gc.collect()

    # --- STEP 6: Apply to All Files ---
    log(f"[STEP 6] Applying to Parquet files...")
    files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.parquet"))
    # tasks = [(f, real_drop_ids, map_df) for f in files]
    tasks = [(f, valid_ids_set, map_df) for f in files]
    
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(NUM_WORKERS_IO) as pool:
        list(tqdm(pool.imap_unordered(process_file_apply, tasks), total=len(tasks)))
    
    log(f"[TOTAL TIME] {time.time()-t_total:.1f}s")
    log(f"[DONE] Output dir: {OUTPUT_FINAL_DIR}")

if __name__ == "__main__":
    run_pipeline()
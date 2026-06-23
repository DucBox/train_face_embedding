import os
import glob
import numpy as np
import polars as pl
import faiss
import igraph as ig
import multiprocessing
from tqdm import tqdm

# --- CONFIG ---
CENTERS_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_results/template_v1/centers.parquet"
INPUT_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/dbscan_results/dbscan_v1" # Folder chứa parquet sau DBSCAN
OUTPUT_FINAL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_test/v1_ivf"

LOWER_THR = 0.43
UPPER_THR = 0.56
DIMENSION = 512
N_CLUSTERS = 4096 
N_PROBE = 64
NUM_WORKERS_IO = 16

NUM_DEBUG_IDS = 100000 

def process_file_apply(args):
    file_path, drop_set, map_df = args
    try:
        df = pl.read_parquet(file_path)
        if df.height == 0: return
        
        if drop_set:
            df = df.filter(~pl.col("person_id").is_in(drop_set))
        if df.height == 0: return

        if map_df is not None:
            df = (
                df.join(map_df, on="person_id", how="left")
                  .with_columns(
                      pl.col("new_id").fill_null(pl.col("person_id")).alias("person_id")
                  )
                  .drop("new_id")
            )
        
        out_name = os.path.basename(file_path)
        df.write_parquet(os.path.join(OUTPUT_FINAL_DIR, out_name))
    except Exception as e:
        print(f"[ERR] File {file_path}: {e}")

def run_pipeline():
    # --- STEP 1: LOAD DATA ---
    print(f"[STEP 1] Loading Centers from {CENTERS_PATH}...")
    df = pl.read_parquet(CENTERS_PATH)
    
    # === DEBUG LOGIC ===
    if NUM_DEBUG_IDS is not None:
        print(f"!!! DEBUG MODE ON: Processing only first {NUM_DEBUG_IDS} IDs !!!")
        df = df.head(NUM_DEBUG_IDS)

    ids = df["person_id"].to_list()
    counts = df["img_count"].to_numpy()
    vecs = np.stack(df["embedding_center"].to_numpy()).astype('float32')
    
    idx_to_id = {i: pid for i, pid in enumerate(ids)}
    print(f" -> Loaded {len(ids)} centers.")

    # --- STEP 2: FAISS SEARCH ---
    # Tự động giảm N_CLUSTERS nếu dữ liệu test quá ít (để tránh lỗi train IVF)
    actual_clusters = min(N_CLUSTERS, int(len(ids) / 30)) 
    if actual_clusters < 1: actual_clusters = 1
    
    print(f"[STEP 2] Building FAISS Index (IVF{actual_clusters})...")
    quantizer = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, actual_clusters, faiss.METRIC_INNER_PRODUCT)
    
    print(" -> Training Index...")
    index.train(vecs)
    print(" -> Adding Vectors...")
    index.add(vecs)
    index.nprobe = N_PROBE
    
    print(f" -> Running Range Search (> {LOWER_THR})...")
    lims, D, I = index.range_search(vecs, LOWER_THR)
    
    lims = lims.astype('int64')
    D = D.astype('float32')
    I = I.astype('int64')
    
    # --- STEP 3: VECTORIZATION & GRAPH ---
    print("[STEP 3] Converting Search Results to Edges (NumPy)...")
    
    # Numpy diff và repeat giờ sẽ hoạt động ổn định với int64
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
    
    print(f" -> Found {len(merge_edges)} merge edges and {len(drop_src)} drop candidates.")

    print("[STEP 4] Solving Graph Components (iGraph)...")
    # Handle trường hợp test 1000 ID mà không có cạnh nào
    if len(merge_edges) > 0:
        g = ig.Graph(len(ids), edges=merge_edges, directed=False)
        components = g.connected_components(mode="weak")
    else:
        components = []

    print(" -> Resolving Merges...")
    merge_map_idx = {i: i for i in range(len(ids))}
    leader_counts = counts.copy()

    for comp in components:
        if len(comp) < 2: continue
        leader = max(comp, key=lambda x: (counts[x], -x))
        total_imgs = sum(counts[node] for node in comp)
        leader_counts[leader] = total_imgs
        for node in comp:
            merge_map_idx[node] = leader

    print(" -> Resolving Drops...")
    final_drop_indices = set()
    
    for u, v in zip(drop_src, drop_tgt):
        lu, lv = merge_map_idx[u], merge_map_idx[v]
        if lu == lv: continue
        if leader_counts[lu] > leader_counts[lv]:
            final_drop_indices.add(lv)
        elif leader_counts[lv] > leader_counts[lu]:
            final_drop_indices.add(lu)
        else:
            final_drop_indices.add(max(lu, lv))

    print("[STEP 5] Finalizing ID Maps...")
    real_drop_ids = set()
    for idx, leader in merge_map_idx.items():
        if leader in final_drop_indices:
            real_drop_ids.add(idx_to_id[idx])
            
    real_merge_map = {}
    for idx, leader in merge_map_idx.items():
        orig_id = idx_to_id[idx]
        if orig_id in real_drop_ids: continue
        if idx != leader:
            real_merge_map[orig_id] = idx_to_id[leader]

    print(f" -> STATS: DROP {len(real_drop_ids)} IDs | MERGE {len(real_merge_map)} IDs")

    with open(os.path.join(OUTPUT_FINAL_DIR, "drop_list_id.txt"), "w") as f:
        for pid in real_drop_ids:
            f.write(f"{pid}\n")

    if real_merge_map:
        audit_df = pl.DataFrame({
            "og_id": list(real_merge_map.keys()),
            "new_id": list(real_merge_map.values())
        })
        audit_df.write_csv(os.path.join(OUTPUT_FINAL_DIR, "merge_mapping.csv"))

    print("-> Saved logs file")

    # --- STEP 6: APPLY ---
    print(f"[STEP 6] Applying changes to Parquet files...")
    os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.parquet"))
    
    map_df = None
    if real_merge_map:
        map_df = pl.DataFrame({
            "person_id": list(real_merge_map.keys()), 
            "new_id": list(real_merge_map.values())
        })

    tasks = [(f, real_drop_ids, map_df) for f in files]
    
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(NUM_WORKERS_IO) as pool:
        list(tqdm(pool.imap_unordered(process_file_apply, tasks), total=len(tasks)))

    print("\n[DONE] Pipeline Finished Successfully.")

if __name__ == "__main__":
    run_pipeline()
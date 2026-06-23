import os
import glob
import mxnet as mx
import numpy as np
import polars as pl
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm

# --- CONFIG ---
CRAWL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/re_index_v1_shards"
WEBFACE_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/webface42m_synthetic6m/train.rec"
PUBLIC_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_reindex/train.rec"
OUTPUT_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id/person_reindex_id_by_img_count.parquet"
NUM_WORKERS = 16

def analyze_shard(rec_path):
    idx_path = rec_path[:-3] + 'idx'
    if not os.path.exists(idx_path): return Counter()

    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    stats = Counter()

    # LOGIC: Ưu tiên check Header (cho WebFace), Fallback về Keys (cho Crawl Shards)
    try:
        header0, _ = mx.recordio.unpack(record.read_idx(0))
        if header0.flag > 0:
            # File chuẩn InsightFace (WebFace): Đọc từ 1 -> Header Limit
            iterator = range(1, int(header0.label[0]))
        else:
            # File Shard thường (Crawl): Đọc toàn bộ keys
            iterator = record.keys
    except:
        iterator = record.keys

    for idx in iterator:
        try:
            s = record.read_idx(idx)
            if s is None: continue
            header, _ = mx.recordio.unpack(s)
            
            label = header.label
            if isinstance(label, (np.ndarray, list, tuple)):
                label = label[0]
            
            stats[int(label)] += 1
        except: continue
        
    return stats

def main():
    # 1. Gom file input
    files = sorted(glob.glob(os.path.join(CRAWL_DIR, "*.rec")))
    
    if os.path.exists(WEBFACE_PATH):
        files.append(WEBFACE_PATH)
    elif os.path.exists(WEBFACE_PATH + ".rec"):
        files.append(WEBFACE_PATH + ".rec")

    if os.path.exists(PUBLIC_PATH):
        files.append(PUBLIC_PATH)
    elif os.path.exists(PUBLIC_PATH + ".rec"):
        files.append(PUBLIC_PATH + ".rec")
    
    
    print(f"Processing {len(files)} files (Mixed Crawl & WebFace)...")

    # 2. Chạy Multiprocessing
    total_stats = Counter()
    workers = min(NUM_WORKERS, len(files))

    with Pool(workers) as pool:
        for part_stats in tqdm(pool.imap_unordered(analyze_shard, files), total=len(files)):
            total_stats.update(part_stats)

    # 3. In tổng quan
    print("-" * 30)
    print(f"TOTAL IMAGES : {sum(total_stats.values()):,}")
    print(f"TOTAL IDS    : {len(total_stats):,}")
    print("-" * 30)

    # 4. Lưu Parquet
    # print(f"Saving to {OUTPUT_PARQUET}...")
    # df = pl.DataFrame({
    #     "person_id": list(total_stats.keys()),
    #     "img_count": list(total_stats.values())
    # }).sort("person_id")
    
    # df.write_parquet(OUTPUT_PARQUET)
    # print("[DONE]")

if __name__ == "__main__":
    main()
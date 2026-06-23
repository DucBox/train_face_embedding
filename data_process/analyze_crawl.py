import os
import glob
import mxnet as mx
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

CRAWL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id/re_index_v1_shards"
NUM_WORKERS = 16

def analyze_shard(rec_path):
    idx_path = rec_path[:-3] + 'idx'
    if not os.path.exists(idx_path): return 0, set()

    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    local_count = 0
    local_ids = set()

    for idx in record.keys:
        header, _ = mx.recordio.unpack(record.read_idx(idx))
        label = header.label
        if isinstance(label, (np.ndarray, list, tuple)):
            label = label[0]
        
        local_ids.add(int(label))
        local_count += 1
        
    return local_count, local_ids

def main():
    files = sorted(glob.glob(os.path.join(CRAWL_DIR, "*.rec")))
    print(f"Found {len(files)} shards. Analyzing with {NUM_WORKERS} workers...")

    total_images = 0
    total_ids = set()

    with Pool(NUM_WORKERS) as pool:
        for count, ids in tqdm(pool.imap_unordered(analyze_shard, files), total=len(files)):
            total_images += count
            total_ids.update(ids)

    print("-" * 30)
    print(f"CRAWL DATASET STATS")
    print("-" * 30)
    print(f"Total Images : {total_images:,}")
    print(f"Total Persons: {len(total_ids):,}")
    print("-" * 30)

if __name__ == "__main__":
    main()
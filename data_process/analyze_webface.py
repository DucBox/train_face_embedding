# import os
# import glob
# import mxnet as mx
# import numpy as np
# from multiprocessing import Pool
# from tqdm import tqdm

# # --- CONFIG ---
# # Điền đường dẫn file .rec cụ thể HOẶC folder chứa các file .rec
# WEBFACE_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M/train.rec" 
# NUM_WORKERS = 16

# def analyze_shard(rec_path):
#     idx_path = rec_path[:-3] + 'idx'
#     if not os.path.exists(idx_path): 
#         print("No path found")
#         return 0, set()

#     record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
#     local_count = 0
#     local_ids = set()

#     for idx in record.keys:
#         header, _ = mx.recordio.unpack(record.read_idx(idx))
#         label = header.label
#         if isinstance(label, (np.ndarray, list, tuple)):
#             label = label[0]
        
#         local_ids.add(int(label))
#         local_count += 1
        
#     return local_count, local_ids

# def main():
#     if os.path.isdir(WEBFACE_PATH):
#         files = sorted(glob.glob(os.path.join(WEBFACE_PATH, "*.rec")))
#     elif os.path.isfile(WEBFACE_PATH):
#         files = [WEBFACE_PATH]
#     elif os.path.isfile(WEBFACE_PATH + ".rec"):
#         files = [WEBFACE_PATH + ".rec"]
#     else:
#         print(f"[ERR] Path not found: {WEBFACE_PATH}")
#         return

#     print(f"Found {len(files)} file(s). Scanning...")

#     total_images = 0
#     total_ids = set()
    
#     workers = min(NUM_WORKERS, len(files))

#     with Pool(workers) as pool:
#         for count, ids in tqdm(pool.imap_unordered(analyze_shard, files), total=len(files)):
#             total_images += count
#             total_ids.update(ids)

#     print("-" * 30)
#     print(f"WEBFACE STATS")
#     print("-" * 30)
#     print(f"Total Images : {total_images:,}")
#     print(f"Total Persons: {len(total_ids):,}")
#     print("-" * 30)

# if __name__ == "__main__":
#     main()


import os
import glob
import mxnet as mx
import numpy as np
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm

# --- CONFIG ---
WEBFACE_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/webface42m_synthetic6m/train.rec" 
NUM_WORKERS = 16

def analyze_webface_shard(rec_path):
    idx_path = rec_path[:-3] + 'idx'
    if not os.path.exists(idx_path): return Counter()

    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    stats = Counter()

    # --- LOGIC CHUẨN: Check Header tại Index 0 ---
    try:
        # Đọc index 0
        s = record.read_idx(0)
        header0, _ = mx.recordio.unpack(s)
        
        if header0.flag > 0:
            print("Header0 > 0")
            max_idx = int(header0.label[0])
            iterator = range(1, max_idx)
        else:
            print("Header0 < 0")
            iterator = record.keys
    except Exception:
        iterator = record.keys

    # Loop đếm
    for idx in iterator:
        try:
            s = record.read_idx(idx)
            if s is None: continue
            
            header, _ = mx.recordio.unpack(s)
            
            label = header.label
            if isinstance(label, (np.ndarray, list, tuple)):
                label = label[0]
            
            stats[int(label)] += 1
        except:
            continue
            
    return stats

def main():
    # 1. Xác định input
    if os.path.isdir(WEBFACE_PATH):
        files = sorted(glob.glob(os.path.join(WEBFACE_PATH, "*.rec")))
    elif os.path.isfile(WEBFACE_PATH):
        files = [WEBFACE_PATH]
    elif os.path.isfile(WEBFACE_PATH + ".rec"):
        files = [WEBFACE_PATH + ".rec"]
    else:
        print(f"[ERR] File not found: {WEBFACE_PATH}")
        return

    print(f"Analyzing WebFace ({len(files)} files)...")

    # 2. Chạy Multiprocessing
    # Lưu ý: Nếu chỉ có 1 file to, nó sẽ chỉ chạy 1 core.
    total_stats = Counter()
    workers = min(NUM_WORKERS, len(files))
    
    with Pool(workers) as pool:
        for file_stat in tqdm(pool.imap_unordered(analyze_webface_shard, files), total=len(files)):
            total_stats.update(file_stat)

    # 3. Kết quả
    print("-" * 30)
    print(f"WEBFACE STATS (InsightFace Logic)")
    print("-" * 30)
    print(f"TOTAL IMAGES   : {sum(total_stats.values()):,}")
    print(f"TOTAL PERSONS  : {len(total_stats):,}")
    print("-" * 30)

if __name__ == "__main__":
    main()
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import gc
import multiprocessing
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from tqdm import tqdm

INPUT_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/face-embedding-normalize/face-embedding-normalize"
OUTPUT_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/dbscan_results/dbscan_v3"

DBSCAN_EPS = 0.47
DBSCAN_MIN_SAMPLES = 3
DBSCAN_METRIC = 'cosine'
NUM_WORKERS = 16

def process_person_group(df):
    # Hàm này chạy trong worker process
    try:
        if len(df) < DBSCAN_MIN_SAMPLES:
            return None
            
        X = np.array(df["embedding_normalized"].to_list())
        
        # DBSCAN
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric=DBSCAN_METRIC, n_jobs=1)
        labels = clustering.fit_predict(X)
        
        # Chọn cụm tốt nhất
        valid_indices = np.where(labels != -1)[0]
        if len(valid_indices) == 0: return None
        
        valid_labels = labels[valid_indices]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        if len(unique_labels) == 0: return None
            
        best_cluster_label = unique_labels[np.argmax(counts)]
        
        # Filter (Logic an toàn tránh deadlock numpy mask)
        return (
            df.with_columns(pl.Series("temp_label", labels))
              .filter(pl.col("temp_label") == best_cluster_label)
              .drop("temp_label")
        )
    except Exception:
        return None

def main():
    # 1. Lấy danh sách sub-folders
    # Giả sử cấu trúc: root/output_embedding_normalize_1, ..._2
    sub_folders = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    ctx = multiprocessing.get_context("spawn")

    # 2. Loop qua từng folder (Tuần tự để tiết kiệm RAM)
    for folder_path in tqdm(sub_folders, desc="Processing Folders"):
        if not os.path.isdir(folder_path): continue
        
        folder_name = os.path.basename(folder_path)
        print(f"\nLoading folder: {folder_name}...")
        
        try:
            # Load TOÀN BỘ parquet trong sub-folder này
            # Vì 1 user nằm trọn trong folder này, nhưng có thể rải rác ở nhiều file con
            # Nên bắt buộc phải load hết folder này vào mới group được.
            print("Start")
            cols = ['person_id', 'embedding_normalized', 'aligned_s3_path']
            df = pl.read_parquet(folder_path, columns = cols)
            print("Loaded folder")
            if df.height == 0: 
                print("Nothing inside folders")
                continue
            else:
                print("Splitting users into groups")
            # Chia nhóm user
            person_dfs = df.partition_by("person_id", maintain_order=False)
            del df # Giải phóng dataframe to ngay lập tức
            gc.collect()
            
            print(f"  - Users: {len(person_dfs)}")
            print(f"  - Processing with {NUM_WORKERS} workers...")
            
            folder_results = []
            
            # Xử lý song song
            with ctx.Pool(NUM_WORKERS) as pool:
                for res in pool.imap_unordered(process_person_group, person_dfs, chunksize=100):
                    if res is not None:
                        folder_results.append(res)
            
            # Ghi kết quả của Folder này
            if folder_results:
                out_df = pl.concat(folder_results)
                out_file = os.path.join(OUTPUT_ROOT, f"cleaned_{folder_name}.parquet")
                out_df.write_parquet(out_file)
                print(f"  - Saved: {out_file} ({len(out_df)} rows)")
            
            # Cleanup triệt để trước khi qua folder mới
            del person_dfs, folder_results
            gc.collect()

        except Exception as e:
            print(f"ERR processing {folder_name}: {e}")

if __name__ == "__main__":
    main()
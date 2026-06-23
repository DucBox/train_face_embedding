import os
import glob
import shutil
import numpy as np
import polars as pl
from sklearn.preprocessing import normalize
import multiprocessing
from tqdm import tqdm

INPUT_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/dbscan_results/dbscan_v1"
OUTPUT_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_results/template_v1/centers.parquet"
TEMP_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/temp_shards"
SHARD_OUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/temp_shard_results"

NUM_WORKERS = 32
NUM_SHARDS = 64  # Tăng lên 128/256 nếu vẫn OOM

# Chỉ tính template cho person_id >= ngưỡng này.
# Dùng khi đã merge với WebFace (ID < 3M): các ID đó k cần tính lại.
# Set None để tính tất cả.
MIN_PERSON_ID = None  # int hoặc None

def get_shard_id(person_id: int) -> int:
    return person_id % NUM_SHARDS

def process_file_partial(file_path):
    """Phase 1 worker: tính sum_vectors cục bộ, spill ra đĩa theo shard."""
    try:
        df = pl.read_parquet(file_path, columns=["person_id", "embedding_normalized"])
        if df.height == 0:
            return True

        if MIN_PERSON_ID is not None:
            df = df.filter(pl.col("person_id") >= MIN_PERSON_ID)
        if df.height == 0:
            return True

        df = df.sort("person_id")
        counts_df = df.group_by("person_id", maintain_order=True).agg(pl.len().alias("img_count"))

        unique_ids = counts_df["person_id"].to_list()
        counts = counts_df["img_count"].to_numpy()
        matrix = np.stack(df["embedding_normalized"].to_numpy()).astype("float32")

        indices = np.zeros(len(counts), dtype=int)
        np.cumsum(counts[:-1], out=indices[1:])
        sum_vectors = np.add.reduceat(matrix, indices)

        shard_ids = np.array([get_shard_id(pid) for pid in unique_ids])
        base_name = os.path.basename(file_path)

        for s_id in np.unique(shard_ids):
            mask = shard_ids == s_id
            shard_dir = os.path.join(TEMP_DIR, f"shard_{s_id}")
            os.makedirs(shard_dir, exist_ok=True)
            pl.DataFrame({
                "person_id": np.array(unique_ids)[mask],
                "img_count": counts[mask],
                "sum_vec": sum_vectors[mask].tolist(),
            }).write_parquet(os.path.join(shard_dir, f"partial_{base_name}"))

        return True
    except Exception as e:
        print(f"Err {file_path}: {e}")
        return False


def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    for d in (TEMP_DIR, SHARD_OUT_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(INPUT_ROOT, "*.parquet")))
    print(f"Found {len(parquet_files)} files")

    # --- PHASE 1: MAP ---
    # Workers spill partial sum_vectors ra đĩa, main process không giữ data nào trong RAM
    print("Phase 1: Map & Disk Spill...")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_file_partial, parquet_files), total=len(parquet_files)))

    # --- PHASE 2: REDUCE ---
    # Xử lý từng shard tuần tự → peak RAM = kích thước 1 shard, không phải toàn bộ data
    print("\nPhase 2: Reduce by Shard...")
    total_centers = 0

    for s_id in tqdm(range(NUM_SHARDS), desc="Reducing shards"):
        shard_dir = os.path.join(TEMP_DIR, f"shard_{s_id}")
        if not os.path.exists(shard_dir):
            continue

        shard_files = glob.glob(os.path.join(shard_dir, "*.parquet"))
        if not shard_files:
            continue

        df = pl.scan_parquet(shard_files).collect()
        if df.height == 0:
            continue

        df = df.sort("person_id")
        unique_pids, start_indices = np.unique(df["person_id"].to_numpy(), return_index=True)
        counts_arr = df["img_count"].to_numpy()
        # sum_vec là list column → dùng to_list() rồi np.array, không dùng to_numpy()
        vecs_arr = np.array(df["sum_vec"].to_list(), dtype="float32")

        final_counts = np.add.reduceat(counts_arr, start_indices)
        final_sum_vectors = np.add.reduceat(vecs_arr, start_indices)
        center_vectors = normalize(final_sum_vectors, axis=1)

        # Ghi kết quả shard ra file riêng, không giữ trong RAM
        pl.DataFrame({
            "person_id": unique_pids,
            "embedding_center": center_vectors.tolist(),
            "img_count": final_counts,
        }).write_parquet(os.path.join(SHARD_OUT_DIR, f"part_{s_id:04d}.parquet"))

        total_centers += len(unique_pids)

        # Dọn temp input của shard ngay để giải phóng disk
        shutil.rmtree(shard_dir)

    # --- PHASE 3: MERGE SHARD OUTPUTS ---
    # Centers nhỏ hơn nhiều so với input (1 vector/person thay vì N ảnh/person)
    # scan_parquet đọc lazy, collect một lần rồi ghi output cuối
    print("\nPhase 3: Merging shard outputs...")
    pl.scan_parquet(os.path.join(SHARD_OUT_DIR, "*.parquet")).collect().write_parquet(OUTPUT_FILE)
    shutil.rmtree(SHARD_OUT_DIR)

    print(f"Done. Saved to {OUTPUT_FILE}")
    print(f"Total Centers: {total_centers}")


if __name__ == "__main__":
    main()

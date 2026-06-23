import os
import re
import io
import tarfile
import random
import polars as pl
import boto3
from tqdm import tqdm

# --- CONFIG ---
INPUT_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/v1_ivf_real" 
DEBUG_OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/visual_v1_ivf_real_debug"
NUM_SAMPLES = 1000

# --- S3 CONFIG ---
S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "H?3o0nn4Irej"
BUCKET_NAME = "ttnt"

def parse_s3_info(full_path: str):
    split_token = ".tar/"
    idx = full_path.find(split_token)
    
    if idx == -1:
        return None, None, None
    
    s3_key = full_path[:idx+4]  # .../person_xxxx.tar
    member_name = full_path[idx+5:] # image_name.jpg
    
    tar_name = os.path.basename(s3_key)
    og_id_match = re.search(r"person_(\d+)", tar_name)
    og_id = og_id_match.group(1) if og_id_match else "unknown"
    
    return og_id, s3_key, member_name

def download_images(s3_client, bucket, s3_key, member_names, save_dir):
    """Download tar từ S3, giải nén đúng các file member vào save_dir"""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        file_content = obj['Body'].read()
        
        with tarfile.open(fileobj=io.BytesIO(file_content), mode="r") as tar:
            for member in member_names:
                try:
                    f = tar.extractfile(member)
                    if f:
                        out_path = os.path.join(save_dir, os.path.basename(member))
                        with open(out_path, "wb") as out_f:
                            out_f.write(f.read())
                except KeyError:
                    print(f"[WARN] File {member} not found in {s3_key}")
                    
    except Exception as e:
        print(f"[ERR] Failed to process {s3_key}: {e}")

def main():
    # 1. Load Data & Extract Info
    print(f"Loading data from {INPUT_DATA_DIR}...")
    df = pl.scan_parquet(os.path.join(INPUT_DATA_DIR, "*.parquet"))
    print(len(df))
    # 2. Logic tìm Merge: Gom nhóm theo person_id mới
    print("Finding merged IDs...")
    
    df = df.with_columns(
        pl.col("aligned_s3_path").str.extract(r"person_(\d+)\.tar", 1).alias("og_id")
    )

    # Group by New ID và đếm số lượng og_id unique
    merge_stats = (
        df.group_by("person_id")
        .agg([
            pl.col("og_id").n_unique().alias("n_sources"),
            pl.col("og_id").unique().alias("source_ids"),
            pl.col("aligned_s3_path")
        ])
        .filter(pl.col("n_sources") > 1) # Chỉ lấy thằng nào được gộp từ > 1 nguồn
        .collect()
    )

    if merge_stats.height == 0:
        print("No merged IDs found! Check your data.")
        return

    print(f"Found {merge_stats.height} merged identities.")
    
    # 3. Lấy mẫu ngẫu nhiên
    samples = merge_stats.sample(n=min(NUM_SAMPLES, merge_stats.height), with_replacement=False)
    
    # 4. Init S3 Client
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

    # 5. Download Loop
    print(f"\nDownloading samples to '{DEBUG_OUTPUT_DIR}'...")
    
    for row in tqdm(samples.iter_rows(named=True), total=samples.height):
        new_id = row["person_id"]
        paths = row["aligned_s3_path"]
        
        download_plan = {}
        
        for p in paths:
            og_id, s3_key, member = parse_s3_info(p)
            if not s3_key: continue
            
            if s3_key not in download_plan:
                download_plan[s3_key] = {"og_id": og_id, "members": []}
            download_plan[s3_key]["members"].append(member)

        for s3_key, data in download_plan.items():
            og_id = data["og_id"]
            members = data["members"]
            
            save_dir = os.path.join(DEBUG_OUTPUT_DIR, f"person_{new_id}", f"person_{og_id}")
            os.makedirs(save_dir, exist_ok=True)
            
            download_images(s3, BUCKET_NAME, s3_key, members, save_dir)
            
    print(f"\n[DONE] Check folder '{DEBUG_OUTPUT_DIR}' for visual inspection.")

if __name__ == "__main__":
    main()


# import os
# import re
# import io
# import tarfile
# import polars as pl
# import boto3
# from tqdm import tqdm
# import gc

# # --- CONFIG ---
# INPUT_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/v1_ivf_real"
# DF_ID_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_v1/centers_v2.parquet"
# DEBUG_OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/visual_v1_ivf_real_debug"
# SAMPLES_PER_BIN = 50

# S3_ENDPOINT = "http://s3-data.cyberspace.vn"
# S3_ACCESS_KEY = "ttnt"
# S3_SECRET_KEY = "H?3o0nn4Irej"
# BUCKET_NAME = "ttnt"

# # Bin config: [low, high, label, idx]
# BINS = [
#     (3, 6, "bin_3_5", 0),
#     (6, 21, "bin_6_20", 1),
#     (21, 51, "bin_21_50", 2),
#     (51, 101, "bin_51_100", 3),
#     (101, 501, "bin_101_500", 4),
#     (501, 1001, "bin_501_1000", 5),
#     (1001, float('inf'), "bin_1000_plus", 6)
# ]

# BIN_LABELS = {idx: label for _, _, label, idx in BINS}

# def parse_s3_info(full_path: str):
#     split_token = ".tar/"
#     idx = full_path.find(split_token)
#     if idx == -1:
#         return None, None, None
    
#     s3_key = full_path[:idx+4]  
#     member_name = full_path[idx+5:] 
#     tar_name = os.path.basename(s3_key)
#     og_id_match = re.search(r"person_(\d+)", tar_name)
#     og_id = og_id_match.group(1) if og_id_match else "unknown"
    
#     return og_id, s3_key, member_name

# def download_images(s3_client, bucket, s3_key, member_names, save_dir):
#     try:
#         obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
#         with tarfile.open(fileobj=io.BytesIO(obj['Body'].read()), mode="r") as tar:
#             for member in set(member_names):  # Unique
#                 try:
#                     f = tar.extractfile(member)
#                     if f:
#                         out_path = os.path.join(save_dir, os.path.basename(member))
#                         with open(out_path, "wb") as out_f:
#                             out_f.write(f.read())
#                 except KeyError:
#                     pass
#     except Exception as e:
#         print(f"[ERR] {s3_key}: {e}")

# def main():
#     # 1. Chỉ đọc df_id & binning
#     df_id = pl.read_parquet(DF_ID_PATH)
#     print(f"df_id: {df_id.shape}")
    
#     df_id = df_id.with_columns(
#         pl.when(pl.col("img_count") < 3).then(-1)
#         .when(pl.col("img_count") < 6).then(0)
#         .when(pl.col("img_count") < 21).then(1)
#         .when(pl.col("img_count") < 51).then(2)
#         .when(pl.col("img_count") < 101).then(3)
#         .when(pl.col("img_count") < 501).then(4)
#         .when(pl.col("img_count") < 1001).then(5)
#         .otherwise(6).alias("bin_idx")
#     )
    
#     print("Bin stats:", df_id.filter(pl.col("bin_idx")>=0).group_by("bin_idx").agg(pl.count()).sort("bin_idx"))
    
#     # 2. Sample person_id theo bin (df_id chỉ!)
#     samples_per_bin = {}
#     for bin_idx in range(7):
#         bin_persons = df_id.filter(pl.col("bin_idx") == bin_idx)["original_person_id"]
#         n_sample = min(SAMPLES_PER_BIN, len(bin_persons))
#         sampled_pids = bin_persons.sample(n=n_sample, seed=42).to_list()
#         samples_per_bin[bin_idx] = sampled_pids
#         print(f"bin {bin_idx}: {len(sampled_pids)} persons")
    
#     all_pids = sum(samples_per_bin.values(), [])
#     print(f"Total {len(all_pids)} person_ids to download")
    
#     # 3. Lấy paths LAZY cho sampled pids
#     df_paths = pl.scan_parquet(f"{INPUT_DATA_DIR}/*.parquet").filter(
#         pl.col("person_id").is_in(all_pids)
#     ).collect()
    
#     # 4. Map pid → paths + bin_idx
#     pid_to_bin = {row["person_id"]: row["bin_idx"] for row in df_id.select(["person_id", "bin_idx"]).iter_rows(named=True)}
#     pid_to_paths = {}
#     for row in df_paths.iter_rows(named=True):
#         pid = row["person_id"]
#         if pid not in pid_to_paths:
#             pid_to_paths[pid] = []
#         pid_to_paths[pid].append(row["aligned_s3_path"])
    
#     # 5. Download (same as before)
#     s3 = boto3.client(
#         "s3",
#         endpoint_url=S3_ENDPOINT,
#         aws_access_key_id=S3_ACCESS_KEY,
#         aws_secret_access_key=S3_SECRET_KEY
#     )
#     for bin_idx, pids in samples_per_bin.items():
#         bin_folder = BIN_LABELS[bin_idx]
#         print(f"\n--- Downloading {bin_folder} ---")
        
#         for pid in tqdm(pids, desc=f"{bin_folder}"):
#             bin_dir = os.path.join(DEBUG_OUTPUT_DIR, bin_folder, f"person_{pid}")
#             os.makedirs(bin_dir, exist_ok=True)
            
#             paths = pid_to_paths.get(pid, [])
#             download_plan = {}
#             for p in paths:
#                 og_id, s3_key, member = parse_s3_info(p)
#                 if s3_key:
#                     if s3_key not in download_plan:
#                         download_plan[s3_key] = {"og_id": og_id, "members": set()}
#                     download_plan[s3_key]["members"].add(member)
            
#             for s3_key, data in download_plan.items():
#                 og_dir = os.path.join(bin_dir, f"person_{data['og_id']}")
#                 os.makedirs(og_dir, exist_ok=True)
#                 download_images(s3, BUCKET_NAME, s3_key, list(data["members"]), og_dir)
            
#             gc.collect()
    
#     print("DONE!")


# if __name__ == "__main__":
#     main()

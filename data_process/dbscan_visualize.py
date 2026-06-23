import os
import io
import random
import tarfile
import boto3
import polars as pl
from tqdm import tqdm
import glob
# --- CONFIGURATION ---
INPUT_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/dbscan_results/dbscan_v1/*.parquet"
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/visualize_dbscan"

S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "<S3_SECRET_KEY>"
BUCKET_NAME = "ttnt"

# Define ranges: (min, max, label)
SAMPLING_CONFIG = [
    (3, 5, "low_3-5"),
    (6, 20, "mid_6-20"),
    (21, 50, "high_21-50"),
    (100, 200, "dense_100-200")
]
SAMPLES_PER_GROUP = 50

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def main():
    print(f"[INIT] Loading data from {INPUT_PARQUET}...")
    files = glob.glob(INPUT_PARQUET)
    df = pl.read_parquet(files, columns=["person_id", "aligned_s3_path"])
    
    # 1. Group & Count
    print("[INIT] Analyzing counts...")
    counts = df.group_by("person_id").len()
    
    tasks = []
    
    # 2. Sampling Logic
    for (min_c, max_c, label) in SAMPLING_CONFIG:
        # Filter IDs in range
        candidate_ids = counts.filter(
            (pl.col("len") >= min_c) & (pl.col("len") <= max_c)
        )["person_id"].to_list()
        
        # Random Sample
        if len(candidate_ids) > SAMPLES_PER_GROUP:
            picked_ids = random.sample(candidate_ids, SAMPLES_PER_GROUP)
        else:
            picked_ids = candidate_ids
            
        print(f"   + Group [{label}]: Found {len(candidate_ids)} candidates -> Picked {len(picked_ids)}")
        
        for pid in picked_ids:
            # Get all paths for this person
            paths = df.filter(pl.col("person_id") == pid)["aligned_s3_path"].to_list()
            tasks.append({
                "pid": pid,
                "label": label,
                "paths": paths
            })

    print(f"\n[RUN] Starting download for {len(tasks)} persons (Sequential)...")
    s3 = get_s3_client()
    
    for task in tqdm(tasks, desc="Downloading"):
        pid = task['pid']
        label = task['label']
        paths = task['paths']
        
        # Parse TAR Key from the first path (Assuming all imgs of 1 person are in 1 TAR)
        # Format: path/to/person.tar/image.jpg
        # Split by '.tar/'
        first_path = paths[0]
        if ".tar/" not in first_path:
            print(f"[SKIP] Invalid path format: {first_path}")
            continue
            
        tar_key = first_path.split(".tar/")[0] + ".tar"
        
        # Prepare local folder
        local_folder = os.path.join(OUTPUT_DIR, label, f"person_{pid}")
        os.makedirs(local_folder, exist_ok=True)
        
        # Set of image filenames we want to keep
        target_members = set(p.split(".tar/")[1] for p in paths)
        
        try:
            # Download TAR to Memory
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=tar_key)
            tar_bytes = io.BytesIO(obj['Body'].read())
            
            with tarfile.open(fileobj=tar_bytes, mode='r') as tar:
                for member in tar.getmembers():
                    if member.name in target_members:
                        f = tar.extractfile(member)
                        if f:
                            # Save to local
                            save_path = os.path.join(local_folder, member.name)
                            with open(save_path, "wb") as out_f:
                                out_f.write(f.read())
                                
        except Exception as e:
            print(f"\n[ERR] Failed PID {pid} (Key: {tar_key}): {e}")

    print("\n[DONE] Check folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
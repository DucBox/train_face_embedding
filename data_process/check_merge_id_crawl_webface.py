import os
import io
import re
import glob
import tarfile
import boto3
import polars as pl
from tqdm import tqdm

# --- CONFIG ---
# 1. Đường dẫn Crawl Data (Bộ clean)
CRAWL_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real"

# 2. Đường dẫn Webface Data (Bộ gốc)
WEBFACE_DATA_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/webface42m_embedding_normalize"

# Output folder
DEBUG_OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/merge_crawl_webface_debug/visual_webface_crawl_merge_check"
NUM_SAMPLES = 100

# Ngưỡng phân biệt ID
OFFSET_THRESHOLD = 3000000

# --- S3 CONFIG ---
S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "H?3o0nn4Irej"
BUCKET_NAME = "ttnt"

def parse_s3_info(full_path: str):
    """
    Input: .../person_0734098.tar/0.jpg
    Output: (og_id_string, s3_key, member_name)
    """
    split_token = ".tar/"
    idx = full_path.find(split_token)
    
    if idx == -1:
        return None, None, None
    
    s3_key = full_path[:idx+4]
    member_name = full_path[idx+5:]
    
    tar_name = os.path.basename(s3_key)
    # Lấy ID gốc từ tên file tar (bất kể là crawl hay webface đều có format person_XXXX.tar)
    og_id_match = re.search(r"person_(\d+)", tar_name)
    og_id = og_id_match.group(1) if og_id_match else "unknown"
    
    return og_id, s3_key, member_name

def download_images(s3_client, bucket, s3_key, member_names, save_dir):
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
                    pass
    except Exception as e:
        print(f"[ERR] Failed {s3_key}: {e}")

def main():
    print("--- STEP 1: LOADING DATA ---")
    
    # 1. Load Crawl Data -> Gán nhãn 'crawl'
    print(f"Loading Crawl Data...")
    lf_crawl = pl.scan_parquet(os.path.join(CRAWL_DATA_DIR, "*.parquet")).select([
        "person_id", "aligned_s3_path"
    ]).with_columns(pl.lit("crawl").alias("data_source"))

    # 2. Load WebFace Data -> Gán nhãn 'webface'
    webface_pattern = os.path.join(WEBFACE_DATA_ROOT, "*", "*.parquet")
    print(f"Loading WebFace Data...")
    lf_webface = pl.scan_parquet(webface_pattern).select([
        "person_id", "aligned_s3_path"
    ]).with_columns(pl.lit("webface").alias("data_source"))

    # 3. Gộp lại
    lf_combined = pl.concat([lf_crawl, lf_webface])

    # --- STEP 2: LOGIC FILTER & MERGE ---
    print(f"--- STEP 2: FINDING CRAWL DATA MERGED INTO WEBFACE (ID < {OFFSET_THRESHOLD}) ---")
    
    merge_stats = (
        lf_combined
        # FILTER 1: Chỉ quan tâm đến các ID thuộc dải WebFace (nhỏ hơn 3 triệu)
        .filter(pl.col("person_id") < OFFSET_THRESHOLD)
        
        # Group lại để xem thành phần bên trong
        .group_by("person_id")
        .agg([
            pl.col("data_source").unique().alias("sources"),
            pl.col("data_source").n_unique().alias("n_sources"),
            pl.struct(["data_source", "aligned_s3_path"]).alias("details")
        ])
        
        # FILTER 2: Chỉ lấy những ID nào có sự xuất hiện của cả 2 nguồn (Webface gốc + Crawl bị gộp vào)
        # Hoặc ít nhất là có nguồn 'crawl' rơi vào ID < 3tr này.
        # Logic chặt chẽ nhất: n_sources > 1 (tức là vừa có webface vừa có crawl)
        .filter(pl.col("n_sources") > 1) 
        .collect()
    )

    if merge_stats.height == 0:
        print("No cross-dataset merges found (No Crawl data merged into WebFace IDs).")
        return

    print(f"Found {merge_stats.height} WebFace IDs containing Crawl data.")
    
    # --- STEP 3: SAMPLING & DOWNLOAD ---
    samples = merge_stats.sample(n=min(NUM_SAMPLES, merge_stats.height), with_replacement=False)
    
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

    print(f"\n--- STEP 3: DOWNLOADING TO {DEBUG_OUTPUT_DIR} ---")
    
    for row in tqdm(samples.iter_rows(named=True), total=samples.height):
        new_id = row["person_id"] # Đây là ID < 3 triệu
        details = row["details"]
        
        # Tạo plan download để gom nhóm request S3
        download_plan = {}
        
        for item in details:
            source = item["data_source"]      # 'crawl' hoặc 'webface'
            path = item["aligned_s3_path"]
            
            og_id, s3_key, member = parse_s3_info(path)
            if not s3_key: continue
            
            if s3_key not in download_plan:
                download_plan[s3_key] = {
                    "source": source,
                    "og_id": og_id,
                    "members": []
                }
            download_plan[s3_key]["members"].append(member)

        # Download và lưu vào folder có prefix
        for s3_key, data in download_plan.items():
            source = data["source"]
            og_id = data["og_id"] # ID gốc trong file tar
            members = data["members"]
            
            # CẤU TRÚC FOLDER:
            # person_5234/
            #    ├── webface_person_5234  (Data gốc của webface)
            #    └── crawl_person_7890    (Data crawl bị gộp vào ID 5234 này)
            
            folder_name = f"{source}_person_{og_id}"
            save_dir = os.path.join(DEBUG_OUTPUT_DIR, f"person_{new_id}", folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            download_images(s3, BUCKET_NAME, s3_key, members, save_dir)
            
    print(f"\n[DONE] Check folder '{DEBUG_OUTPUT_DIR}'. Look for 'crawl_' folders inside WebFace IDs.")

if __name__ == "__main__":
    main()
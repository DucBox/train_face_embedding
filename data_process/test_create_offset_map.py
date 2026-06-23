import boto3
import tarfile
import pandas as pd
import time
import io
import multiprocessing
from multiprocessing import Pool

# --- CONFIG ---
BUCKET_NAME = "ttnt"
ROOT_PREFIX = "cv/crawled-datasets/face-google-search/"
TEST_TARGET_FOLDER = "data_0" # Chỉ test trên folder này
OUTPUT_FILE = "test_offset_result.parquet"
NUM_WORKERS = 16 # Test thì 16 là đủ


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="H?3o0nn4Irej",
    )

class S3StreamWrapper:
    def __init__(self, raw_stream):
        self.raw_stream = raw_stream
        self.position = 0

    def read(self, size=-1):
        chunk = self.raw_stream.read(size)
        if chunk:
            self.position += len(chunk)
        return chunk

    def tell(self):
        return self.position

def list_target_tars(bucket, prefix, target_subfolder):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    tar_keys = []
    
    print(f"[INFO] Scanning for tars in {target_subfolder}...")
    # Trick: Thêm target_subfolder vào prefix để scan nhanh hơn
    search_prefix = f"{prefix}{target_subfolder}/"
    
    for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".tar"):
                    tar_keys.append(obj["Key"])
    return tar_keys

def process_single_tar(args):
    bucket, tar_key = args
    results = []
    try:
        s3 = get_s3_client()
        response = s3.get_object(Bucket=bucket, Key=tar_key)
        # Quan trọng: Dùng wrapper để track offset thực tế
        stream_wrapper = S3StreamWrapper(response['Body'])
        
        with tarfile.open(fileobj=stream_wrapper, mode='r|') as tar:
            for member in tar:
                if member.isfile():
                    results.append({
                        'tar_path': tar_key,
                        'member_name': member.name,
                        'start_byte': member.offset_data,
                        'length': member.size
                    })
    except Exception as e:
        print(f"[ERROR] {tar_key}: {e}")
        return []
    return results

def main():
    t0 = time.time()
    
    # 1. Get List
    tar_keys = list_target_tars(BUCKET_NAME, ROOT_PREFIX, TEST_TARGET_FOLDER)
    if not tar_keys:
        print("No files found.")
        return
        
    print(f"[INFO] Found {len(tar_keys)} files in {TEST_TARGET_FOLDER}")
    
    # 2. Run Pool
    tasks = [(BUCKET_NAME, key) for key in tar_keys]
    all_offsets = []
    processed = 0
    
    print(f"[RUN] Starting {NUM_WORKERS} workers...")
    t_start_process = time.time()
    
    with Pool(processes=NUM_WORKERS) as pool:
        for res in pool.imap_unordered(process_single_tar, tasks):
            processed += 1
            if res:
                all_offsets.extend(res)
            
            if processed % 10 == 0:
                print(f"[PROG] {processed}/{len(tar_keys)} files done...", end='\r')

    total_time = time.time() - t_start_process
    print(f"\n[DONE] Processed {processed} tars in {total_time:.2f}s")
    
    # 3. Save & Stats
    if all_offsets:
        df = pd.DataFrame(all_offsets)
        df.to_parquet(OUTPUT_FILE, index=False)
        
        avg_time_per_tar = total_time / processed
        fps = len(all_offsets) / total_time
        
        print("-" * 40)
        print(f"RESULTS for {TEST_TARGET_FOLDER}:")
        print(f" - Total extracted rows: {len(df)}")
        print(f" - Avg time per TAR: {avg_time_per_tar:.3f}s")
        print(f" - Extraction Speed: {fps:.1f} images/s (Virtual)")
        print(f" - Saved to: {OUTPUT_FILE}")
        
        # Estimate cho 9500 folder (Giả sử mỗi folder tương đương)
        est_total_folders = 9500
        est_total_time_hours = (total_time * est_total_folders) / 3600
        print(f"\n[ESTIMATE] Dự kiến chạy hết 9500 folder mất khoảng: {est_total_time_hours:.2f} giờ")
        print("-" * 40)

if __name__ == "__main__":
    main()
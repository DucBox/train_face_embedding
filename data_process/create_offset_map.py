import boto3
import tarfile
import pandas as pd
import time
import io
import os
import multiprocessing
from multiprocessing import Pool

BUCKET_NAME = "ttnt"
# ROOT_PREFIX = "cv/crawled-datasets/face-google-search/"
ROOT_PREFIX = "cv/projects/face-recognition/webface42m_image_folder"
OUTPUT_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_full_webface42m.parquet"
NUM_WORKERS = 48
SAVE_BATCH_SIZE = 50 

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
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

def list_all_tar_files(bucket, prefix):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    tar_keys = []
    
    print(f"[INFO] Scanning S3 bucket: {bucket} | prefix: {prefix}")
    scan_start = time.time()
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".tar"):
                    tar_keys.append(obj["Key"])
                    
    print(f"[INFO] Found {len(tar_keys)} tar files. Scan time: {time.time() - scan_start:.2f}s")
    return tar_keys

def process_single_tar(args):
    bucket, tar_key = args
    results = []
    
    try:
        s3 = get_s3_client()
        response = s3.get_object(Bucket=bucket, Key=tar_key)
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
        print(f"[ERROR] Failed {tar_key}: {str(e)}")
        return []

    return results

def main():
    start_time = time.time()
    
    tar_keys = list_all_tar_files(BUCKET_NAME, ROOT_PREFIX)
    
    if not tar_keys:
        print("[INFO] No tar files found.")
        return

    total_files = len(tar_keys)
    tasks = [(BUCKET_NAME, key) for key in tar_keys]
    
    all_offsets = []
    processed_count = 0
    batch_counter = 0

    print(f"[INFO] Starting {NUM_WORKERS} workers to extract offsets...")
    
    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_tar, tasks):
            processed_count += 1
            if result:
                all_offsets.extend(result)
            
            if processed_count % 1000 == 0 or processed_count == total_files:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count
                remain_files = total_files - processed_count
                eta = remain_files * avg_time
                
                print(f"[PROGRESS] {processed_count}/{total_files} ({processed_count/total_files*100:.2f}%) "
                      f"| Rows: {len(all_offsets)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            if len(all_offsets) >= 2000000:
                df = pd.DataFrame(all_offsets)
                part_name = f"/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_part_{batch_counter:04d}.parquet"
                df.to_parquet(part_name, index=False)
                print(f"[SAVE] Saved batch {part_name} ({len(df)} rows)")
                
                all_offsets = []
                batch_counter += 1

    if all_offsets:
        df = pd.DataFrame(all_offsets)
        part_name = f"/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_part_{batch_counter:04d}.parquet"
        df.to_parquet(part_name, index=False)
        print(f"[SAVE] Saved final batch {part_name} ({len(df)} rows)")

    print(f"[DONE] Total time: {time.time() - start_time:.2f}s")

    print("[INFO] Merging all parts into single file...")
    try:
        import glob
        files = sorted(glob.glob("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_part_*.parquet"))
        if files:
            dfs = [pd.read_parquet(f) for f in files]
            full_df = pd.concat(dfs, ignore_index=True)
            full_df.to_parquet(OUTPUT_FILE, index=False)
            print(f"[SUCCESS] Final offset table saved: {OUTPUT_FILE} ({len(full_df)} rows)")
            
            for f in files:
                os.remove(f)
    except Exception as e:
        print(f"[ERROR] Merge failed: {e}")

if __name__ == "__main__":
    main()
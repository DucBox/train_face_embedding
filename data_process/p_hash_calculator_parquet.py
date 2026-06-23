import boto3
import time
import io
import tarfile
import imagehash
from PIL import Image
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os
import time
import pandas as pd

BUCKET_NAME = "ttnt"
ROOT_PREFIX = "cv/crawled-datasets/face-google-search/" # Folder cha chứa các folder data_xxx

OUTPUT_DIR = "phash_results_all"
ERROR_DIR = "phash_error_all"
BATCH_SIZE = 10000000

MAX_WORKERS = 64

#None = All
START = None
LIMIT_DATA_FOLDERS = None
LIMIT_TARS_PER_FOLDER = None
FOLDER = None 

# --- S3 CONNECTION ---
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="H?3o0nn4Irej",
    )

def list_subfolders(bucket: str, prefix: str):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    subfolders = []
    
    print(f"[{time.strftime('%H:%M:%S')}] 📂 Scanning subfolders in: {prefix}")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        if "CommonPrefixes" in page:
            for p in page["CommonPrefixes"]:
                subfolders.append(p["Prefix"])
    # print(f"List subfolder data: {subfolders}")         
    return sorted(subfolders)

def list_tar_keys(bucket: str, prefix: str):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    tar_keys = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar"):
                tar_keys.append(obj["Key"])
    
    return sorted(tar_keys)

def save_batch(data, folder, prefix, batch_idx):
    if not data:
        return
    
    cols = ['image_s3_path', 'phash'] if prefix == 'part' else ['image_s3_path', 'error_msg']
    df = pd.DataFrame(data, columns = cols)

    filename = f"{prefix}_{batch_idx:05d}.parquet"
    save_path = os.path.join(folder, filename)

    df.to_parquet(save_path, index = False)
    print(f"Saved {prefix} batch {batch_idx} ({len(df)} rows)")
    
def process_single_tar(args):
    worker_name = mp.current_process().name
    start_time = time.time()
    bucket, tar_key = args
    # print(f"[{time.strftime('%H:%M:%S')}] 🚀 {worker_name} START: {tar_key}")

    results = []
    errors = []

    try:
        s3 = get_s3_client()
        resp = s3.get_object(Bucket=bucket, Key=tar_key)
        file_content = resp['Body'].read()

        with tarfile.open(fileobj=io.BytesIO(file_content), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.lower().endswith(('.jpg', '.jpeg', '.png'))):
                    full_path = f"{tar_key}/{member.name}" 
                    try:
                        f = tar.extractfile(member)
                        if f:
                            img = Image.open(f)
                            h = str(imagehash.phash(img))
                            
                            results.append(f"{full_path},{h}")
                            
                    except (OSError, Image.DecompressionBombError, Exception) as e:
                        errors.append(f"{full_path},{str(e)}")

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]  {worker_name} TAR ERROR {tar_key}: {e}")
        errors.append(f"{tar_key},TAR_FILE_ERROR: {str(e)}")
        return [], errors

    end = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}]  {worker_name} DONE : {tar_key} ({len(results)} imgs in {end:.2f}s)")
    
    return results, errors

# --- MAIN ---
if __name__ == "__main__":
    t_start_scan = time.time()
    all_tar_tasks = []

    # 1. Lấy danh sách các folder data (Dynamic)
    data_folders = list_subfolders(BUCKET_NAME, ROOT_PREFIX)
    print(f"-> Found {len(data_folders)} data folders.")

    # APPLY LIMIT DATA FOLDERS
    if LIMIT_DATA_FOLDERS is not None:
        data_folders = data_folders[START:LIMIT_DATA_FOLDERS]
        print(f" List data folder: {data_folders}")
        print(f" TEST MODE: Limiting to first {LIMIT_DATA_FOLDERS} folders.")

    # 2. Duyệt từng folder để lấy file tar
    for folder_prefix in data_folders:
        tars = list_tar_keys(BUCKET_NAME, folder_prefix)
        original_count = len(tars)
        
        # APPLY LIMIT TARS PER FOLDER
        if LIMIT_TARS_PER_FOLDER is not None:
            tars = tars[:LIMIT_TARS_PER_FOLDER]
        
        # Add vào list tổng
        for key in tars:
            all_tar_tasks.append((BUCKET_NAME, key))
            
        print(f"   - {folder_prefix}: selected {len(tars)}/{original_count} tars")

    print(f"\n[{time.strftime('%H:%M:%S')}] Total tasks prepared: {len(all_tar_tasks)} files. Scan time: {time.time()-t_start_scan:.2f}s")
    
    # 3. Chạy Multiprocessing
    if len(all_tar_tasks) > 0:
        print(f"--- Processing with {MAX_WORKERS} workers ---")
        t_start_process = time.time()

        os.makedirs(OUTPUT_DIR, exist_ok = True)
        os.makedirs(ERROR_DIR, exist_ok = True)

        buffer_ok = []
        buffer_err = []

        batch_count = 0
        total_ok = 0
        total_err = 0

        with Pool(processes=MAX_WORKERS) as pool:
            for res_list, err_list in pool.imap_unordered(process_single_tar, all_tar_tasks):
                if res_list:
                    for item in res_list:
                        path, h = item.split(',')
                        buffer_ok.append((path, h))

                if err_list:
                    for item in err_list:
                        parts = item.split(',')
                        if len(parts) == 2:
                            buffer_err.append((parts[0], parts[1]))
                        else:
                            buffer_err.append((item, 'Unknown'))

                if len(buffer_ok) > BATCH_SIZE:
                    batch_count += 1
                    save_batch(buffer_ok, OUTPUT_DIR, "part", batch_count)
                    total_ok += len(buffer_ok)
                    buffer_ok = []

                if len(buffer_err) > 1:
                    batch_count += 1
                    save_batch(buffer_err, ERROR_DIR, "part", batch_count)
                    total_err += len(buffer_err)
                    buffer_err = []

        if len(buffer_ok) > BATCH_SIZE:
            batch_count += 1
            save_batch(buffer_ok, OUTPUT_DIR, "part", batch_count)
            total_ok += len(buffer_ok)

        if len(buffer_err) > 1:
            batch_count += 1
            save_batch(buffer_err, ERROR_DIR, "part", batch_count)
            total_err += len(buffer_err)
        
        print(f"\n\n DONE! Total images: {total_ok}")
        print(f"⏱️ Total time: {time.time() - t_start_process:.2f}s")
        print(f"   - Valid Images: {total_ok}")
        print(f"   - Bad Images  : {total_err}")
    else:
        print("No tasks found.")
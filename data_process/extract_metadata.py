import boto3
import time
import io
import tarfile
import json
import multiprocessing as mp
from multiprocessing import Pool
import os
import pandas as pd

# --- CONFIGURATION ---
BUCKET_NAME = "ttnt"
ROOT_PREFIX = "cv/crawled-datasets/face-google-search/" 

OUTPUT_DIR = "metadata_results_all"
ERROR_DIR = "metadata_error_all"
BATCH_SIZE = 10000000  # Batch lớn do dữ liệu text nhẹ
MAX_WORKERS = 128

# Config giới hạn để test (để None = chạy hết)
START = None
LIMIT_DATA_FOLDERS = None
LIMIT_TARS_PER_FOLDER = None

# --- S3 CONNECTION ---
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
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
    print(f"Folders: {subfolders}")
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
    if not data: return
    
    if prefix == 'meta':
        # Cấu trúc 4 cột yêu cầu
        cols = ['image_s3_path', 'name', 'width', 'height']
        df = pd.DataFrame(da
        ta, columns=cols)
    else:
        cols = ['file_path', 'error_msg']
        df = pd.DataFrame(data, columns=cols)

    filename = f"{prefix}_{batch_idx:05d}.parquet"
    save_path = os.path.join(folder, filename)
    
    # Lưu parquet
    df.to_parquet(save_path, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Saved batch {batch_idx} ({len(df)} rows)")

# --- WORKER FUNCTION ---
def process_single_tar_metadata(args):
    bucket, tar_key = args
    results = []
    errors = []

    try:
        s3 = get_s3_client()
        resp = s3.get_object(Bucket=bucket, Key=tar_key)
        file_content = resp['Body'].read()
        
        # Đọc tar từ RAM
        with tarfile.open(fileobj=io.BytesIO(file_content), mode="r") as tar:
            for member in tar.getmembers():
                # CHỈ XỬ LÝ FILE JSON
                if member.isfile() and member.name.lower().endswith('.json'):
                    
                    # 1. Thiết lập giá trị mặc định (cho trường hợp lỗi)
                    name_val = "Unknown"
                    width_val = 0
                    height_val = 0
                    
                    # Tái tạo đường dẫn ảnh tương ứng (để làm Key Join)
                    # Giả định: abc.json -> tương ứng với ảnh abc.jpg trong cùng tar
                    base_name = os.path.splitext(member.name)[0]
                    img_filename = f"{base_name}.jpg" 
                    full_path = f"{tar_key}/{img_filename}"
                    
                    try:
                        f = tar.extractfile(member)
                        if f:
                            meta = json.load(f)
                            
                            # 2. Lấy dữ liệu thực tế
                            # Dùng .get() và ép kiểu an toàn
                            raw_name = meta.get("caption")
                            if raw_name:
                                name_val = raw_name
                            
                            # Ép kiểu int cho width/height, nếu lỗi hoặc None thì về 0
                            try:
                                width_val = int(meta.get("width", 0))
                            except:
                                width_val = 0
                                
                            try:
                                height_val = int(meta.get("height", 0))
                            except:
                                height_val = 0
                                
                    except Exception as e:
                        # Nếu file json lỗi, không đọc được -> giữ nguyên giá trị mặc định (Unknown, 0, 0)
                        # Vẫn append vào danh sách kết quả
                        pass

                    # 3. Append kết quả
                    results.append((full_path, name_val, width_val, height_val))

    except Exception as e:
        # Lỗi cấp độ file TAR (không tải được, file hỏng...)
        errors.append((tar_key, str(e)))
        return [], errors
    
    return results, errors

# --- MAIN ---
if __name__ == "__main__":
    t_start = time.time()
    
    # 1. Prepare Tasks
    data_folders = list_subfolders(BUCKET_NAME, ROOT_PREFIX)
    if LIMIT_DATA_FOLDERS: 
        data_folders = data_folders[:LIMIT_DATA_FOLDERS]
    
    all_tasks = []
    for folder in data_folders:
        tars = list_tar_keys(BUCKET_NAME, folder)
        if LIMIT_TARS_PER_FOLDER: 
            tars = tars[:LIMIT_TARS_PER_FOLDER]
        for t in tars: 
            all_tasks.append((BUCKET_NAME, t))
            
    print(f"Total tasks: {len(all_tasks)}")
    
    # 2. Run Multiprocessing
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)
    
    buffer_ok = []
    buffer_err = []
    batch_count = 0
    
    if len(all_tasks) > 0:
        print(f"--- Processing Metadata with {MAX_WORKERS} workers ---")
        with Pool(processes=MAX_WORKERS) as pool:
            for res, err in pool.imap_unordered(process_single_tar_metadata, all_tasks):
                if res: buffer_ok.extend(res)
                if err: buffer_err.extend(err)
                
                # Save Batch OK
                if len(buffer_ok) > BATCH_SIZE:
                    batch_count += 1
                    save_batch(buffer_ok, OUTPUT_DIR, "meta", batch_count)
                    buffer_ok = []
                
                # Save Batch Error (chỉ save nếu lỗi nhiều để tránh rác file)
                if len(buffer_err) > 5000: 
                    save_batch(buffer_err, ERROR_DIR, "error", batch_count)
                    buffer_err = []

        # Save remaining
        if buffer_ok:
            batch_count += 1
            save_batch(buffer_ok, OUTPUT_DIR, "meta", batch_count)
            
        if buffer_err:
            batch_count += 1
            save_batch(buffer_err, ERROR_DIR, "error", batch_count)
            
        print(f"\nDONE. Metadata saved to {OUTPUT_DIR}")
        print(f"⏱️ Total time: {time.time() - t_start:.2f}s")
    else:
        print("No tasks found.")
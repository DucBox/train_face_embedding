import boto3
import tarfile
import io
import time
import pandas as pd
import os
from multiprocessing import Pool, current_process

INPUT_PARQUET_FOLDER = "/workspace/FaceNist/raw_data_processing/phash_results_parquet_all"
BUCKET_NAME = "ttnt"
SOURCE_PREFIX_ROOT = "cv/crawled-datasets/face-google-search/" 
DEST_PREFIX = "cv/unique_dataset_repacked/"

MAX_WORKERS = 8  

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
    )

def worker_repack_tar(args):
    bucket, tar_key, keep_files_set = args
    worker_name = current_process().name
    s3 = get_s3_client()
    
    # 1. Tính toán đường dẫn đích (Mirror Structure)
    # Ví dụ: cv/raw/data_0/abc.tar cv/unique/data_0/abc.tar
    if SOURCE_PREFIX_ROOT in tar_key:
        dest_key = tar_key.replace(SOURCE_PREFIX_ROOT, DEST_PREFIX)
    else:
        # Fallback nếu path không khớp prefix (đề phòng)
        print(f"Source prefix root {SOURCE_PREFIX_ROOT} not in Tar key {tar_key}")
        filename = tar_key.split('/')[-1]
        dest_key = os.path.join(DEST_PREFIX, "unknown_source", filename)

    # 2. Download Full Source TAR (Load vào RAM)
    t0 = time.time()
    resp = s3.get_object(Bucket=bucket, Key=tar_key)
    input_bytes = resp['Body'].read()
    
    # 3. Repack (Xử lý trên RAM)
    output_buffer = io.BytesIO()
    valid_count = 0
    
    # Mở input tar để đọc, output tar để ghi
    with tarfile.open(fileobj=io.BytesIO(input_bytes), mode="r") as in_tar, tarfile.open(fileobj=output_buffer, mode="w") as out_tar:
        
        for member in in_tar:
            # Chỉ xử lý nếu là file và tên file nằm trong danh sách cần giữ
            if member.isfile() and member.name in keep_files_set:
                f = in_tar.extractfile(member)
                if f:
                    # Reset file info để tránh lỗi format khi add sang tar mới
                    member.size = f.getbuffer().nbytes 
                    out_tar.addfile(member, f)
                    valid_count += 1
    
    # 4. Upload New TAR (Nếu có dữ liệu)
    if valid_count > 0:
        output_buffer.seek(0) 
        end = time.time() - t0
        s3.put_object(Bucket=bucket, Key=dest_key, Body=output_buffer)
        print(f"[{time.strftime('%H:%M:%S')}] {worker_name} Saved: {dest_key} ({valid_count} imgs) in {end:.2f}s")
        return valid_count
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {worker_name} Empty result (Skipped): {tar_key} in {end:.2f}s")
        return 0


# --- MAIN PROCESS ---
if __name__ == "__main__":
    t_start = time.time()
    
    print(f"--- PHASE 1: Loading & Grouping Data from {INPUT_PARQUET_FOLDER} ---")
    
    # 1. Đọc toàn bộ Parquet (Pandas tự load cả folder)
    df = pd.read_parquet(INPUT_PARQUET_FOLDER)

    print(f"Loaded raw rows: {len(df)}")
    
    # 2. Lọc unique lần cuối (Đảm bảo chắc chắn 100%)
    df_unique = df.drop_duplicates(subset=['phash'], keep='first')
    print(f"Unique images to keep: {len(df_unique)}")

    # 3. Group by TAR (Tạo Task Map)
    # Input path format: "path/to/folder/file.tar/image_name.jpg"
    tasks_map = {} # Key: tar_path, Value: Set(image_names)
    
    print("Grouping images by TAR file...")
    for full_path in df_unique['image_s3_path']:
        # Tách chuỗi dựa trên ".tar/"
        # Lưu ý: Cần xử lý cẩn thận nếu tên file không chuẩn, nhưng format crawl thường ổn định
        parts = full_path.split('.tar/')
        if len(parts) >= 2:
            tar_path = parts[0] + ".tar"
            img_name = parts[1]
            
            if tar_path not in tasks_map:
                tasks_map[tar_path] = set()
            tasks_map[tar_path].add(img_name)
            
    print(f"Generated {len(tasks_map)} TAR tasks.")
    
    # Chuẩn bị list arguments cho pool
    pool_args = []
    for tar_key, img_set in tasks_map.items():
        pool_args.append((BUCKET_NAME, tar_key, img_set))

    # --- PHASE 2: EXECUTION ---
    print(f"\n--- PHASE 2: Repacking with {MAX_WORKERS} workers ---")
    
    total_saved_images = 0
    processed_tars = 0
    
    with Pool(processes=MAX_WORKERS) as pool:
        # Sử dụng imap_unordered để theo dõi tiến độ
        for count in pool.imap_unordered(worker_repack_tar, pool_args):
            total_saved_images += count
            processed_tars += 1
            
            if processed_tars % 10 == 0:
                print(f"\r[{time.strftime('%H:%M:%S')}] Processed {processed_tars}/{len(tasks_map)} TARs | Saved {total_saved_images} imgs...", end="", flush=True)

    print(f"\n\nMIGRATION COMPLETE!")
    print(f"New Data Location: s3://{BUCKET_NAME}/{DEST_PREFIX}")
    print(f"Total Unique Images Saved: {total_saved_images}")
    print(f"⏱Total Execution Time: {time.time() - t_start:.2f}s")
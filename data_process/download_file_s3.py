import boto3
import tarfile
import io
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# --- CẤU HÌNH ---
BUCKET_NAME = "ttnt"
LOCAL_SAVE_DIR = "s3_samples"
MAX_WORKERS = 16 

# List các file mẫu bạn muốn tải (Format: Key_Tar/Filename_Inside)
TARGET_FILES = [
    "cv/processed-datasets/aligned_face_112_112/person_1349000_1349999/person_1349692.tar/cv_crawled-datasets_face-google-search_data_8000_00597.tar_005976363.jpg_face_3.jpg",
    "cv/processed-datasets/aligned_face_112_112/person_161000_161999/person_0161663.tar/cv_crawled-datasets_face-google-search_data_9500_00042.tar_000428711.jpg_face_0.jpg",
]

# --- KẾT NỐI S3 ---
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
    )

def download_files_from_tar(args):
    """
    Worker xử lý 1 file tar: tìm và trích xuất danh sách các file con (members) được yêu cầu.
    """
    tar_key, target_members = args
    s3 = get_s3_client()
    downloaded_count = 0
    errors = []
    
    try:
        # 1. Tải file TAR về RAM
        # print(f"Downloading TAR: {tar_key}")
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=tar_key)
        file_content = resp['Body'].read()
        
        # 2. Mở file TAR trong RAM
        with tarfile.open(fileobj=io.BytesIO(file_content), mode="r") as tar:
            # Lấy danh sách tên file trong tar để check nhanh
            all_members = tar.getnames()
            
            for target_file in target_members:
                # target_file ví dụ: "000490005.json"
                # Logic check: Đôi khi trong tar có folder con, nên check endswith hoặc exact match
                # Ở đây giả định khớp chính xác tên file
                
                if target_file in all_members:
                    member = tar.getmember(target_file)
                    f = tar.extractfile(member)
                    if f:
                        # 3. Lưu xuống đĩa
                        # Tạo cấu trúc folder local tương ứng để dễ quản lý (tùy chọn)
                        # Ví dụ: downloaded_files/data_0/00049.tar/000490005.json
                        
                        save_path = os.path.join(LOCAL_SAVE_DIR, tar_key, target_file)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        with open(save_path, "wb") as out_f:
                            out_f.write(f.read())
                        
                        downloaded_count += 1
                else:
                    errors.append(f"Not found in tar: {target_file}")

    except Exception as e:
        return tar_key, 0, [f"Error processing tar: {str(e)}"]

    return tar_key, downloaded_count, errors

def main(file_list):
    t_start = time.time()
    
    # 1. Gom nhóm (Grouping) theo file TAR
    # Mục đích: Nếu cần tải 10 file trong cùng 1 tar, chỉ tải tar đó 1 lần.
    tasks_map = defaultdict(list)
    
    print("--- 1. Parsing & Grouping paths ---")
    for full_path in file_list:
        # Logic tách chuỗi: Tìm ".tar/"
        if ".tar/" in full_path:
            parts = full_path.split(".tar/")
            tar_key = parts[0] + ".tar"
            inner_file = parts[1]
            
            tasks_map[tar_key].append(inner_file)
        else:
            print(f"⚠️ Skip invalid path format: {full_path}")
            
    print(f"-> Found {len(tasks_map)} unique TAR files to process.")

    # 2. Chuyển thành list task cho Multiprocessing
    worker_tasks = []
    for tar_key, files_inside in tasks_map.items():
        worker_tasks.append((tar_key, files_inside))

    # 3. Chạy Multiprocessing
    print(f"--- 2. Downloading with {MAX_WORKERS} workers ---")
    total_downloaded = 0
    
    with Pool(processes=MAX_WORKERS) as pool:
        for tar_key, count, errs in pool.imap_unordered(download_files_from_tar, worker_tasks):
            total_downloaded += count
            if count > 0:
                print(f"✅ {tar_key}: Extracted {count} files.")
            
            if errs:
                for e in errs:
                    print(f"❌ {tar_key}: {e}")

    print(f"\nDONE! Downloaded {total_downloaded} files to '{LOCAL_SAVE_DIR}'")
    print(f"Time: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    main(TARGET_FILES)
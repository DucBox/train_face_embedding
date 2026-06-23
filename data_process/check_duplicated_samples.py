# import pandas as pd

# # 1. Config để in full nội dung cột, không bị '...' nếu in dạng DataFrame
# pd.set_option('display.max_colwidth', None)

# # Đọc dữ liệu (giữ nguyên path của bạn)
# df = pd.read_parquet("/workspace/FaceNist/raw_data_processing/phash_results_all")

# # 2. Tìm các phash bị trùng (xuất hiện ít nhất 2 lần)
# # keep=False để lấy tất cả các bản ghi trùng, không giữ lại cái nào là unique
# duplicate_phashes = df[df.duplicated(subset=['phash'], keep=False)]

# # Lấy danh sách 5 mã phash trùng nhau đầu tiên (unique để lấy danh sách mã)
# target_phashes = duplicate_phashes['phash'].unique()[:5]

# print(f"--- Tìm thấy {len(target_phashes)} nhóm phash trùng nhau (hiển thị 5 nhóm đầu) ---\n")

# # 3. Lặp và in ra cặp ảnh cho mỗi phash
# for i, phash in enumerate(target_phashes, 1):
#     # Lấy 2 dòng đầu tiên có phash này
#     pair_paths = df[df['phash'] == phash]['image_s3_path'].head(2).values
    
#     print(f"Cặp {i} (Phash: {phash}):")
#     # Print từng path (cách này chắc chắn không bị cắt)
#     for path in pair_paths:
#         print(f" - {path}")
#     print("-" * 50)

import boto3
import tarfile
import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- CẤU HÌNH TỪ USER ---
BUCKET_NAME = "ttnt"
OUTPUT_DIR = "./collision_check_results"  # Nơi lưu ảnh tải về

# Dữ liệu đầu vào (Copy từ yêu cầu của bạn)
# Cấu trúc: (Phash, [List các đường dẫn])
DATA_PAIRS = [
    ("f97aa48c38666cd4", [
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490005.jpg",
        "cv/crawled-datasets/face-google-search/data_0/00094.tar/000944691.jpg"
    ]),
    ("8a5aa17a43337277", [
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490001.jpg",
        "cv/crawled-datasets/face-google-search/data_0/00083.tar/000836532.jpg"
    ]),
    ("cc1e3cd30bba1939", [
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490012.jpg",
        "cv/crawled-datasets/face-google-search/data_0/00061.tar/000619942.jpg"
    ]),
    ("e4b45ee7865b8183", [
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490006.jpg",
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490287.jpg"
    ]),
    ("85c61a696f326a3e", [
        "cv/crawled-datasets/face-google-search/data_0/00049.tar/000490029.jpg",
        "cv/crawled-datasets/face-google-search/data_0/00092.tar/000927918.jpg"
    ]),
]

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="H?3o0nn4Irej",
    )

def parse_s3_path(full_path):
    """
    Tách đường dẫn user cung cấp thành S3 Key của file TAR và tên file ảnh bên trong.
    Input: cv/.../00049.tar/000490005.jpg
    Output: (cv/.../00049.tar, 000490005.jpg)
    """
    parts = full_path.split(".tar/")
    if len(parts) != 2:
        return None, None
    tar_key = parts[0] + ".tar"
    member_name = parts[1]
    return tar_key, member_name

def process_tar_group(tar_key, extraction_tasks):
    """
    Xử lý 1 file TAR: Tải về (stream) -> Tìm các file ảnh cần thiết -> Lưu ra đĩa
    extraction_tasks: Dict { member_name: [list_local_save_paths] }
    (Một ảnh có thể thuộc về nhiều cặp hash khác nhau nên cần list save paths)
    """
    s3 = get_s3_client()
    print(f"Processing TAR: {tar_key} (Looking for {len(extraction_tasks)} images)...")
    
    try:
        # Stream file TAR từ S3 (không tải toàn bộ xuống đĩa để tiết kiệm IO)
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=tar_key)
        stream = io.BytesIO(obj['Body'].read())
        
        with tarfile.open(fileobj=stream, mode='r') as tar:
            for member in tar:
                # Nếu file trong tar có tên nằm trong danh sách cần lấy
                if member.name in extraction_tasks:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read()
                        # Lưu file vào tất cả các đích cần thiết (các folder Hash khác nhau)
                        for save_path in extraction_tasks[member.name]:
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            with open(save_path, "wb") as out:
                                out.write(content)
                            print(f"  -> Saved: {save_path}")
    except Exception as e:
        print(f"ERROR processing {tar_key}: {e}")

def main():
    # 1. Tổ chức lại dữ liệu: Group by TAR Key
    # Cấu trúc: tasks[tar_key][member_name] = [local_path_1, local_path_2...]
    tasks = defaultdict(lambda: defaultdict(list))
    
    for idx, (phash, paths) in enumerate(DATA_PAIRS):
        pair_name = f"Pair_{idx+1}_{phash}"
        
        for path in paths:
            tar_key, member_name = parse_s3_path(path)
            if tar_key:
                # Tạo đường dẫn lưu file local: ./output/Pair_1_hash/tên_ảnh.jpg
                save_path = os.path.join(OUTPUT_DIR, pair_name, member_name)
                tasks[tar_key][member_name].append(save_path)

    print(f"Optimization: Found {len(tasks)} unique TAR files to download for {len(DATA_PAIRS)} pairs.")

    # 2. Thực thi đa luồng (Multi-thread) vì chủ yếu là IO bound (Download S3)
    # Mỗi luồng xử lý 1 file TAR trọn vẹn
    with ThreadPoolExecutor(max_workers=5) as executor:
        for tar_key, extraction_map in tasks.items():
            executor.submit(process_tar_group, tar_key, extraction_map)

if __name__ == "__main__":
    main()
import mxnet as mx
import boto3
import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import threading
import os

# --- CẤU HÌNH ---
REC_PATH = '/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_reindex/train.rec'
IDX_PATH = '/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_reindex/train.idx'
# S3 Config
S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "<S3_SECRET_KEY>"
BUCKET_NAME = 'ttnt'
ROOT_PREFIX = 'cv/projects/face-recognition/public5m_reindex_image_folder'

# Tuning Performance
MAX_WORKERS = 32      # Số luồng upload S3
QUEUE_SIZE_LIMIT = 200 # Giới hạn queue để tránh tràn RAM

CHECKPOINT_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_reindex/checkpoints_save_s3.txt"

# Init S3 Client (Global)
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)
# Lock để các thread ghi file txt ko bị đè nhau
file_lock = threading.Lock()

def get_shard_name(person_id):
    shard_size = 10000
    start = (person_id // shard_size) * shard_size
    end = start + shard_size - 1
    return f"person_{start}_{end}"

def upload_worker(person_id, image_list):
    """Upload xong mới ghi checkpoint"""
    try:
        # 1. Nén Tar
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for idx, img_bytes in enumerate(image_list):
                info = tarfile.TarInfo(name=f"{idx}.jpg")
                info.size = len(img_bytes)
                tar.addfile(tarinfo=info, fileobj=io.BytesIO(img_bytes))
        
        tar_buffer.seek(0)
        
        # 2. Upload S3
        shard_folder = get_shard_name(person_id)
        s3_key = f"{ROOT_PREFIX}/{shard_folder}/person_{person_id:07d}.tar"
        
        s3_client.upload_fileobj(tar_buffer, BUCKET_NAME, s3_key)
        tar_buffer.close()

        # 3. GHI CHECKPOINT (Quan trọng)
        # Chỉ khi upload ko lỗi mới chạy xuống đây
        with file_lock:
            with open(CHECKPOINT_FILE, 'a') as f:
                f.write(f"{person_id}\n")
                f.flush() # Đẩy buffer xuống OS
                os.fsync(f.fileno()) # Đẩy OS xuống ổ cứng vật lý ngay lập tức
        
        return True
    except Exception as e:
        print(f"FAILED Person {person_id}: {e}")
        return False

def main():
    # --- LOAD CHECKPOINT ---
    completed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading checkpoint from {CHECKPOINT_FILE}...")
        with open(CHECKPOINT_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    completed_ids.add(int(line.strip()))
        print(f"Resuming... Found {len(completed_ids)} completed persons.")

    print("Init RecordIO...")
    imgrec = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, 'r')
    
    # Safe Header Check
    s = imgrec.read_idx(0)
    if s is not None:
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            print("Header > 0")
            idx_list = range(1, int(header.label[0]))
        else:
            print("Header < 0")
            idx_list = imgrec.keys
    else:
        idx_list = imgrec.keys

    print("Start ETL Pipeline...")
    
    current_person_id = None
    person_images = []
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = []
    
    pbar = tqdm(total=len(idx_list), unit="img")

    for i in idx_list:
        s = imgrec.read_idx(i)
        if s is None: continue

        header, img_bytes = mx.recordio.unpack(s)
        
        # Parse Label
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)
        
        # --- RESUME LOGIC (Skip ID đã làm) ---
        if label in completed_ids:
            pbar.update(1)
            continue
        
        # Logic gom nhóm
        if label != current_person_id:
            # Submit người cũ
            if current_person_id is not None:
                while len(futures) > QUEUE_SIZE_LIMIT:
                    futures = [f for f in futures if not f.done()]
                    if len(futures) > QUEUE_SIZE_LIMIT:
                        time.sleep(0.05)
                
                # Copy list và submit
                futures.append(executor.submit(upload_worker, current_person_id, list(person_images)))
            
            # Reset
            current_person_id = label
            person_images = []
            
        person_images.append(img_bytes)
        pbar.update(1)

    # Submit người cuối cùng
    if current_person_id is not None and person_images:
        executor.submit(upload_worker, current_person_id, person_images)

    print("Finalizing uploads...")
    executor.shutdown(wait=True)
    print("Done.")

if __name__ == "__main__":
    main()
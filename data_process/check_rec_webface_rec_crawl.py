import os
import glob
import cv2
import mxnet as mx
import numpy as np

# --- CONFIG ---
WEBFACE_PREFIX = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M/train"  # Prefix của file .rec/.idx WebFace gốc
CRAWL_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01/crawl_v1_shards" # Folder chứa shards Crawl
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01/verify_merged_data"
MERGE_THRESHOLD = 3000000
N_PERSONS = 100     # Số lượng ID cần check
MAX_IMGS_PER_SRC = 100 # Số ảnh tối đa mỗi nguồn để đỡ nặng
def extract_images(rec_path, target_ids, source_tag):
    # Check file exists
    if not os.path.exists(rec_path[:-3] + 'idx'):
        print(f"[WARN] IDX not found for {rec_path}")
        return

    record = mx.recordio.MXIndexedRecordIO(rec_path[:-3] + 'idx', rec_path, 'r')
    
    for idx in record.keys:
        header, img_bytes = mx.recordio.unpack(record.read_idx(idx))
        
        # --- FIX START: Xử lý trường hợp label là array ---
        label = header.label
        if isinstance(label, (np.ndarray, list, tuple)):
            label = label[0] # Lấy phần tử đầu tiên (ID)
        pid = int(label)
        # --- FIX END ---
        
        if pid not in target_ids: continue
        
        try:
            save_dir = os.path.join(OUTPUT_DIR, f"person_{pid}")
            os.makedirs(save_dir, exist_ok=True)
            
            curr_count = len(glob.glob(os.path.join(save_dir, f"{source_tag}_*.jpg")))
            if curr_count >= MAX_IMGS_PER_SRC: continue

            img = mx.image.imdecode(img_bytes).asnumpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            fname = f"{source_tag}_{idx}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), img)
        except: pass

def main():
    crawl_shards = sorted(glob.glob(os.path.join(CRAWL_DIR, "*.rec")))
    target_ids = set()
    
    print("Scanning Crawl data to find merged IDs...")
    for p in crawl_shards:
        if len(target_ids) >= N_PERSONS: break
        
        record = mx.recordio.MXIndexedRecordIO(p[:-3] + 'idx', p, 'r')
        for idx in record.keys:
            header, _ = mx.recordio.unpack(record.read_idx(idx))
            
            # Cần fix cả chỗ này cho đồng bộ
            label = header.label
            if isinstance(label, (np.ndarray, list, tuple)):
                label = label[0]
            pid = int(label)
            
            if pid < MERGE_THRESHOLD:
                target_ids.add(pid)
                if len(target_ids) >= N_PERSONS: break

    print(f"Selected {len(target_ids)} IDs: {list(target_ids)}")

    print("Extracting from Crawl shards...")
    for p in crawl_shards:
        extract_images(p, target_ids, "crawl")

    print("Extracting from WebFace...")
    extract_images(WEBFACE_PREFIX + ".rec", target_ids, "webface")

    print(f"[DONE] Check folder {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
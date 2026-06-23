import os
import glob
import cv2
import mxnet as mx
from tqdm import tqdm

INPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01/crawl_v1_shards"
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01/verify_shards_vis"
THRESHOLD = 3000000
TARGET_IDS = 20
MAX_IMGS = 100

def main():
    rec_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.rec")))
    print(f"Found {len(rec_files)} shards.")

    # Dictionary: {person_id: count_saved}
    collected_webface = {}
    collected_crawl = {}

    for rec_path in rec_files:
        # Check stop condition
        done_wf = len(collected_webface) == TARGET_IDS and all(c >= MAX_IMGS for c in collected_webface.values())
        done_cr = len(collected_crawl) == TARGET_IDS and all(c >= MAX_IMGS for c in collected_crawl.values())
        if done_wf and done_cr: break

        print(f"Processing {os.path.basename(rec_path)}...")
        record = mx.recordio.MXIndexedRecordIO(rec_path[:-3] + 'idx', rec_path, 'r')
        
        for idx in record.keys:
            header, img_bytes = mx.recordio.unpack(record.read_idx(idx))
            pid = int(header.label)
            
            is_webface = pid < THRESHOLD
            tracker = collected_webface if is_webface else collected_crawl
            
            if pid not in tracker:
                if len(tracker) >= TARGET_IDS: continue
                tracker[pid] = 0
            
            if tracker[pid] >= MAX_IMGS: continue

            try:
                img = mx.image.imdecode(img_bytes).asnumpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                cat = "webface" if is_webface else "crawl"
                save_dir = os.path.join(OUTPUT_DIR, f"{cat}_{pid}")
                os.makedirs(save_dir, exist_ok=True)
                
                cv2.imwrite(os.path.join(save_dir, f"img_{tracker[pid]}.jpg"), img)
                tracker[pid] += 1
            except:
                pass

    print(f"[DONE] Saved samples to {OUTPUT_DIR}")
    print(f"Webface IDs: {list(collected_webface.keys())}")
    print(f"Crawl IDs: {list(collected_crawl.keys())}")

if __name__ == "__main__":
    main()
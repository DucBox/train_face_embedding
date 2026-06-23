import mxnet as mx
import numpy as np
from tqdm import tqdm
import numbers

def analyze_insightface_file(rec_path):
    idx_path = rec_path.replace(".rec", ".idx")
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    
    # 1. Đọc Header tại Index 0
    s0 = imgrec.read_idx(0)
    header0, _ = mx.recordio.unpack(s0)
    
    print("-" * 30)
    print(f"FILE: {rec_path}")
    print("-" * 30)
    
    if header0.flag > 0:
        claimed_images = int(header0.label[0])
        claimed_ids = int(header0.label[1])
        print(f"[Header Info] Total Images: {claimed_images:,}")
        print(f"[Header Info] Total IDs: {claimed_ids:,}")
        # Dải index dữ liệu thật
        indices = range(1, claimed_images)
    else:
        print("[Warning] File does not have a standard InsightFace Header (Flag=0)")
        indices = list(imgrec.keys)

    # 2. Quét thực tế để lấy danh sách ID
    actual_ids = set()
    total_samples = 0
    
    print(f"Scanning {len(indices):,} samples to verify IDs...")
    for i in tqdm(indices):
        try:
            s = imgrec.read_idx(i)
            if s is None: continue
            header, _ = mx.recordio.unpack(s)
            
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            
            actual_ids.add(int(label))
            total_samples += 1
        except Exception as e:
            print(f"Error at index {i}: {e}")
            break

    # 3. Thống kê kết quả
    if not actual_ids:
        print("No labels found!")
        return

    min_id = min(actual_ids)
    max_id = max(actual_ids)
    unique_count = len(actual_ids)
    gap_count = (max_id - min_id + 1) - unique_count

    print("-" * 30)
    print(f"ACTUAL STATS:")
    print(f"Actual Images Scanned: {total_samples:,}")
    print(f"Actual Unique IDs   : {unique_count:,}")
    print(f"Actual ID Range     : {min_id} -> {max_id}")
    print(f"ID Gaps (Holes)     : {gap_count:,} {'(Clean!)' if gap_count == 0 else '(Fragmented!)'}")
    
    if gap_count > 0:
        print(f"Conclusion: Max ID ({max_id}) is much larger than unique count ({unique_count}).")
        print("You MUST re-index to save VRAM.")
    else:
        print("Conclusion: IDs are continuous. Simple offset is enough if starting from 0.")
    print("-" * 30)

if __name__ == "__main__":
    # Đường dẫn tới file public của bạn
    analyze_insightface_file("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M/train.rec")
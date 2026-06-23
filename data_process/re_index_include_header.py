import os
import mxnet as mx
import numpy as np
from tqdm import tqdm

def reindex_public_cleaned(input_rec, output_folder, base_max_id):
    input_idx = input_rec.replace(".rec", ".idx")
    filename = os.path.basename(input_rec)
    
    os.makedirs(output_folder, exist_ok=True)
    output_rec = os.path.join(output_folder, filename)
    output_idx = os.path.join(output_folder, filename.replace(".rec", ".idx"))

    source_record = mx.recordio.MXIndexedRecordIO(input_idx, input_rec, 'r')
    target_record = mx.recordio.MXIndexedRecordIO(output_idx, output_rec, 'w')

    # Thực tế từ scan của bạn
    ACTUAL_NUM_IMAGES = 4674365
    ACTUAL_NUM_IDS = 558208

    # 1. Ghi Header0 mới (Sửa lại số ID từ 5.2M về 558K)
    # Flag > 0 để InsightFace vẫn nhận ra cấu trúc chuẩn
    new_header0 = mx.recordio.IRHeader(1, [ACTUAL_NUM_IMAGES, ACTUAL_NUM_IDS], 0, 0)
    target_record.write_idx(0, mx.recordio.pack(new_header0, b''))

    print(f"Re-indexing {ACTUAL_NUM_IMAGES:,} images...")
    
    # 2. Dịch chuyển ID (Vì ID gốc từ 0-558207 nên chỉ cần + OFFSET)
    # Offset = base_max_id + 1
    offset = base_max_id + 1
    
    for i in tqdm(range(1, ACTUAL_NUM_IMAGES)):
        s = source_record.read_idx(i)
        header, img_bytes = mx.recordio.unpack(s)
        
        # Lấy label cũ và cộng offset
        old_label = header.label
        if not isinstance(old_label, (int, float, np.number)):
            old_label = old_label[0]
            
        new_label = int(old_label) + offset
        
        new_header = mx.recordio.IRHeader(header.flag, new_label, header.id, header.id2)
        target_record.write_idx(i, mx.recordio.pack(new_header, img_bytes))

    source_record.close()
    target_record.close()
    
    print("-" * 30)
    print(f"DONE!")
    print(f"Base Max ID      : {base_max_id}")
    print(f"Added IDs        : {ACTUAL_NUM_IDS}")
    print(f"Final Max ID     : {base_max_id + ACTUAL_NUM_IDS}")
    print(f"Config num_classes: {base_max_id + ACTUAL_NUM_IDS + 1}")
    print("-" * 30)

if __name__ == "__main__":
    BASE_MAX_ID = 2058464
    INPUT_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m/train.rec"
    OUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_reindex"
    
    reindex_public_cleaned(INPUT_FILE, OUT_DIR, BASE_MAX_ID)
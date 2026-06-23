import os
import cv2
import numpy as np
import mxnet as mx

# Cấu hình
folder = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/re_index_v1_shards"

rec_pattern = "train_{}.rec"
idx_pattern = "train_{}.idx"
output_root = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/visualization/person_grid_images"
N = 1000  # Chỉ lấy N person_id đầu tiên

if not os.path.exists(output_root):
    os.makedirs(output_root)

grid_h, grid_w = 20, 20
tile_h, tile_w = 112, 112

def process_part(part_idx, person_counter):
    rec_path = os.path.join(folder, rec_pattern.format(part_idx))
    idx_path = os.path.join(folder, idx_pattern.format(part_idx))
    
    if not os.path.exists(rec_path):
        return person_counter

    print(f"Processing {rec_pattern.format(part_idx)} ...")
    
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    header0, _ = mx.recordio.unpack_img(imgrec.read_idx(0))
    
    if header0.flag > 0:
        print(f"  Header flag > 0: range(1, {int(header0.label[0])})")
        imgidx = np.arange(1, int(header0.label[0]))
    else:
        print(f"  Header flag <= 0: read all keys")
        imgidx = np.array(list(imgrec.keys))
    
    # Chỉ lấy N person_id đầu tiên
    person_dict = {}
    person_counter[0] += 1  # Đánh số part
    
    for i, idx in enumerate(imgidx):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(imgidx)} images")
        
        s = imgrec.read_idx(idx)
        if s is None:
            continue
            
        header, img_encoded = mx.recordio.unpack_img(s)
        img = mx.image.imdecode(img_encoded).asnumpy()
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        pid = int(header.id)
        
        # Giới hạn N person_id
        if len(person_dict) >= N:
            break
            
        if pid not in person_dict:
            person_dict[pid] = []
        person_dict[pid].append(img)
    
    imgrec.close()
    
    # Tạo grid
    for pid, images in person_dict.items():
        pid_dir = os.path.join(output_root, str(pid))
        if not os.path.exists(pid_dir):
            os.makedirs(pid_dir)
        
        num_images = len(images)
        print(f"  Person {pid}: {num_images} images")
        
        canvas = 255 * np.ones((tile_h * grid_h, tile_w * grid_w, 3), dtype=np.uint8)
        
        for i, img in enumerate(images):
            if img.shape[:2] != (tile_h, tile_w):
                img = cv2.resize(img, (tile_w, tile_h))
            
            y = (i // grid_w) * tile_h
            x = (i % grid_w) * tile_w
            canvas[y:y+tile_h, x:x+tile_w] = img
        
        out_path = os.path.join(pid_dir, f"{pid}.jpg")
        cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    return person_counter

# Chạy với giới hạn N
person_counter = [0]
part_idx = 1
total_persons = 0

while total_persons < N:
    total_persons = len(os.listdir(output_root)) if os.path.exists(output_root) else 0
    if total_persons >= N:
        break
    
    person_counter = process_part(part_idx, person_counter)
    part_idx += 1
    rec_path = os.path.join(folder, rec_pattern.format(part_idx))
    if not os.path.exists(rec_path):
        break

print(f"Done! Created {min(N, len(os.listdir(output_root)))} person grids.")

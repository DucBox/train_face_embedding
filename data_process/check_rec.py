import os
import mxnet as mx
import numpy as np

folder = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M"
rec_path = os.path.join(folder, "train.rec")
idx_path = os.path.join(folder, "train.idx")

print("=== DEBUG train_1.rec ===")
imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

# 1. Check header 0
print("\n1. Đọc header 0:")
s0 = imgrec.read_idx(0)
header0, img0 = mx.recordio.unpack_img(s0)
print(f"   header0.flag: {header0.flag}")
print(f"   header0.id: {header0.id}")
print(f"   header0.label: {header0.label}")
print(f"   img0 shape: {img0.shape if img0 is not None else 'None'}")
print(f"   img0 size: {img0.size if img0 is not None else 0}")

# 2. Vì flag <= 0, lấy keys
print("\n2. Lấy keys:")
keys = list(imgrec.keys)
print(f"   Total keys: {len(keys)}")
print(f"   First 5 keys: {keys[:5]}")
print(f"   Last 5 keys: {keys[-5:]}")

# 3. Debug record đầu tiên gây lỗi (idx=1)
print("\n3. Debug record đầu tiên (idx=1):")
for test_idx in [1, 2, 3, 10, 100]:
    print(f"\n   --- Record {test_idx} ---")
    s = imgrec.read_idx(test_idx)
    if s is None:
        print("      s is None")
        continue
    
    try:
        header, img_encoded = mx.recordio.unpack_img(s)
        print(f"      header.id: {header.id}")
        print(f"      header.flag: {header.flag}")
        print(f"      header.label: {header.label}")
        print(f"      img_encoded len: {len(img_encoded) if img_encoded is not None else 0}")
        
        if img_encoded is not None and len(img_encoded) > 0:
            img = mx.image.imdecode(img_encoded)
            if img is not None:
                img_np = img.asnumpy()
                print(f"      img shape: {img_np.shape}")
                print(f"      img dtype: {img_np.dtype}")
            else:
                print("      imdecode trả về None")
        else:
            print("      img_encoded rỗng")
            
    except Exception as e:
        print(f"      ERROR: {str(e)}")

# 4. Check format img_encoded (JPEG?)
print("\n4. Check JPEG magic bytes:")
jpeg_magic = b'\xff\xd8\xff'
for test_idx in [1, 2, 3]:
    s = imgrec.read_idx(test_idx)
    header, img_encoded = mx.recordio.unpack_img(s)
    if img_encoded and len(img_encoded) > 0:
        is_jpeg = img_encoded[:3] == jpeg_magic
        print("   idx " + str(test_idx) + ": JPEG? " + str(is_jpeg))
        print("   idx " + str(test_idx) + ": first 10 bytes: " + str(list(img_encoded[:10])))

imgrec.close()

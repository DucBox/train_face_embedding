import os
import argparse
import boto3
import pandas as pd
import numpy as np
import io
import tarfile
import ast
import cv2
import shutil
import random
from PIL import Image
import multiprocessing
from functools import partial

# --- CONFIG ---
S3_BUCKET = "ttnt"
RAW_PREFIX = "cv/crawled-datasets/face-google-search/"
OUTPUT_DEBUG_DIR = "./debug_output_samples"
NUM_SAMPLES = 1000  # Số lượng ảnh muốn test

# --- HELPER FUNCTIONS ---
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
    )

def decode_image_pil(bytes_data):
    try:
        return Image.open(io.BytesIO(bytes_data)).convert('RGB')
    except:
        return None

def crop_face_custom(pil_img, bbox):
    """
    LOGIC QUAN TRỌNG: Phải khớp 100% với code Inference.
    Logic: Mở rộng 50% mỗi bên.
    bbox: [x1, y1, x2, y2]
    """
    width, height = pil_img.size
    tlx, tly, brx, bry = bbox[0], bbox[1], bbox[2], bbox[3]
    w = int(brx - tlx)
    h = int(bry - tly)
    
    # Logic mở rộng
    x1 = int(tlx - w/2)
    y1 = int(tly - h/2)
    x2 = int(brx + w/2)
    y2 = int(bry + h/2)
    
    # Boundary check
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(width, x2); y2 = min(height, y2)
    
    return pil_img.crop((x1, y1, x2, y2))

def parse_s3_info(s3_path):
    parts = s3_path.split('/')
    try:
        if "face-google-search" in parts:
            idx = parts.index("face-google-search") + 1
            return parts[idx], parts[idx+1]
        for p in parts:
            if p.startswith("data_"):
                idx = parts.index(p)
                return p, parts[idx+1]
        return None, None
    except:
        return None, None

def sanitize_filename(s3_path):
    """Biến s3 path thành tên file an toàn để lưu trên disk"""
    # Ví dụ: cv/../data_0/abc.tar/img.jpg -> data_0_abc_tar_img.jpg
    name = s3_path.replace("cv/crawled-datasets/face-google-search/", "")
    return name.replace("/", "_")

# --- WORKER FUNCTION ---
def process_batch_debug(df_chunk, output_dir):
    s3 = get_s3_client()
    
    # Group by TAR để tối ưu download
    grouped = df_chunk.groupby(['data_folder', 'tar_file'])
    
    processed_count = 0
    
    for (data_folder, tar_filename), group in grouped:
        s3_key = f"{RAW_PREFIX}{data_folder}/{tar_filename}"
        
        try:
            # 1. Download TAR
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            tar_bytes = io.BytesIO(obj['Body'].read())
            
            # Map filename -> row
            file_map = {row['s3_path'].split('/')[-1]: row for _, row in group.iterrows()}
            
            with tarfile.open(fileobj=tar_bytes, mode='r') as tar:
                for member in tar:
                    if member.name in file_map:
                        row = file_map[member.name]
                        
                        # 2. Decode
                        f = tar.extractfile(member)
                        if not f: continue
                        pil_img = decode_image_pil(f.read())
                        if pil_img is None: continue
                        
                        try:
                            # Parse BBox
                            bbox = ast.literal_eval(row['bboxs'])[0]
                            
                            # Tên file cơ sở
                            base_name = sanitize_filename(row['s3_path'])
                            
                            # --- A. LƯU ẢNH CROP (INPUT MODEL) ---
                            # Dùng đúng logic custom crop
                            cropped_pil = crop_face_custom(pil_img, bbox)
                            crop_path = os.path.join(output_dir, f"{base_name}_CROP.jpg")
                            cropped_pil.save(crop_path)
                            
                            # --- B. LƯU ẢNH GỐC + VẼ BBOX (VISUAL CHECK) ---
                            # Convert PIL -> OpenCV (RGB -> BGR) để vẽ
                            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            
                            # Vẽ bbox gốc (màu xanh lá)
                            tlx, tly, brx, bry = map(int, bbox[:4])
                            cv2.rectangle(img_cv, (tlx, tly), (brx, bry), (0, 255, 0), 2)
                            
                            # Vẽ bbox mở rộng (màu xanh dương - optional để check logic crop)
                            w = brx - tlx; h = bry - tly
                            ex_x1 = int(tlx - w/2); ex_y1 = int(tly - h/2)
                            ex_x2 = int(brx + w/2); ex_y2 = int(bry + h/2)
                            # cv2.rectangle(img_cv, (ex_x1, ex_y1), (ex_x2, ex_y2), (255, 0, 0), 2)
                            
                            draw_path = os.path.join(output_dir, f"{base_name}_VISUAL.jpg")
                            cv2.imwrite(draw_path, img_cv)
                            
                            # --- C. LƯU ẢNH GỐC RAW (Optional) ---
                            raw_path = os.path.join(output_dir, f"{base_name}_RAW.jpg")
                            pil_img.save(raw_path)
                            
                            processed_count += 1
                            
                        except Exception as e:
                            print(f"[ERR] Processing img {member.name}: {e}")
                            continue
                            
        except Exception as e:
            print(f"[ERR] Download tar {tar_filename}: {e}")
            continue
            
    return processed_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet_path', type=str, required=True, help="Path file parquet input")
    parser.add_argument('--data_ids', type=int, nargs='+', default=None, help="Filter data folder (optional)")
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help="Số lượng ảnh mẫu")
    args = parser.parse_args()

    # 1. Setup Folder
    if os.path.exists(OUTPUT_DEBUG_DIR):
        print(f"[INFO] Removing old debug folder: {OUTPUT_DEBUG_DIR}")
        shutil.rmtree(OUTPUT_DEBUG_DIR)
    os.makedirs(OUTPUT_DEBUG_DIR)
    print(f"[INFO] Saving debug images to: {OUTPUT_DEBUG_DIR}")

    # 2. Load & Sample Data
    print(f"[INFO] Loading Parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path, columns=['s3_path', 'bboxs'])
    
    # Parse helper columns
    df['data_folder'] = df['s3_path'].apply(lambda x: parse_s3_info(x)[0])
    df['tar_file'] = df['s3_path'].apply(lambda x: parse_s3_info(x)[1])
    
    # Filter theo data_id nếu có
    if args.data_ids:
        target_folders = {f"data_{i}" for i in args.data_ids}
        df = df[df['data_folder'].isin(target_folders)]
    
    # Random Sample
    total_rows = len(df)
    sample_size = min(args.num_samples, total_rows)
    print(f"[INFO] Sampling {sample_size}/{total_rows} rows...")
    
    if total_rows > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()

    # 3. Multiprocessing Debug
    num_workers = 128
    chunks = np.array_split(df_sample, num_workers)
    
    print(f"[RUN] Starting {num_workers} workers to download & visualize...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Dùng partial để truyền output_dir cố định
        func = partial(process_batch_debug, output_dir=OUTPUT_DEBUG_DIR)
        results = pool.map(func, chunks)
        
    total_processed = sum(results)
    print(f"\n[DONE] Successfully saved {total_processed} samples to {OUTPUT_DEBUG_DIR}")
    print("Format tên file:")
    print("  1. *_VISUAL.jpg: Ảnh gốc vẽ bbox xanh lá.")
    print("  2. *_CROP.jpg:   Ảnh input model (đã crop custom).")

if __name__ == "__main__":
    main()
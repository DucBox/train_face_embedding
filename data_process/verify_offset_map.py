import boto3
import pandas as pd
import io
from PIL import Image
import os

INPUT_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_full_face_align.parquet"
BUCKET_NAME = "ttnt"
VERIFY_DIR = "verify_images"

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="<S3_SECRET_KEY>",
    )

def download_range(s3, bucket, key, start, length):
    # Range Header: bytes=START-END (Inclusive)
    end = start + length - 1
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp['Body'].read()

def main():
    if not os.path.exists(INPUT_PARQUET):
        print("Chưa có file parquet kết quả.")
        return
        
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows.")
    
    # Lấy 5 mẫu ngẫu nhiên
    samples = df.sample(3)
    s3 = get_s3_client()
      
    os.makedirs(VERIFY_DIR, exist_ok=True)
    
    print(f"\n[VERIFY] Downloading 5 random samples using Offset...")
    
    for idx, row in samples.iterrows():
        tar_path = row['tar_path']
        img_name = row['member_name']
        start = row['start_byte']
        length = row['length']
        
        print(f" -> Check: {img_name} in {tar_path} | Offset: {start} | Len: {length}")
        
        try:
            # RANGE REQUEST - Logic cốt lõi
            img_bytes = download_range(s3, BUCKET_NAME, tar_path, start, length)
            
            # Check size
            if len(img_bytes) != length:
                print(f"   [FAIL] Size mismatch. Got {len(img_bytes)}, expected {length}")
                continue
                
            # Try Decode Image
            img = Image.open(io.BytesIO(img_bytes))
            save_path = os.path.join(VERIFY_DIR, f"check_{idx}_{img_name}")
            img.save(save_path)
            print(f"   [OK] Saved to {save_path}")
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")

if __name__ == "__main__":
    main()
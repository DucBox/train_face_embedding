import os
import boto3
import mxnet as mx
import polars as pl
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
TEST_MODE = True  # Set False để chạy full
NUM_SAMPLES = 100 # Số lượng mẫu để test

CLEAN_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/v1_ivf_real"
OFFSET_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_full_face_align.parquet" 
OUTPUT_PREFIX = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/crawl_test_output/crawl_data_v1"      

S3_CFG = {
    "endpoint_url": "http://s3-data.cyberspace.vn",
    "aws_access_key_id": "ttnt",
    "aws_secret_access_key": "H?3o0nn4Irej",
    "bucket": "ttnt"
}
NUM_THREADS = 8

def get_s3_client():
    return boto3.client("s3", endpoint_url=S3_CFG["endpoint_url"],
                        aws_access_key_id=S3_CFG["aws_access_key_id"],
                        aws_secret_access_key=S3_CFG["aws_secret_access_key"])

def fetch_bytes(client, key, start, length):
    try:
        resp = client.get_object(Bucket=S3_CFG["bucket"], Key=key, Range=f"bytes={start}-{start+length-1}")
        return resp['Body'].read()
    except Exception as e:
        print(f"[ERR] {key}: {e}")
        return None

def main():
    # 1. Load & Join Data
    print("Loading data...")
    df_clean = pl.scan_parquet(os.path.join(CLEAN_DATA_DIR, "*.parquet")).select(["aligned_s3_path", "person_id"])
    
    df_offset = pl.scan_parquet(OFFSET_PARQUET).with_columns(
        (pl.col("tar_path") + pl.lit("/") + pl.col("member_name")).alias("aligned_s3_path")
    )
    
    # Inner Join để lấy offset của những ảnh đã clean
    query = df_clean.join(df_offset, on="aligned_s3_path", how="inner")
    
    if TEST_MODE:
        print(f"--- TEST MODE: Sampling {NUM_SAMPLES} images ---")
        df_final = query.head(NUM_SAMPLES).collect()
    else:
        df_final = query.collect()
        
    data_rows = df_final.to_dicts()
    print(f"Total images to process: {len(data_rows)}")

    # 2. Setup RecordIO Writer
    os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)
    record = mx.recordio.MXIndexedRecordIO(idx_path=OUTPUT_PREFIX + '.idx', uri=OUTPUT_PREFIX + '.rec', flag='w')

    # 3. Parallel Download & Write
    def worker(row):
        s3 = get_s3_client()
        img_bytes = fetch_bytes(s3, row['tar_path'], row['start_byte'], row['length'])
        return row, img_bytes

    print("Starting packing...")
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = {ex.submit(worker, r): r for r in data_rows}
        
        global_idx = 0
        for future in tqdm(as_completed(futures), total=len(data_rows)):
            row, img_bytes = future.result()
            if not img_bytes: continue
            
            # Header: label=person_id, id=index
            header = mx.recordio.IRHeader(0, float(row['person_id']), global_idx, 0)
            packed_s = mx.recordio.pack(header, img_bytes)
            
            record.write_idx(global_idx, packed_s)
            global_idx += 1

    record.close()
    print(f"[DONE] Created {OUTPUT_PREFIX}.rec | Total records: {global_idx}")

if __name__ == "__main__":
    main()

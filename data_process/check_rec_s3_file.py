import os
import boto3
import mxnet as mx
import polars as pl
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
REC_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/crawl_test_output/train_part_000"    
OFFSET_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_full_face_align.parquet" 
CLEAN_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real"

OUTPUT_REC_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/verify_data/from_rec"         # Folder chứa ảnh bung từ .rec
OUTPUT_S3_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/verify_data/from_s3"           # Folder chứa ảnh tải từ S3

S3_CFG = {
    "endpoint": "http://s3-data.cyberspace.vn",
    "ak": "ttnt",
    "sk": "H?3o0nn4Irej",
    "bucket": "ttnt"
}

def get_s3():
    return boto3.client("s3", endpoint_url=S3_CFG["endpoint"],
                        aws_access_key_id=S3_CFG["ak"], aws_secret_access_key=S3_CFG["sk"])

def fetch_s3_bytes(client, key, start, length):
    try:
        resp = client.get_object(Bucket=S3_CFG["bucket"], Key=key, Range=f"bytes={start}-{start+length-1}")
        return resp['Body'].read()
    except: return None

def main():
    # --- PART 1: EXTRACT FROM .REC ---
    print(f"--- 1. Extracting from {REC_PATH}.rec ---")
    os.makedirs(OUTPUT_REC_DIR, exist_ok=True)
    
    record = mx.recordio.MXIndexedRecordIO(REC_PATH + '.idx', REC_PATH + '.rec', 'r')
    extracted_ids = set()
    
    # Duyệt qua tất cả index trong file .idx
    keys = list(record.keys)
    print(f"Found {len(keys)} records.")

    for idx in tqdm(keys):
        item = record.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(item)
        
        person_id = int(header.label)
        extracted_ids.add(person_id)
        
        # Save: person_{id}_{index}.jpg
        fname = f"person_{person_id}_idx{idx}.jpg"
        with open(os.path.join(OUTPUT_REC_DIR, fname), "wb") as f:
            f.write(img_bytes)

    print(f"Extracted unique Person IDs: {len(extracted_ids)}")

    # --- PART 2: DOWNLOAD FROM S3 FOR COMPARISON ---
    print(f"\n--- 2. Downloading original S3 data for these IDs ---")
    os.makedirs(OUTPUT_S3_DIR, exist_ok=True)

    # Load & Filter Data
    df_clean = pl.scan_parquet(os.path.join(CLEAN_DATA_DIR, "*.parquet")).select(["aligned_s3_path", "person_id"])
    df_offset = pl.scan_parquet(OFFSET_PARQUET).with_columns(
        (pl.col("tar_path") + pl.lit("/") + pl.col("member_name")).alias("aligned_s3_path")
    )
    
    # Chỉ lấy dữ liệu của những person_id đã tìm thấy trong file .rec
    df_target = (
        df_clean.join(df_offset, on="aligned_s3_path", how="inner")
        .filter(pl.col("person_id").is_in(list(extracted_ids)))
        .collect()
    )
    
    rows = df_target.to_dicts()
    print(f"Downloading {len(rows)} matching images from S3...")

    def worker(row):
        client = get_s3()
        data = fetch_s3_bytes(client, row['tar_path'], row['start_byte'], row['length'])
        if data:
            # Save: person_{id}_{member_name}
            # Lấy tên file gốc để dễ đối chiếu
            safe_name = os.path.basename(row['member_name'])
            fname = f"person_{row['person_id']}_{safe_name}"
            with open(os.path.join(OUTPUT_S3_DIR, fname), "wb") as f:
                f.write(data)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(tqdm(ex.map(worker, rows), total=len(rows)))

    print("\n[DONE] Verify manually:")
    print(f"1. {OUTPUT_REC_DIR} (Data inside .rec)")
    print(f"2. {OUTPUT_S3_DIR} (Data direct from S3)")

if __name__ == "__main__":
    main()
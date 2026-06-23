
import os
import time
import boto3
import botocore.config
import mxnet as mx
import polars as pl
import numpy as np
from multiprocessing import Process, Queue 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 

NUM_PROCESSES = 32    
THREADS_PER_PROC = 32    

CLEAN_DATA_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real_only_new_id_after_merge_webface_public"
OFFSET_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/offset_map/offset_table_full_face_align.parquet"
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/crawl_v1_shards" 

S3_CFG = { "endpoint": "http://s3-data.cyberspace.vn", "ak": "ttnt", "sk": "<S3_SECRET_KEY>", "bucket": "ttnt" }

def process_shard(shard_id, data_rows, output_dir, progress_queue):
    """Worker chỉ lo tải và bắn tín hiệu tiến độ về Queue"""
    
    # Setup S3 & Writer
    boto_config = botocore.config.Config(max_pool_connections=THREADS_PER_PROC + 5)
    s3_client = boto3.client("s3", endpoint_url=S3_CFG["endpoint"],
                             aws_access_key_id=S3_CFG["ak"], aws_secret_access_key=S3_CFG["sk"], config=boto_config)
    
    prefix = os.path.join(output_dir, f"train_part_{shard_id:03d}")
    record = mx.recordio.MXIndexedRecordIO(prefix + '.idx', prefix + '.rec', 'w')
    
    # Worker function
    def fetch_one(row):
        try:
            resp = s3_client.get_object(Bucket=S3_CFG["bucket"], Key=row['tar_path'], 
                                     Range=f"bytes={row['start_byte']}-{row['start_byte']+row['length']-1}")
            return row['person_id'], resp['Body'].read()
        except: return None, None

    local_idx = 0
    batch_count = 0
    
    with ThreadPoolExecutor(max_workers=THREADS_PER_PROC) as ex:
        futures = [ex.submit(fetch_one, r) for r in data_rows]
        
        for fut in futures:
            pid, img_bytes = fut.result()
            if img_bytes:
                header = mx.recordio.IRHeader(0, pid, local_idx, 0)
                s = mx.recordio.pack(header, img_bytes)
                record.write_idx(local_idx, s)
                local_idx += 1
                
                # Cứ xong 100 ảnh thì báo cáo về mẹ 1 lần (để đỡ nghẽn Queue)
                batch_count += 1
                if batch_count >= 100:
                    progress_queue.put(batch_count)
                    batch_count = 0
        
        # Gửi nốt số lẻ còn lại
        if batch_count > 0:
            progress_queue.put(batch_count)

    record.close()
    # Gửi tín hiệu kết thúc của worker này (None)
    progress_queue.put(None) 

def main():
    # 1. Load Data
    print("Loading Metadata...")
    df_clean = pl.scan_parquet(os.path.join(CLEAN_DATA_DIR, "*.parquet")).select(["aligned_s3_path", "person_id"])
    df_offset = pl.scan_parquet(OFFSET_PARQUET).with_columns((pl.col("tar_path") + pl.lit("/") + pl.col("member_name")).alias("aligned_s3_path"))
    all_rows = df_clean.join(df_offset, on="aligned_s3_path", how="inner").with_columns(pl.col("person_id").cast(pl.Float64)).collect().to_dicts()
    
    total_images = len(all_rows)
    print(f"Total: {total_images} images. Launching {NUM_PROCESSES} processes...")
    
    shards = np.array_split(all_rows, NUM_PROCESSES)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Setup Queue & Processes
    progress_queue = Queue()
    processes = []
    
    for i in range(NUM_PROCESSES):
        shard_data = shards[i].tolist()
        # Truyền queue vào hàm con
        p = Process(target=process_shard, args=(i, shard_data, OUTPUT_DIR, progress_queue))
        p.start()
        processes.append(p)
    
    # 3. Main Loop: Monitoring Progress
    # Thanh này sẽ chạy mượt từ 0 -> 31 triệu
    finished_workers = 0
    with tqdm(total=total_images, unit="img", desc="Total Progress") as pbar:
        while finished_workers < NUM_PROCESSES:
            item = progress_queue.get() # Chờ tin nhắn từ con
            
            if item is None:
                finished_workers += 1 # Một worker đã báo done
            else:
                pbar.update(item) # Cộng thêm số ảnh vừa xong vào thanh tổng

    # 4. Cleanup
    for p in processes:
        p.join()
    print("\n[ALL DONE] All shards completed.")

if __name__ == "__main__":
    main()

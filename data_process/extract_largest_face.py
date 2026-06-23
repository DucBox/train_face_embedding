import pandas as pd
import numpy as np
import multiprocessing
import ast
import os
import time
import shutil

# --- CẤU HÌNH ---
INPUT_PARQUET_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/drop_duplicate"
OUTPUT_PARQUET_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/largest_face"
NUM_WORKERS = 128

def safe_parse(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val
    except:
        return []

def get_largest_face_logic(row):
    try:
        bboxes = safe_parse(row['bboxs'])
        landmarks = safe_parse(row['landmarks'])

        if not bboxes or not landmarks or len(bboxes) != len(landmarks):
            return None, None

        if len(bboxes) == 1:
            return str([bboxes[0]]), str([landmarks[0]])

        # Logic tìm max area
        max_area = 0
        max_idx = 0
        for idx, box in enumerate(bboxes):
            # box: [x1, y1, x2, y2]
            w = box[2] - box[0]
            h = box[3] - box[1]
            area = w * h
            if area > max_area:
                max_area = area
                max_idx = idx
        
        return str([bboxes[max_idx]]), str([landmarks[max_idx]])
    except:
        return None, None

def process_dataframe_chunk(df_chunk):
    results = df_chunk.apply(get_largest_face_logic, axis=1, result_type='expand')
    df_chunk['bboxs'] = results[0]
    df_chunk['landmarks'] = results[1]
    df_chunk = df_chunk.dropna(subset=['bboxs', 'landmarks'])
    
    df_chunk['new_num_faces'] = df_chunk['bboxs'].apply(
        lambda x: len(ast.literal_eval(x)) if x else 0
    )
    return df_chunk

def main():
    start_time = time.time()
    
    # 1. Setup Output
    if os.path.exists(OUTPUT_PARQUET_DIR):
        print(f"[INFO] Cleaning output dir: {OUTPUT_PARQUET_DIR}")
        shutil.rmtree(OUTPUT_PARQUET_DIR)
    os.makedirs(OUTPUT_PARQUET_DIR, exist_ok = True)

    # 2. Load Data (Cách gọn)
    print(f"[INFO] Loading parquet from {INPUT_PARQUET_DIR}...")
    # engine='pyarrow' thường nhanh và ổn định hơn fastparquet
    # cols = ['s3_path', 'phash', 'bboxs', 'landmarks', 'num_faces', 'name', 'width', 'height']
    df = pd.read_parquet(INPUT_PARQUET_DIR, engine='pyarrow')

    initial_count = len(df)
    print(f"[INFO] Loaded {initial_count} rows.")

    # 3. Fast Filter
    print("[INFO] Filtering metadata...")
    df = df[(df['num_faces'] > 0)]
    # df = df.drop_duplicates(subset=['phash'])
    print(f" -> Remaining rows: {len(df)}")

    filter_time = time.time() - start_time
    print(f"Load and filer in {filter_time:.2f}s")

    start_time = time.time()
    # 4. Multiprocessing
    print(f"[INFO] Processing largest face logic with {NUM_WORKERS} workers...")
    chunks = np.array_split(df, NUM_WORKERS * 64)
    
    processed_chunks = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap(process_dataframe_chunk, chunks)):
            processed_chunks.append(res)
            if (i+1) % 1000 == 0: print(f" -> Processed chunk {i+1}")

    # 5. Save
    print("[INFO] Saving result...")
    cleaned_df = pd.concat(processed_chunks, ignore_index=True)
    print(f"[INFO] Final size: {len(cleaned_df)} (Removed {initial_count - len(cleaned_df)})")

    cleaned_df.to_parquet(os.path.join(OUTPUT_PARQUET_DIR, "all_data.parquet"), index=False)
    print(len(cleaned_df))
    print(cleaned_df.columns)
    print(f"[SUCCESS] Done in {time.time() - start_time:.2f}s")

    print(cleaned_df.head(2))

if __name__ == "__main__":
    main()
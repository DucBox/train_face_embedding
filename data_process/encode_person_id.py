import pandas as pd
import os
import numpy as np

# --- CONFIG ---
# Đường dẫn file parquet sạch đã clean trước đó
INPUT_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/phash_unique_metadata.parquet"
# Đường dẫn file output
OUTPUT_PARQUET = "/workspace/FaceNist/raw_data_processing/output_parquet/phash_unique_metadata_with_id.parquet"
# Lưu riêng file map để tra cứu sau này (Optional)
MAPPING_CSV = "/workspace/FaceNist/raw_data_processing/output_parquet/person_id_name_mapping.csv"

def main():
    if not os.path.exists(INPUT_PARQUET):
        print(f"[ERROR] Input file not found: {INPUT_PARQUET}")
        return

    # 1. Load Data
    print(f"[INFO] Reading parquet: {INPUT_PARQUET}...")
    # Nếu RAM ít, chỉ nên read các cột cần thiết, nhưng ở đây cần lưu lại full nên read hết
    df = pd.read_parquet(INPUT_PARQUET)
    
    print(f"[INFO] Loaded {len(df)} rows.")

    if 'name' not in df.columns:
        print("[ERROR] Column 'name' not found in dataframe.")
        return

    # 2. Xử lý NaN (nếu có)
    # Nếu name bị null, fill bằng 'unknown' để tránh lỗi
    if df['name'].isnull().any():
        print(f"[WARN] Found {df['name'].isnull().sum()} null names. Filling with 'unknown'.")
        df['name'] = df['name'].fillna('unknown')

    # 3. Encode Name -> Person ID
    print("[INFO] Encoding Person IDs...")
    
    # Sử dụng factorize với sort=True để ID tăng dần theo thứ tự Alphabet của tên
    # person_0 -> adam, person_1 -> alex ...
    ids, uniques = pd.factorize(df['name'], sort=True)
    
    df['person_id'] = ids.astype(np.int32) # Lưu int32 cho nhẹ

    # 4. Save Mapping (Rất quan trọng để debug)
    print(f"[INFO] Saving mapping table to {MAPPING_CSV}...")
    mapping_df = pd.DataFrame({
        'person_id': range(len(uniques)),
        'name': uniques
    })
    mapping_df.to_csv(MAPPING_CSV, index=False)

    # 5. Save Parquet
    print(f"[INFO] Saving result to {OUTPUT_PARQUET}...")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    # 6. Verify & Stats
    print("-" * 40)
    print("RESULTS:")
    print(f" - Total Rows: {len(df)}")
    print(f" - Total Unique Persons: {len(uniques)}")
    print(f" - Person ID Range: 0 to {len(uniques)-1}")
    print("\nSample Data:")
    print(df[['name', 'person_id', 's3_path']].head(5))
    print("-" * 40)

if __name__ == "__main__":
    main()
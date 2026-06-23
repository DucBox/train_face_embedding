import pandas as pd
import glob
import os

OUTPUT_RESULTS_DIR = "/workspace/FaceNist/raw_data_processing/metadata_results_all"

# Lấy danh sách tất cả file parquet
files = glob.glob(os.path.join(OUTPUT_RESULTS_DIR, "*.parquet"))

print(f"Scanning {len(files)} files to fix column names...")

count_fixed = 0

for f in files:
    try:
        # 1. Đọc file (đọc meta data hoặc load hết)
        df = pd.read_parquet(f)
        
        # 2. Kiểm tra xem có cột sai tên 'image_s3_path' không
        if 'image_s3_path' in df.columns:
            # 3. Đổi tên thành 'bboxs'
            df = df.rename(columns={'image_s3_path': 's3_path'})
            
            # 4. Ghi đè lại file cũ (Overwrite)
            # Dùng engine pyarrow hoặc fastparquet
            df.to_parquet(f, index=False)
            
            count_fixed += 1
            print(f" -> Fixed file: {os.path.basename(f)}")
            
    except Exception as e:
        print(f"Error reading file {f}: {e}")

print(f"Done! Total files fixed: {count_fixed}")

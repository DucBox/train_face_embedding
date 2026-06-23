import pandas as pd

# Thay đường dẫn file parquet clean của bạn vào đây
PARQUET_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/largest_face/data_0_1000_3500_4500_9500.parquet" 

def check_broken_tar():
    try:
        # Chỉ load cột s3_path để check cho nhanh
        print(f"[INFO] Reading {PARQUET_PATH}...")
        df = pd.read_parquet(PARQUET_PATH, columns=['s3_path'])
        
        # Target cần tìm
        target = "data_1000/01116.tar"
        
        # Filter xem có dòng nào chứa target không
        matches = df[df['s3_path'].str.contains(target, regex=False)]
        
        count = len(matches)
        print("-" * 30)
        print(f"Checking for: {target}")
        
        if count == 0:
            print(f"-> RESULT: 0 rows found.")
            print("=> CONFIRMED: File lỗi đã bị loại bỏ hoàn toàn khỏi clean data.")
        else:
            print(f"-> RESULT: Found {count} rows.")
            print("=> WARNING: Vẫn còn dữ liệu sót lại. Có thể file chỉ bị lỗi phần đuôi, phần đầu vẫn đọc được.")
            print("Sample paths:")
            print(matches['s3_path'].head().values)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_broken_tar()
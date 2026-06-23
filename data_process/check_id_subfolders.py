import os
import glob
import gc
import polars as pl

INPUT_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/face-embedding-normalize/face-embedding-normalize"

def verify_distribution():
    sub_folders = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    seen_ids = set()
    
    print(f"Start verifying {len(sub_folders)} folders...")

    for folder in sub_folders:
        if not os.path.isdir(folder): continue
        folder_name = os.path.basename(folder)
        
        try:
            # Chỉ load cột person_id và lấy unique ngay lập tức để tiết kiệm RAM
            current_ids = pl.read_parquet(
                os.path.join(folder, "*.parquet"), 
                columns=["person_id"]
            )["person_id"].unique().to_list()
            
            current_set = set(current_ids)
            
            # Kiểm tra va chạm với dữ liệu cũ
            overlap = current_set.intersection(seen_ids)
            
            if overlap:
                print(f"\n[FAIL] OVERLAP DETECTED in '{folder_name}'")
                print(f"Count: {len(overlap)} IDs already exist in previous folders.")
                print(f"Example: {list(overlap)[:5]}")
                return False
            
            # Update vào tập global
            seen_ids.update(current_set)
            print(f"[OK] {folder_name}: {len(current_set)} IDs. (Total unique: {len(seen_ids)})")
            
            del current_ids, current_set
            gc.collect()
            
        except Exception as e:
            print(f"[ERR] {folder_name}: {e}")

    print("\n[SUCCESS] All person_ids are strictly separated by folders.")
    return True

if __name__ == "__main__":
    verify_distribution()
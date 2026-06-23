import os
import glob
import re

def rename_training_files(folder_path):
    # Tìm tất cả các file .rec có định dạng train_part_XXX.rec
    rec_files = sorted(glob.glob(os.path.join(folder_path, "train_part_*.rec")))
    
    if not rec_files:
        print(f"Không tìm thấy file nào khớp định dạng train_part_*.rec tại {folder_path}")
        return

    print(f"Đang xử lý {len(rec_files)} cặp file...")

    for i, old_rec_path in enumerate(rec_files, start=1):
        # Đường dẫn file .idx tương ứng
        old_idx_path = old_rec_path.replace(".rec", ".idx")
        
        # Tên mới: train_1.rec, train_2.rec, ...
        new_rec_name = f"train_{i}.rec"
        new_idx_name = f"train_{i}.idx"
        
        new_rec_path = os.path.join(folder_path, new_rec_name)
        new_idx_path = os.path.join(folder_path, new_idx_name)

        # Thực hiện đổi tên file .rec
        os.rename(old_rec_path, new_rec_path)
        
        # Thực hiện đổi tên file .idx (nếu tồn tại)
        if os.path.exists(old_idx_path):
            os.rename(old_idx_path, new_idx_path)
            print(f"Renamed: {os.path.basename(old_rec_path)} -> {new_rec_name}")
        else:
            print(f"Warning: Không tìm thấy file index cho {os.path.basename(old_rec_path)}")

    print("Hoàn thành đổi tên.")

if __name__ == "__main__":
    # Thay đổi đường dẫn này tới folder chứa dữ liệu crawl đã re-index
    TARGET_FOLDER = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/re_index_v1_shards"
    rename_training_files(TARGET_FOLDER)
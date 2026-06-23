import pandas as pd

print("Loading Phash data...")
df_phash = pd.read_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/phash_unique_metadata.parquet")
print(len(df_phash))

df_phash = df_phash.drop_duplicates(subset = ['phash'])
print(len(df_phash))

print("Loading Face Results data...")
df_results = pd.read_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/face_detect")
print(len(df_results))

print("Merging dataframes...")
merged_df = pd.merge(df_results, df_phash, on='s3_path', how='inner')
merged_df.to_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/drop_duplicate/all_data.parquet", index = False)
print(len(merged_df))
print(merged_df.head(5))
print(merged_df.columns)

# print("Merging dataframes...")

# how='inner': Chỉ giữ lại ảnh có thông tin ở cả 2 bên
# on='s3_path': Ghép nối dựa trên cột này
# merged_df = pd.merge(df_results, df_phash, on='s3_path', how='inner')

# print(f"Merged Total: {len(merged_df)} rows")

# merged_df.to_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/drop_duplicate/data_0_1000_3500_4500_9500.parquet", index = False)

# Tìm những s3_path có trong Results nhưng KHÔNG có trong Phash
# Lưu ý: Chuyển về set để tìm kiếm nhanh hơn với dữ liệu lớn (42M vs 180M)
# phash_paths = set(df_phash['s3_path'])
# missing_paths = df_results[~df_results['s3_path'].isin(phash_paths)]

# print(f"Số lượng ảnh không tìm thấy phash: {len(missing_paths)}")
# print(missing_paths['s3_path'].head(20))

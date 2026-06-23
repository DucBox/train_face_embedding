import pandas as pd

print("Loading Phash data...")
df_phash = pd.read_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/drop_duplicate/data_0_1000_3500_4500_9500.parquet")
print(len(df_phash))

df_phash = df_phash.drop_duplicates(subset = ['phash'])
print(len(df_phash))
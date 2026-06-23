import pandas as pd
pd.set_option('display.max_colwidth', None)
df = pd.read_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/phash_unique_metadata.parquet")

print(len(df))

# print(df.head(20))

print(df.columns)
# print(df.new_num_faces.unique)

# df_metadata = pd.read_parquet("/workspace/FaceNist/raw_data_processing/metadata_results_all")
# print(len(df_metadata))

# print(df.tail(200))

# sample = df[df['image_s3_path']=='cv/crawled-datasets/face-google-search/data_0/01126.tar/011260003.jpg']
# print(sample)
# df_unique = df.drop_duplicates(subset = ['phash'])
# print(len(df_unique))

# print(len(df_unique)) /home/ducnq7/FaceNist/raw_data_processing/phash_results_parquet_all
 



import polars as pl
import numpy as np
from sklearn.preprocessing import normalize
import os
import glob
from tqdm import tqdm
import gc
import re

def normalize_file(file_path, output_dir):
    print(f"Reading: {os.path.basename(file_path)}")
    df = pl.read_parquet(file_path)  # CHỈ 1 FILE
    
    embeddings = df["embedding"].to_list()
    normalized = []
    batch_size = 10000
    
    for i in range(0, len(embeddings), batch_size):
        
        batch = np.array(embeddings[i:i + batch_size])
        normalized.extend(normalize(batch, axis=1, norm='l2').tolist())
    
    df = df.with_columns(pl.Series("embedding_normalized", normalized)).sort("person_id")
    
    base_name = os.path.basename(file_path)
    out_path = os.path.join(output_dir, base_name.replace('.parquet', '_norm.parquet'))
    df.write_parquet(out_path)
    print(f"Saved: {os.path.basename(out_path)}")
    
    del df, embeddings, normalized
    gc.collect()

base_input_dir = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding"
base_output_dir = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize"
os.makedirs(base_output_dir, exist_ok = True)

for folder_name in os.listdir(base_input_dir):
    if folder_name.startswith("embeddings_output_"):
        input_folder = os.path.join(base_input_dir, folder_name)
        output_dir = os.path.join(base_output_dir, f"normalized_{folder_name}")
        
        print(f"\n🔄 Processing folder: {folder_name} → {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(input_folder, "*.parquet")))
        print(f"  Found {len(files)} files")
        
        for file in tqdm(files, desc=f"{folder_name}", leave=False):
            normalize_file(file, output_dir)
        
        print(f"✅ {folder_name} DONE")

print("🎉 ALL FOLDERS COMPLETED!")
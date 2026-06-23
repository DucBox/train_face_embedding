import os
import glob
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

INPUT_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/public5m_embedding_normalize"
OUTPUT_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_public5m/centers.parquet"
EMBEDDING_COL = "embedding_normalized"
EXPECTED_DIM = 512

def main():
    if os.path.exists(OUTPUT_FILE):
        try: os.remove(OUTPUT_FILE)
        except: pass

    files = sorted(glob.glob(os.path.join(INPUT_ROOT, "public5m_normalized_public5m*", "*.parquet")))
    if not files: 
        print("No files found")

    print(f"Loading {len(files)} files...")
    
    df_list = []
    for f in files:
        try:
            print(f"Reading file {f} ...")
            d = pd.read_parquet(f, columns=["person_id", EMBEDDING_COL])
            d = d.dropna()
            df_list.append(d)
        except:
            print(f"Error reading {f}")
            pass
    
    if not df_list: 
        print("No df list")

    print("Concatenating...")
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()

    print("Sorting...")
    df = df.sort_values("person_id")

    print("Converting to Numpy...")
    try:
        matrix = np.vstack(df[EMBEDDING_COL].tolist()).astype(np.float32)
    except Exception as e:
        print(f"OOM or Convert Error: {e}")
        sys.exit(1)

    if matrix.shape[1] != EXPECTED_DIM:
        print("Dimension mismatch")
        sys.exit(1)

    ids = df["person_id"].values
    del df
    gc.collect()

    print("Aggregating...")
    unique_ids, start_indices = np.unique(ids, return_index=True)
    
    end_indices = np.append(start_indices[1:], len(ids))
    counts = end_indices - start_indices
    
    sum_vectors = np.add.reduceat(matrix, start_indices)
    
    print("Normalizing...")
    centers = normalize(sum_vectors, axis=1)

    print("Saving...")
    final_df = pd.DataFrame({
        "person_id": unique_ids,
        "img_count": counts,
        "embedding_center": list(centers)
    })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Done. Total IDs: {len(final_df)}")

if __name__ == "__main__":
    main()
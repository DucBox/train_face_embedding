import pandas as pd
import os
import sys

TARGET_FILE = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_clean/template_v1/centers.parquet"
OFFSET = 3_000_000

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"File not found: {TARGET_FILE}")
        sys.exit(1)

    print(f"Reading {TARGET_FILE}...")
    df = pd.read_parquet(TARGET_FILE)

    print(f"Applying offset {OFFSET}...")
    df['original_person_id'] = df['person_id']
    df['person_id'] = df['person_id'] + OFFSET

    cols = ['person_id', 'original_person_id'] + [c for c in df.columns if c not in ['person_id', 'original_person_id']]
    df = df[cols]

    print("Saving...")
    df.to_parquet(TARGET_FILE, index=False)
    
    print(f"Done. Sample ID: {df.iloc[0]['person_id']} (Old: {df.iloc[0]['original_person_id']})")

if __name__ == "__main__":
    main()
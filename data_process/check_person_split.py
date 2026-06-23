import polars as pl
import glob
import os
from collections import Counter
from tqdm import tqdm

INPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data_merge/v1_ivf_real"
OUTPUT_CSV = os.path.join(INPUT_DIR, "duplicate_person_ids.csv")

def check():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    print(f"Total files: {len(files)}")

    person_file_count = Counter()

    for f in tqdm(files):
        unique_ids = (
            pl.scan_parquet(f)
            .select("person_id")
            .unique()
            .collect()
            ["person_id"]
            .to_list()
        )
        person_file_count.update(unique_ids)

    total = len(person_file_count)
    duplicates = {pid: cnt for pid, cnt in person_file_count.items() if cnt >= 2}
    dup_count = len(duplicates)

    print(f"\nTotal unique person_ids : {total}")
    print(f"Person IDs in >= 2 files: {dup_count} ({dup_count/total*100:.2f}%)")

    # Distribution: bao nhiêu người nằm ở đúng N files
    dist = Counter(duplicates.values())
    print("\nDistribution (file_count → số người):")
    for n_files in sorted(dist):
        print(f"  {n_files} files: {dist[n_files]} persons")

    # Ghi toàn bộ ra CSV (person_id, file_count)
    if duplicates:
        pl.DataFrame({
            "person_id": list(duplicates.keys()),
            "file_count": list(duplicates.values()),
        }).sort("file_count", descending=True).write_csv(OUTPUT_CSV)
        print(f"\nSaved all duplicates to: {OUTPUT_CSV}")

if __name__ == "__main__":
    check()

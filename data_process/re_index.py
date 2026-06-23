import os
import glob
import mxnet as mx
import numpy as np
import csv
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

MAX_WORKERS = 64

def get_record_files(folder_path):
    rec_files = sorted(glob.glob(os.path.join(folder_path, "*.rec")))
    pairs = []
    for rec in rec_files:
        idx = rec.replace(".rec", ".idx")
        if os.path.exists(idx):
            pairs.append((idx, rec))
    return pairs

def scan_worker(file_pair, base_max_id):
    idx_path, rec_path = file_pair
    overflow_ids = set()
    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    
    for key in record.keys:
        s = record.read_idx(key)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, (int, float, np.int64, np.float32)):
            label = label[0]
        
        if label > base_max_id:
            overflow_ids.add(int(label))
    return overflow_ids

def rewrite_worker(file_pair, output_folder, mapping, base_max_id):
    idx_path, rec_path = file_pair
    filename = os.path.basename(rec_path)
    idx_filename = os.path.basename(idx_path)
    
    os.makedirs(output_folder, exist_ok=True)
    out_rec_path = os.path.join(output_folder, filename)
    out_idx_path = os.path.join(output_folder, idx_filename)
    
    source_record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    target_record = mx.recordio.MXIndexedRecordIO(out_idx_path, out_rec_path, 'w')
    
    for key in source_record.keys:
        s = source_record.read_idx(key)
        header, img_bytes = mx.recordio.unpack(s)
        
        old_label = header.label
        if not isinstance(old_label, (int, float, np.int64, np.float32)):
            old_label = old_label[0]
        old_label = int(old_label)
        
        new_label = mapping.get(old_label, old_label) if old_label > base_max_id else old_label
        
        new_header = mx.recordio.IRHeader(header.flag, new_label, header.id, header.id2)
        packed_s = mx.recordio.pack(new_header, img_bytes)
        target_record.write_idx(key, packed_s)
        
    target_record.close()
    return f"Processed {filename}"

def main_reindex(base_max_id, input_folder, output_folder):
    file_pairs = get_record_files(input_folder)
    if not file_pairs:
        print("No .rec/.idx files found.")
        return

    print(f"--- PHASE 1: Scanning for IDs > {base_max_id} ---")
    unique_overflow_ids = set()
    with Pool(processes=MAX_WORKERS) as pool:
        scan_func = partial(scan_worker, base_max_id=base_max_id)
        results = list(tqdm(pool.imap_unordered(scan_func, file_pairs), total=len(file_pairs)))
    
    for res in results:
        unique_overflow_ids.update(res)
    
    sorted_ids = sorted(list(unique_overflow_ids))
    mapping = {old_id: (base_max_id + 1 + i) for i, old_id in enumerate(sorted_ids)}
    
    # Save mapping to CSV
    mapping_path = os.path.join(output_folder, "id_mapping.csv")
    os.makedirs(output_folder, exist_ok=True)
    with open(mapping_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original_id", "new_id"])
        for old_id, new_id in mapping.items():
            writer.writerow([old_id, new_id])
    
    print(f"Mapping saved to {mapping_path}")
    print(f"New ID range: {base_max_id + 1} -> {base_max_id + len(sorted_ids)}")

    print(f"--- PHASE 2: Rewriting files ---")
    with Pool(processes=MAX_WORKERS) as pool:
        rewrite_func = partial(rewrite_worker, output_folder=output_folder, mapping=mapping, base_max_id=base_max_id)
        list(tqdm(pool.imap(rewrite_func, file_pairs), total=len(file_pairs)))

    print(f"DONE. Total unique classes now: {base_max_id + 1 + len(sorted_ids)}")

if __name__ == "__main__":
    BASE_MAX_ID = 2616672
    INPUT_FOLDER = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/crawl_v1_shards"
    OUTPUT_FOLDER = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/train_data/version01_only_new_id_after_webface_public/re_index_v1_shards"
    main_reindex(BASE_MAX_ID, INPUT_FOLDER, OUTPUT_FOLDER)
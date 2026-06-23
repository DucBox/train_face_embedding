import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)
import numpy as np
import polars as pl
import boto3
import tarfile
import io
import time
import ast
from multiprocessing import Pool
from tqdm.auto import tqdm
from skimage import transform as trans

CHECKPOINT_FILE = "output_parquet/data_process/checkpoints/checkpoint_done_pids.txt"

INPUT_FACES_PARQUET = "output_parquet/data_process/top_N_faces/top5_faces_exploded.parquet"
INPUT_OFFSET_PARQUET = "output_parquet/data_process/offset_map/offset_table_full.parquet" 

OUTPUT_BUCKET = "ttnt"
OUTPUT_ROOT_PREFIX = "cv/processed-datasets/aligned_face_112_112"
NUM_WORKERS = 32
SHARD_SIZE = 1000

S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "H?3o0nn4Irej"

arcface_src = np.array(
    [[30.29459953, 51.69630051], [65.53179932, 51.50139999], [48.02519989, 71.73660278],
     [33.54930115, 92.3655014], [62.72990036, 92.20410156]],
    dtype=np.float32)
arcface_src[:, 0] += 10.0
arcface_src = np.expand_dims(arcface_src, axis=0)

def estimate_norm(lmk, image_size=112):
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_error = float('inf')
    src = arcface_src
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T).T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
    return min_M

def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def align_face(image, landmark):
    if len(landmark) == 0: return None
    pts5 = np.array(landmark, dtype=np.float32).reshape(5, 2)
    nimg = norm_crop(image, pts5)
    return nimg

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=boto3.session.Config(max_pool_connections=10, retries={'max_attempts': 3})
    )

def download_range(client, bucket, key, start, length):
    end = start + length - 1
    resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp['Body'].read()

def worker_process_person(args):
    person_id, shard_folder, df_group = args
    rows = df_group.rows()
    s3 = get_s3_client()
    tar_buffer = io.BytesIO()
    success_count = 0
    
    try:
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for row in rows:
                _, s3_path, face_idx, _, lmk_data, tar_key, offset, length = row
                try:
                    img_bytes = download_range(s3, OUTPUT_BUCKET, tar_key, offset, length)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None: continue
                    
                    if isinstance(lmk_data, str): lmk_data = ast.literal_eval(lmk_data)
                    
                    aligned_img = align_face(img, lmk_data)
                    if aligned_img is None: continue

                    ret, encoded_img = cv2.imencode('.jpg', aligned_img)
                    if not ret: continue
                    
                    img_io = io.BytesIO(encoded_img.tobytes())

                    safe_filename = s3_path.replace("/", "_")
                    tar_name = f"{safe_filename}_face_{face_idx}.jpg"
                    
                    tarinfo = tarfile.TarInfo(name=tar_name)
                    tarinfo.size = len(img_io.getvalue())
                    tar.addfile(tarinfo, img_io)
                    success_count += 1
                except Exception:
                    continue

        if success_count > 0:
            tar_buffer.seek(0)
            output_key = f"{OUTPUT_ROOT_PREFIX}/{shard_folder}/person_{person_id:07d}.tar"
            s3.put_object(Bucket=OUTPUT_BUCKET, Key=output_key, Body=tar_buffer)
            return person_id, 1
            
    except Exception as e:
        print(f"[ERR] Person {person_id}: {e}")
        return 0
    return 0

def args_generator(tasks_map, done_pids):
    for pid_key, df_group in tasks_map.items():
        pid = pid_key[0]

        if pid in done_pids:
            continue

        start_shard = (pid // SHARD_SIZE) * SHARD_SIZE
        end_shard = start_shard + SHARD_SIZE - 1
        shard_folder = f"person_{start_shard}_{end_shard}"

        yield (pid, shard_folder, df_group)

if __name__ == "__main__":
    t0 = time.time()

    print("[INIT] Loading Data...")
    df_faces = pl.read_parquet(INPUT_FACES_PARQUET)
    df_offset = pl.read_parquet(INPUT_OFFSET_PARQUET)
    
    print(f"[INFO] Test Faces: {df_faces.height}")

    df_offset = df_offset.with_columns(
        (pl.col("tar_path") + "/" + pl.col("member_name")).alias("s3_path")
    )
    
    df_master = df_faces.join(df_offset, on="s3_path", how="inner")
    del df_faces
    del df_offset
    
    import gc
    gc.collect()

    df_master = df_master.select([
        "person_id", "s3_path", "face_index", "bbox", "landmark",
        "tar_path", "start_byte", "length"
    ])

    print(f"[INFO] Master Plan Ready: {df_master.height} faces. Sorting and Partitioning...")
    tasks_map = df_master.partition_by("person_id", as_dict=True, maintain_order=True)
    del df_master
    import gc
    gc.collect()
    print(f"[INFO] Total Unique Persons: {len(tasks_map)}")
    
    done_pids = set()
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Open checkpoint file")
        with open(CHECKPOINT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    done_pids.add(int(line.strip()))
    else:
        print(f"No checkpoint file")
        open(CHECKPOINT_FILE, 'a').close()

    if os.path.exists(CHECKPOINT_FILE):
        print(f"File already created")
        
    print(f"FOUND {len(done_pids)} already processed and saved in s3")

    total_tasks_estimated = len(tasks_map) - len(done_pids)
    print(f"[RUN] Starting Pool with {NUM_WORKERS} workers to process {total_tasks_estimated} tasks...")
    
    success_count = 0
    with open(CHECKPOINT_FILE, "a") as f_ckpt:
        with Pool(processes=NUM_WORKERS) as pool:
            task_gen = args_generator(tasks_map, done_pids)
            for pid_done, status in tqdm(pool.imap_unordered(worker_process_person, task_gen, chunksize=20), total=total_tasks_estimated):
                if status == 1:
                    success_count += 1
                    f_ckpt.write(f"{pid_done}\n")
                    f_ckpt.flush()
            
    print("-" * 40)
    print(f"DEMO COMPLETED in {time.time() - t0:.2f}s")
    print(f"Persons Uploaded: {success_count}")
    print(f"S3 Output: s3://{OUTPUT_BUCKET}/{OUTPUT_ROOT_PREFIX}")
    print("-" * 40)
import os
import time
import io
import tarfile
import cv2
import torch
import numpy as np
import polars as pl
import boto3
from tqdm import tqdm
from backbones import get_model

# --- CONFIGURATION ---
S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "H?3o0nn4Irej"
BUCKET_NAME = "ttnt"
ROOT_PREFIX = "cv/processed-datasets/aligned_face_112_112"

MODEL_NAME = "vit_l_depth36"
MODEL_WEIGHT_PATH = "/workspace/data/code/arcface_torch/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_CHECKPOINT = "/workspace/data/crawled-dataset-processing/face-embedding/checkpoints_align_7.txt"
EMBED_CHECKPOINT = "/workspace/data/crawled-dataset-processing/face-embedding/checkpoints_embedding_7.txt"
OUTPUT_DIR = "/workspace/data/crawled-dataset-processing/face-embedding/embeddings_output_7"
SHARD_SIZE = 1000
SAVE_BATCH_SIZE = 200000     # Số dòng tích lũy để ghi parquet
INFERENCE_BATCH_SIZE = 256 # Batch size cố định cho GPU

# --- UTILS ---

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def load_face_model():
    print(f"[INIT] Loading model {MODEL_NAME} to {DEVICE}...")
    net = get_model(MODEL_NAME, fp16=False)
    net.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
    net.eval()
    net.to(DEVICE)
    return net

def get_shard_folder(person_id):
    start = (person_id // SHARD_SIZE) * SHARD_SIZE
    end = start + SHARD_SIZE - 1
    return f"person_{start}_{end}"

def preprocess_image(img_bgr):
    if img_bgr is None: return None
    img = cv2.resize(img_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).float()
    tensor.div_(255).sub_(0.5).div_(0.5)
    return tensor

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    net = load_face_model()
    s3 = get_s3_client()

    processed_pids = set()
    if os.path.exists(EMBED_CHECKPOINT):
        with open(EMBED_CHECKPOINT, 'r') as f:
            for line in f:
                if line.strip():
                    processed_pids.add(int(line.strip()))
    
    buffer_data = []
    buffer_pids = set()
    file_counter = len(os.listdir(OUTPUT_DIR))

    print("[RUN] Waiting for data...")
    
    while True:
        current_crop_pids = set()
        if os.path.exists(CROP_CHECKPOINT):
            print(f"Open crop align checkpoints")
            with open(CROP_CHECKPOINT, 'r') as f:
                for line in f:
                    if line.strip():
                        current_crop_pids.add(int(line.strip()))
        
        else:
            print(f"No crop align checkpoint")
        
        todo_pids = list(current_crop_pids - processed_pids)
        todo_pids.sort()

        if not todo_pids:
            time.sleep(5)
            continue

        print(f"[INFO] Processing {len(todo_pids)} new persons...")

        for pid in tqdm(todo_pids, desc="Embedding"):
            shard_folder = get_shard_folder(pid)
            tar_key = f"{ROOT_PREFIX}/{shard_folder}/person_{pid:07d}.tar"
            
            try:
                # print(f"[DEBUG] Looking for: {tar_key}")
                obj = s3.get_object(Bucket=BUCKET_NAME, Key=tar_key)
                tar_bytes = io.BytesIO(obj['Body'].read())
                
                batch_tensors = []
                batch_paths = []
                
                with tarfile.open(fileobj=tar_bytes, mode='r') as tar:
                    for member in tar.getmembers():
                        if not member.isfile(): continue
                        f = tar.extractfile(member)
                        if f is None: continue
                        
                        file_bytes = np.frombuffer(f.read(), np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        tensor = preprocess_image(img_bgr)
                        
                        if tensor is not None:
                            batch_tensors.append(tensor)
                            batch_paths.append(f"{tar_key}/{member.name}")

                if not batch_tensors:
                    processed_pids.add(pid)
                    continue

                # --- FIXED BATCH INFERENCE LOGIC ---
                total_imgs = len(batch_tensors)
                pid_embeddings = []
                
                # Loop through chunks
                for i in range(0, total_imgs, INFERENCE_BATCH_SIZE):
                    chunk_tensors = batch_tensors[i : i + INFERENCE_BATCH_SIZE]
                    input_batch = torch.cat(chunk_tensors).to(DEVICE)
                    
                    with torch.no_grad():
                        feat = net(input_batch).cpu().numpy()
                    
                    pid_embeddings.append(feat)
                
                # Concatenate all chunks results
                if pid_embeddings:
                    final_embeddings = np.concatenate(pid_embeddings, axis=0)
                    
                    for path, emb in zip(batch_paths, final_embeddings):
                        buffer_data.append({
                            "person_id": pid,
                            "aligned_s3_path": path,
                            "embedding": emb.tolist()
                        })
                # -----------------------------------

                buffer_pids.add(pid)
                processed_pids.add(pid)

                if len(buffer_data) >= SAVE_BATCH_SIZE:
                    df = pl.DataFrame(buffer_data)
                    save_path = os.path.join(OUTPUT_DIR, f"embed_part_{file_counter:04d}.parquet")
                    df.write_parquet(save_path)
                    print(f"[SAVE] Saved {len(df)} rows.")
                    
                    with open(EMBED_CHECKPOINT, 'a') as f:
                        for p in buffer_pids:
                            f.write(f"{p}\n")
                    
                    buffer_data = []
                    buffer_pids = set()
                    file_counter += 1

            except Exception as e:
                print(f"[ERR] PID {pid}: {e}")
                continue

        if buffer_data:
            df = pl.DataFrame(buffer_data)
            save_path = os.path.join(OUTPUT_DIR, f"embed_part_{file_counter:04d}.parquet")
            df.write_parquet(save_path)
            
            with open(EMBED_CHECKPOINT, 'a') as f:
                for p in buffer_pids:
                    f.write(f"{p}\n")
            
            buffer_data = []
            buffer_pids = set()
            file_counter += 1

if __name__ == "__main__":
    main()

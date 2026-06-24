"""
Stage 01b — EMBED crawl (197M aligned crops on S3) with the NEW model.
Single responsibility: per-person tar on S3 -> backbone -> raw embeddings parquet.

Multi-GPU: the per-person id list is sharded across RANK (one process / GPU).
PER-ID checkpoint: every fully-embedded id is appended to a .txt; on resume,
ids already in the txt are skipped. Buffers flush to a parquet chunk every
embed_flush_rows images (RAM cap).

    torchrun --nproc_per_node=8 embed_s3.py

Output schema: person_id, src='crawl', img_key=aligned_s3_path, embedding[512].
Real mode only (boto3 + torch); test mode uses fixtures instead.
"""
from __future__ import annotations

import io
import os
import re
import sys
import tarfile

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (COL_ID, COL_SRC, COL_KEY, COL_EMB, ckpt_load, ckpt_append,
                    write_emb_parquet, log)
from config import CFG, CRAWL


def _list_pids():
    """Distinct crawl person ids, parsed from the offset_map tar paths."""
    df = pl.read_parquet(CFG.offset_map_path, columns=["tar_path"]).unique()
    pids = set()
    for p in df["tar_path"].to_list():
        m = re.search(r"person_(\d+)\.tar", p)
        if m:
            pids.add(int(m.group(1)))
    return sorted(pids)


def run():
    import cv2
    import boto3
    import torch
    from backbones import get_model

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = CFG.dir_embed(CRAWL)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = CFG.ckpt("embed", CRAWL + f"_r{rank}")
    done = ckpt_load(ckpt)

    pids = [p for i, p in enumerate(_list_pids()) if i % world == rank and p not in done]
    log(f"[embed_s3] rank{rank}: {len(pids):,} pids to embed")

    s3 = boto3.client("s3", endpoint_url=CFG.s3_endpoint,
                      aws_access_key_id=CFG.s3_access_key,
                      aws_secret_access_key=CFG.s3_secret_key)
    net = get_model(CFG.network, fp16=False, num_features=CFG.embedding_size).to(device).eval()
    net.load_state_dict(torch.load(CFG.model_weight, map_location="cpu"), strict=False)

    def prep(b):
        img = cv2.cvtColor(cv2.resize(b, (112, 112)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        return t.div_(255).sub_(0.5).div_(0.5)

    buf_id, buf_key, buf_emb, just_done, chunk = [], [], [], [], len(os.listdir(out_dir))
    bs = CFG.embed_batch_size
    total_imgs = 0

    def flush():
        nonlocal buf_id, buf_key, buf_emb, just_done, chunk
        if buf_id:
            meta = pl.DataFrame({COL_ID: buf_id, COL_SRC: [CRAWL] * len(buf_id), COL_KEY: buf_key})
            write_emb_parquet(
                os.path.join(out_dir, f"part-rank{rank}-{chunk:05d}.parquet"),
                meta, np.concatenate(buf_emb, axis=0), COL_EMB)
            chunk += 1
        ckpt_append(ckpt, just_done)
        buf_id, buf_key, buf_emb, just_done = [], [], [], []

    pbar = tqdm(pids, desc=f"embed_s3 r{rank}", unit="id", position=rank)
    for pid in pbar:
        shard = (pid // CFG.crawl_shard_size) * CFG.crawl_shard_size
        key = f"{CFG.crawl_s3_prefix}/person_{shard}_{shard + CFG.crawl_shard_size - 1}/person_{pid:07d}.tar"
        try:
            obj = s3.get_object(Bucket=CFG.s3_bucket, Key=key)
            tensors, keys = [], []
            with tarfile.open(fileobj=io.BytesIO(obj["Body"].read()), mode="r") as tar:
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    f = tar.extractfile(m)
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    tensors.append(prep(img))
                    keys.append(f"{key}/{m.name}")
            for i in range(0, len(tensors), bs):
                batch = torch.stack(tensors[i:i + bs]).to(device)
                with torch.no_grad():
                    fe = net(batch).cpu().numpy().astype(np.float32)
                buf_emb.append(fe)
                buf_id.extend([pid] * len(fe))
                buf_key.extend(keys[i:i + bs])
            just_done.append(pid)
            total_imgs += len(keys)
            pbar.set_postfix(imgs=f"{total_imgs:,}", chunks=chunk)
            if len(buf_id) >= CFG.embed_flush_rows:
                flush()
        except Exception as e:
            log(f"[embed_s3] pid {pid} ERR: {e}")
    flush()
    log(f"[embed_s3] rank{rank} DONE: {len(pids):,} ids, {total_imgs:,} imgs -> {out_dir}")


if __name__ == "__main__":
    run()

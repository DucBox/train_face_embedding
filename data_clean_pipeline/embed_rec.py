"""
Stage 01a — EMBED rec sources (webface pure + public) with the NEW model.
Single responsibility: read .rec/.idx -> backbone -> raw 512-d embeddings parquet.

Multi-GPU: launch with torchrun; each rank embeds a contiguous shard and writes
its own parquet chunks. Resume = skip chunk files that already exist.

    torchrun --nproc_per_node=8 embed_rec.py --src webface

Output schema (one row / image): person_id, src, img_key="{prefix}:{rec_idx}", embedding[512].
(NOT normalized — that's stage 02.)  Only runs in real mode; in test mode the
fixtures in common.make_fixtures() stand in for this stage.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import COL_ID, COL_SRC, COL_KEY, COL_EMB, log
from config import CFG, SOURCES


def run(src: str):
    import numbers
    import mxnet as mx
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
    from torchvision import transforms
    from backbones import get_model

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = CFG.dir_embed(src)
    os.makedirs(out_dir, exist_ok=True)

    tfm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                              transforms.Normalize([0.5] * 3, [0.5] * 3)])

    class RecDS(Dataset):
        def __init__(self, prefix):
            self.prefix = prefix
            self.rec = mx.recordio.MXIndexedRecordIO(
                os.path.join(CFG.rec_root, f"{prefix}.idx"),
                os.path.join(CFG.rec_root, f"{prefix}.rec"), "r")
            h, _ = mx.recordio.unpack(self.rec.read_idx(0))
            self.idx = (np.arange(1, int(h.label[0])) if h.flag > 0
                        else np.array(list(self.rec.keys)))

        def __len__(self): return len(self.idx)

        def __getitem__(self, i):
            ridx = int(self.idx[i])
            h, img = mx.recordio.unpack(self.rec.read_idx(ridx))
            lbl = h.label if isinstance(h.label, numbers.Number) else h.label[0]
            return tfm(mx.image.imdecode(img).asnumpy()), int(lbl), \
                f"{self.prefix}:{ridx}"

    sets = [RecDS(p) for p in CFG.rec_prefix[src]]
    full = ConcatDataset(sets)
    n = len(full)
    per = (n + world - 1) // world
    lo, hi = rank * per, min(rank * per + per, n)
    shard = Subset(full, list(range(lo, hi)))
    loader = DataLoader(shard, batch_size=CFG.embed_batch_size, shuffle=False,
                        num_workers=CFG.embed_num_workers, pin_memory=True)
    log(f"[embed_rec:{src}] rank{rank}/{world}: total {n:,} imgs, this shard "
        f"[{lo:,},{hi:,}) = {hi - lo:,} imgs over {len(loader):,} batches")

    log(f"[embed_rec:{src}] loading model {CFG.network} <- {CFG.model_weight}")
    net = get_model(CFG.network, fp16=False, num_features=CFG.embedding_size).to(device).eval()
    net.load_state_dict(__import__("torch").load(CFG.model_weight, map_location="cpu"),
                        strict=False)

    buf, rows, chunk = [], 0, 0
    import torch as T

    def flush():
        nonlocal buf, rows, chunk
        if not buf:
            return
        fp = os.path.join(out_dir, f"part-rank{rank}-{chunk:04d}.parquet")
        if not os.path.exists(fp):
            pl.DataFrame(buf).write_parquet(fp)
        buf, rows = [], 0
        chunk += 1

    done = 0
    pbar = tqdm(loader, desc=f"embed_rec:{src} r{rank}", unit="batch", position=rank)
    with T.no_grad():
        for imgs, lbls, keys in pbar:
            feat = net(imgs.to(device)).cpu().numpy().astype(np.float32)
            for k, lb, fe in zip(keys, lbls.numpy(), feat):
                buf.append({COL_ID: int(lb), COL_SRC: src, COL_KEY: k, COL_EMB: fe.tolist()})
            rows += len(keys)
            done += len(keys)
            if rows >= CFG.embed_flush_rows:
                flush()
            pbar.set_postfix(imgs=f"{done:,}", chunks=chunk)
    flush()
    log(f"[embed_rec:{src}] rank{rank} DONE: {done:,} imgs, {chunk} chunks -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=(SOURCES[0], SOURCES[1]), required=True)
    run(ap.parse_args().src)

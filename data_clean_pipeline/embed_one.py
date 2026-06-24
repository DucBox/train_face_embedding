"""
DEBUG: embed + normalize the images of ONE person id and dump to CSV, so you can
eyeball the vectors against a ground truth and confirm the embed flow is correct.

    python embed_one.py --src crawl   --pid 11
    python embed_one.py --src webface --pid 0 --out /tmp/wf0.csv

Preprocessing: ALL sources use the cv2/S3 style (decode BGR -> resize112 ->
BGR2RGB -> /255 -0.5 /0.5), i.e. rec images are decoded with cv2 on their raw
jpeg bytes too (not mx.image.imdecode/torchvision). Logically identical to the
training transform (RGB, /255, (x-0.5)/0.5); only the JPEG decoder differs.

CSV columns: img_key, embedding, embedding_normalized  (512-d vectors, space-joined),
plus norm_raw. Uses CFG (model_weight/network/rec_root/S3). Real-only (torch +
mxnet/boto3/cv2). Note: for rec sources it SCANS the rec to collect the pid's
images (one-off; pass --max-scan to bound a huge rec).
"""
from __future__ import annotations

import argparse
import io
import numbers
import os
import sys
import tarfile

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG, SOURCES, WEBFACE, PUBLIC, CRAWL


def _prep(raw: bytes):
    """s3/cv2-style preprocessing on raw jpeg bytes -> CHW float tensor in [-1,1].
    Used for BOTH rec and crawl sources so the embed path is uniform."""
    import cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    img = cv2.cvtColor(cv2.resize(img, (112, 112)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float().div_(255).sub_(0.5).div_(0.5)


def _net():
    from backbones import get_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(CFG.network, fp16=False, num_features=CFG.embedding_size).to(dev).eval()
    net.load_state_dict(torch.load(CFG.model_weight, map_location="cpu"), strict=False)
    return net, dev


def _imgs_rec(src, pid, max_scan):
    """Scan src's rec(s), collect (img_key, CHW tensor) for label == pid.
    Decodes the raw jpeg bytes with cv2 (_prep), same as the crawl path."""
    import mxnet as mx
    out = []
    for prefix in CFG.rec_prefix[src]:
        rec = mx.recordio.MXIndexedRecordIO(
            os.path.join(CFG.rec_root, f"{prefix}.idx"),
            os.path.join(CFG.rec_root, f"{prefix}.rec"), "r")
        h0, _ = mx.recordio.unpack(rec.read_idx(0))
        keys = (range(1, int(h0.label[0])) if h0.flag > 0 else list(rec.keys))
        for n, k in enumerate(tqdm(keys, desc=f"scan {prefix}", unit="rec")):
            if max_scan and n >= max_scan:
                break
            h, img = mx.recordio.unpack(rec.read_idx(int(k)))
            lbl = int(h.label if isinstance(h.label, numbers.Number) else h.label[0])
            if lbl == pid:
                t = _prep(img)
                if t is not None:
                    out.append((f"{prefix}:{k}", t))
    return out


def _imgs_crawl(pid):
    """GET the person's tar from S3, return (member_name, CHW float tensor)."""
    import boto3
    s3 = boto3.client("s3", endpoint_url=CFG.s3_endpoint,
                      aws_access_key_id=CFG.s3_access_key,
                      aws_secret_access_key=CFG.s3_secret_key)
    start = (pid // CFG.crawl_shard_size) * CFG.crawl_shard_size
    folder = f"person_{start}_{start + CFG.crawl_shard_size - 1}"
    key = f"{CFG.crawl_s3_prefix}/{folder}/person_{pid:07d}.tar"
    print(f"[get] s3://{CFG.s3_bucket}/{key}")
    body = s3.get_object(Bucket=CFG.s3_bucket, Key=key)["Body"].read()
    out = []
    with tarfile.open(fileobj=io.BytesIO(body), mode="r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            t = _prep(tar.extractfile(m).read())
            if t is not None:
                out.append((f"{key}/{m.name}", t))
    return out


def _vec_str(v, prec):
    return " ".join(f"{x:.{prec}f}" for x in v)


def run(src, pid, out_path, max_scan, prec):
    net, dev = _net()
    items = _imgs_crawl(pid) if src == CRAWL else _imgs_rec(src, pid, max_scan)
    if not items:
        print(f"[!] no images found for {src} pid {pid}")
        return
    print(f"[embed] {len(items)} images for {src} pid {pid}")

    rows = []
    with torch.no_grad():
        for key, t in tqdm(items, desc="embed", unit="img"):
            raw = net(t.unsqueeze(0).to(dev))            # (1, 512) raw
            norm = F.normalize(raw, dim=1)
            raw = raw.cpu().numpy()[0]
            norm = norm.cpu().numpy()[0]
            rows.append({
                "img_key": key,
                "norm_raw": float(np.linalg.norm(raw)),
                "embedding": _vec_str(raw, prec),
                "embedding_normalized": _vec_str(norm, prec),
            })
    pl.DataFrame(rows).write_csv(out_path)
    print(f"[done] {len(rows)} rows -> {out_path}")
    print(f"[done] sample norm_raw: {[round(r['norm_raw'], 4) for r in rows[:5]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=SOURCES, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-scan", type=int, default=0,
                    help="rec sources only: cap how many records to scan (0 = all)")
    ap.add_argument("--precision", type=int, default=6, help="float decimals in CSV")
    a = ap.parse_args()
    out = a.out or f"/tmp/embed_{a.src}_{a.pid}.csv"
    run(a.src, a.pid, out, a.max_scan, a.precision)

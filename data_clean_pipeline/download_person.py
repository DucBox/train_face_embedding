"""
Inspect the S3 layout (READ-ONLY). Two uses:

1) Discover folder structure (confirms shard_size & that tars exist):
       python download_person.py --list --prefix cv/processed-datasets/aligned_face_112_112
   Prints the first keys under the prefix — look at the folder names:
       person_0_999    -> shard_size 1000
       person_0_9999   -> shard_size 10000

2) Download ONE person's tar and verify the crops are 112x112 aligned faces:
       python download_person.py --pid 1 \
           --prefix cv/processed-datasets/aligned_face_112_112 --shard-size 1000 \
           --out /tmp/person_1
   Extracts every image, prints count + per-image (H, W, C), and a summary of
   whether they are all 112x112.

Credentials/bucket/endpoint come from config.py (env-overridable). Needs boto3
+ cv2; run on the cluster.
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile

from config import CFG


def _client():
    import boto3
    return boto3.client("s3", endpoint_url=CFG.s3_endpoint,
                        aws_access_key_id=CFG.s3_access_key,
                        aws_secret_access_key=CFG.s3_secret_key)


def _key_for(pid, prefix, shard_size):
    start = (pid // shard_size) * shard_size
    folder = f"person_{start}_{start + shard_size - 1}"
    return f"{prefix}/{folder}/person_{pid:07d}.tar"


def do_list(prefix, n):
    s3 = _client()
    print(f"[list] s3://{CFG.s3_bucket}/{prefix}  (first {n} keys)\n")
    paginator = s3.get_paginator("list_objects_v2")
    shown = 0
    folders = set()
    for page in paginator.paginate(Bucket=CFG.s3_bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            print(f"  {obj['Size']:>12,}  {key}")
            parts = key[len(prefix):].lstrip("/").split("/")
            if parts:
                folders.add(parts[0])
            shown += 1
            if shown >= n:
                break
        if shown >= n:
            break
    if not shown:
        print("  (nothing — wrong prefix or empty)")
        return
    print(f"\n[hint] folder names seen: {sorted(folders)[:5]}")
    print("[hint] 'person_0_999' => shard_size 1000 ; 'person_0_9999' => shard_size 10000")


def do_download(pid, prefix, shard_size, out):
    import cv2
    import numpy as np
    s3 = _client()
    key = _key_for(pid, prefix, shard_size)
    print(f"[get] s3://{CFG.s3_bucket}/{key}")
    try:
        body = s3.get_object(Bucket=CFG.s3_bucket, Key=key)["Body"].read()
    except Exception as e:
        print(f"[ERR] {e}\n[hint] check --prefix and --shard-size (folder naming) and that pid exists")
        return
    os.makedirs(out, exist_ok=True)
    dims, n = {}, 0
    with tarfile.open(fileobj=io.BytesIO(body), mode="r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            raw = tar.extractfile(m).read()
            open(os.path.join(out, os.path.basename(m.name)), "wb").write(raw)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            shape = None if img is None else img.shape
            dims[shape] = dims.get(shape, 0) + 1
            if n < 10:
                print(f"   {m.name}  shape={shape}")
            n += 1
    print(f"\n[summary] {n} images extracted to {out}")
    print(f"[summary] distinct shapes: {dims}")
    all_112 = set(dims) == {(112, 112, 3)}
    print(f"[summary] all 112x112x3 aligned crops? {'YES ✓' if all_112 else 'NO ✗ -> ' + str(list(dims))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list keys under --prefix")
    ap.add_argument("--pid", type=int, help="person id to download")
    ap.add_argument("--prefix", default=CFG.crawl_s3_prefix)
    ap.add_argument("--shard-size", type=int, default=CFG.crawl_shard_size)
    ap.add_argument("--out", default=None)
    ap.add_argument("-n", type=int, default=20, help="how many keys to show in --list")
    a = ap.parse_args()
    if a.list:
        do_list(a.prefix, a.n)
    elif a.pid is not None:
        do_download(a.pid, a.prefix, a.shard_size, a.out or f"/tmp/person_{a.pid}")
    else:
        ap.error("pass --list or --pid")


if __name__ == "__main__":
    main()

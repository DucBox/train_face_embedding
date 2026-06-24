"""
Standalone READ-ONLY S3 inspector (test script). No config/env — just edit the
CONFIG block below and run:

    python download_person.py

MODE = "list"      -> list keys under PREFIX (confirm folder layout / shard_size)
MODE = "download"  -> download person PID's tar and verify crops are 112x112

Needs boto3 + cv2. Run on the cluster.
"""
import io
import os
import tarfile

import cv2
import numpy as np
import boto3

# ============================ CONFIG (edit me) ============================ #
S3_ENDPOINT = "http://s3-data.cyberspace.vn"
S3_BUCKET   = "ttnt"
S3_ACCESS_KEY = "ttnt"
S3_SECRET_KEY = "<PASTE_S3_SECRET_KEY_HERE>"     # paste locally; don't commit it

# which dataset folder on S3 to inspect
PREFIX     = "cv/processed-datasets/aligned_face_112_112"
SHARD_SIZE = 1000          # folder = person_{start}_{start+SHARD_SIZE-1}

MODE = "list"              # "list" or "download"
PID  = 1                   # person id to download (MODE="download")
OUT  = "/tmp/person_1"     # where to extract images
N_LIST = 20                # how many keys to print in list mode
# ========================================================================= #


def client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY)


def key_for(pid):
    start = (pid // SHARD_SIZE) * SHARD_SIZE
    folder = f"person_{start}_{start + SHARD_SIZE - 1}"
    return f"{PREFIX}/{folder}/person_{pid:07d}.tar"


def do_list():
    s3 = client()
    print(f"[list] s3://{S3_BUCKET}/{PREFIX}  (first {N_LIST} keys)\n")
    shown, folders = 0, set()
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=S3_BUCKET, Prefix=PREFIX.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            print(f"  {obj['Size']:>12,}  {obj['Key']}")
            folders.add(obj["Key"][len(PREFIX):].lstrip("/").split("/")[0])
            shown += 1
            if shown >= N_LIST:
                break
        if shown >= N_LIST:
            break
    if not shown:
        print("  (nothing — wrong PREFIX or empty)")
        return
    print(f"\n[hint] folder names: {sorted(folders)[:5]}")
    print("[hint] 'person_0_999' => SHARD_SIZE 1000 ; 'person_0_9999' => 10000")


def do_download():
    s3 = client()
    key = key_for(PID)
    print(f"[get] s3://{S3_BUCKET}/{key}")
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    except Exception as e:
        print(f"[ERR] {e}\n[hint] check PREFIX / SHARD_SIZE / PID")
        return
    os.makedirs(OUT, exist_ok=True)
    dims, n = {}, 0
    with tarfile.open(fileobj=io.BytesIO(body), mode="r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            raw = tar.extractfile(m).read()
            open(os.path.join(OUT, os.path.basename(m.name)), "wb").write(raw)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            shape = None if img is None else img.shape
            dims[shape] = dims.get(shape, 0) + 1
            if n < 10:
                print(f"   {m.name}  shape={shape}")
            n += 1
    print(f"\n[summary] {n} images -> {OUT}")
    print(f"[summary] distinct shapes: {dims}")
    print(f"[summary] all 112x112x3? "
          f"{'YES' if set(dims) == {(112, 112, 3)} else 'NO -> ' + str(list(dims))}")


if __name__ == "__main__":
    do_list() if MODE == "list" else do_download()

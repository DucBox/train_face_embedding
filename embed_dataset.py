"""
Stage 1/3 of the offline hard-case mining pipeline: embed every image in the
training set with a checkpoint and write the result to a Parquet dataset for
reuse (by find_hard_thresholds.py, or any other tool - pandas/DuckDB/Spark can
read it directly).

Supports torchrun for multi-GPU: each rank embeds its own contiguous shard of
the dataset independently and writes its own Parquet files. No NCCL/process-group
is needed since ranks never communicate with each other here.

    torchrun --nproc_per_node=8 embed_dataset.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
        --weight /path/to/model.pt --output-dir /path/to/hard_case_out

Parquet schema written (one row per image), partitioned by file_prefix so a
later reader can filter by source file without a full scan:
    identity     int32   - global class ID (same ID across every .rec source file)
    file_prefix  string  - source file (train_synthetic, train_public, train_1, ...)
    rec_idx      int64   - index inside that file's .rec/.idx, to recover the raw image
    embedding    fixed_size_list<float32>[embedding_size] - L2-normalized embedding

`file_prefix` + `rec_idx` let you pull the exact image back out via
`mx.recordio.MXIndexedRecordIO(...).read_idx(rec_idx)` on the matching .rec/.idx file.
"""
import argparse
import glob
import numbers
import os

import mxnet as mx
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from backbones import get_model
from utils.utils_config import get_config


class MXEvalDataset(Dataset):
    """Deterministic (no augmentation) reader for one .rec/.idx pair.

    Mirrors the header-detection in dataset.MXFaceDataset so files without
    header metadata (train_1.rec, train_2.rec, ...) are read the same way as
    train_synthetic.rec / train_public.rec - both fall back to `imgrec.keys`
    when record 0 isn't a header (flag <= 0).
    """

    def __init__(self, root_dir, file_prefix):
        path_imgrec = os.path.join(root_dir, f"{file_prefix}.rec")
        path_imgidx = os.path.join(root_dir, f"{file_prefix}.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, i):
        idx = int(self.imgidx[i])
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        sample = mx.image.imdecode(img).asnumpy()
        sample = self.transform(sample)
        return sample, int(label), idx


def discover_eval_datasets(cfg, rec_dir=None):
    """Discover .rec/.idx pairs to embed, supporting two naming conventions:

    NEW (rec_out from data_clean_pipeline/write_rec.py):
        train_synthetic_clean.rec  train_public_clean.rec  train_crawl_NNN.rec
        Detected automatically when train_synthetic_clean.rec is present.
        Override the directory with --rec-dir (or rec_dir arg); defaults to cfg.rec.

    CLASSIC (original training data layout):
        train_synthetic.rec / train.rec  train_public.rec  train_N.rec
        Used when the new naming is not detected.
    """
    root_dir = rec_dir or cfg.rec
    datasets, prefixes = [], []

    # --- new rec_out naming ---
    if os.path.exists(os.path.join(root_dir, "train_synthetic_clean.rec")):
        for prefix in ["train_synthetic_clean", "train_public_clean"]:
            if os.path.exists(os.path.join(root_dir, f"{prefix}.rec")):
                datasets.append(MXEvalDataset(root_dir, prefix))
                prefixes.append(prefix)
        for p in sorted(glob.glob(os.path.join(root_dir, "train_crawl_*.rec"))):
            prefix = os.path.basename(p)[:-4]
            datasets.append(MXEvalDataset(root_dir, prefix))
            prefixes.append(prefix)
        return datasets, prefixes

    # --- classic naming ---
    main_prefix = "train_synthetic" if cfg.use_synthetic_data else "train"
    if os.path.exists(os.path.join(root_dir, f"{main_prefix}.rec")):
        datasets.append(MXEvalDataset(root_dir, main_prefix))
        prefixes.append(main_prefix)
    if cfg.use_public_data and os.path.exists(os.path.join(root_dir, "train_public.rec")):
        datasets.append(MXEvalDataset(root_dir, "train_public"))
        prefixes.append("train_public")
    for i in range(1, cfg.num_rec_files):
        prefix = f"train_{i}"
        if os.path.exists(os.path.join(root_dir, f"{prefix}.rec")):
            datasets.append(MXEvalDataset(root_dir, prefix))
            prefixes.append(prefix)

    return datasets, prefixes


def build_backbone(cfg):
    if cfg.network == "vit_l_dinov3":
        return get_model(
            cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size,
            pretrained_path=None, freeze_backbone=False, use_projection=cfg.use_projection,
        )
    return get_model(cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size)


def write_chunk(buf_embeddings, buf_identity, buf_rec_idx, buf_file_prefix,
                 embedding_size, output_dir, rank, chunk_idx):
    """Write one accumulated chunk as its own Parquet part, then the caller can
    drop the buffer - this caps peak RAM to ~flush_rows worth of data instead of
    the whole per-rank shard (which can be tens of GB at full dataset scale)."""
    embeddings = np.concatenate(buf_embeddings, axis=0)
    identity = np.concatenate(buf_identity, axis=0)
    rec_idx = np.concatenate(buf_rec_idx, axis=0)
    file_prefix = np.concatenate(buf_file_prefix, axis=0)

    table = pa.table({
        "identity": pa.array(identity, type=pa.int32()),
        "file_prefix": pa.array(file_prefix, type=pa.string()),
        "rec_idx": pa.array(rec_idx, type=pa.int64()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()), embedding_size),
    })
    pads.write_dataset(
        table,
        base_dir=output_dir,
        format="parquet",
        partitioning=pads.partitioning(pa.schema([("file_prefix", pa.string())]), flavor="hive"),
        basename_template=f"part-rank{rank}-chunk{chunk_idx}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )
    return len(identity)


# Default checkpoint used when --weight isn't passed - edit to match where your
# run actually saved model.pt / model_epoch_N.pt (see config.output in train_v2.py).
DEFAULT_WEIGHT = "/path/to/model.pt"


def main():
    parser = argparse.ArgumentParser(description="Embed the training set to a Parquet dataset")
    parser.add_argument("config", type=str, help="e.g. configs/wf42m_pfc03_40epoch_64gpu_vit_l")
    parser.add_argument("--weight", type=str, default=DEFAULT_WEIGHT,
                         help=f"backbone state_dict .pt (e.g. model.pt), default: {DEFAULT_WEIGHT}")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flush-rows", type=int, default=500_000,
                         help="write a Parquet chunk every this many embedded images, "
                              "instead of holding the whole per-rank shard in RAM at once")
    parser.add_argument("--rec-dir", type=str, default=None,
                         help="override cfg.rec: directory containing the .rec/.idx files to embed. "
                              "Use this to point at rec_out/ from data_clean_pipeline/write_rec.py "
                              "(auto-detected by presence of train_synthetic_clean.rec).")
    args = parser.parse_args()

    cfg = get_config(args.config)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    datasets, prefixes = discover_eval_datasets(cfg, rec_dir=args.rec_dir)
    assert datasets, f"No .rec files found under {cfg.rec}"
    full_set = ConcatDataset(datasets)
    # file_prefix per image, resolved straight to the source-file string (no separate
    # id->name mapping file needed - Parquet dictionary-encodes repeated strings anyway)
    file_prefix_per_image = np.concatenate([
        np.full(len(d), prefixes[pi], dtype=object) for pi, d in enumerate(datasets)
    ])

    n_total = len(full_set)
    per_rank = (n_total + world_size - 1) // world_size
    start, end = rank * per_rank, min(rank * per_rank + per_rank, n_total)
    if rank == 0:
        print(f"Total images: {n_total:,} | world_size={world_size} | per-rank shard size ~{per_rank:,}")

    shard_set = Subset(full_set, list(range(start, end)))
    loader = DataLoader(shard_set, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    net = build_backbone(cfg)
    net.load_state_dict(torch.load(args.weight, map_location="cpu"))
    net.eval().to(device)

    n_shard = end - start
    file_prefix_shard = file_prefix_per_image[start:end]
    os.makedirs(args.output_dir, exist_ok=True)

    buf_embeddings, buf_identity, buf_rec_idx, buf_file_prefix = [], [], [], []
    buf_rows = 0
    chunk_idx = 0
    total_written = 0
    cursor = 0

    progress = tqdm(loader, total=len(loader), desc=f"rank{rank} embed", position=rank, unit="batch")
    with torch.no_grad():
        for imgs, lbls, idxs in progress:
            imgs = imgs.to(device, non_blocking=True)
            feat = F.normalize(net(imgs), dim=1)
            n = feat.size(0)

            buf_embeddings.append(feat.cpu().numpy().astype(np.float32))
            buf_identity.append(lbls.numpy().astype(np.int32))
            buf_rec_idx.append(idxs.numpy().astype(np.int64))
            buf_file_prefix.append(file_prefix_shard[cursor:cursor + n])
            buf_rows += n
            cursor += n

            if buf_rows >= args.flush_rows:
                total_written += write_chunk(buf_embeddings, buf_identity, buf_rec_idx, buf_file_prefix,
                                              cfg.embedding_size, args.output_dir, rank, chunk_idx)
                chunk_idx += 1
                buf_embeddings, buf_identity, buf_rec_idx, buf_file_prefix = [], [], [], []
                buf_rows = 0
                progress.set_postfix(chunks=chunk_idx)

    if buf_rows > 0:
        total_written += write_chunk(buf_embeddings, buf_identity, buf_rec_idx, buf_file_prefix,
                                      cfg.embedding_size, args.output_dir, rank, chunk_idx)
        chunk_idx += 1

    print(f"[rank {rank}] wrote {total_written:,} rows to {args.output_dir} "
          f"(partitioned by file_prefix, {chunk_idx} chunk(s) of <= {args.flush_rows:,} rows each)")


if __name__ == "__main__":
    main()

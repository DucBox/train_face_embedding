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


def discover_eval_datasets(cfg):
    """Same file discovery as dataset.get_dataloader(), eval-mode (no augmentation).

    Uses cfg.use_synthetic_data / cfg.use_public_data directly by name - note
    train_v2.py's call into dataset.get_dataloader() passes these two positionally
    and lands them on the wrong-named (but functionally harmless, since both are
    True in the active config) kwargs; not relevant here since we match by name.
    """
    root_dir = cfg.rec
    datasets, prefixes = [], []

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


def main():
    parser = argparse.ArgumentParser(description="Embed the training set to a Parquet dataset")
    parser.add_argument("config", type=str, help="e.g. configs/wf42m_pfc03_40epoch_64gpu_vit_l")
    parser.add_argument("--weight", type=str, required=True, help="backbone state_dict .pt (e.g. model.pt)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = get_config(args.config)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    datasets, prefixes = discover_eval_datasets(cfg)
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
    embeddings = np.zeros((n_shard, cfg.embedding_size), dtype=np.float32)
    identity = np.zeros((n_shard,), dtype=np.int32)
    rec_idx = np.zeros((n_shard,), dtype=np.int64)
    file_prefix_shard = file_prefix_per_image[start:end]

    cursor = 0
    with torch.no_grad():
        for imgs, lbls, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            feat = F.normalize(net(imgs), dim=1)
            n = feat.size(0)
            embeddings[cursor:cursor + n] = feat.cpu().numpy().astype(np.float32)
            identity[cursor:cursor + n] = lbls.numpy()
            rec_idx[cursor:cursor + n] = idxs.numpy()
            cursor += n
            if rank == 0 and (cursor // args.batch_size) % 50 == 0:
                print(f"[rank 0] {cursor:,}/{n_shard:,}")

    table = pa.table({
        "identity": pa.array(identity, type=pa.int32()),
        "file_prefix": pa.array(file_prefix_shard, type=pa.string()),
        "rec_idx": pa.array(rec_idx, type=pa.int64()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()), cfg.embedding_size),
    })

    os.makedirs(args.output_dir, exist_ok=True)
    pads.write_dataset(
        table,
        base_dir=args.output_dir,
        format="parquet",
        partitioning=pads.partitioning(pa.schema([("file_prefix", pa.string())]), flavor="hive"),
        basename_template=f"part-rank{rank}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )
    print(f"[rank {rank}] wrote {n_shard:,} rows to {args.output_dir} (partitioned by file_prefix)")


if __name__ == "__main__":
    main()

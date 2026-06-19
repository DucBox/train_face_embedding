"""
Offline hard-case mining: use a trained checkpoint to find, at a target FMR
(e.g. 1e-6), the two failure groups on the training set itself:

  - genuine pairs (same identity) scoring BELOW the threshold -> false reject
  - impostor pairs (different identity) scoring ABOVE the threshold -> false accept

Comparing every image against every other image is infeasible at this scale
(71M images), so identities are represented by their centroid (mean of L2-normalized
embeddings) - this is the standard template-based approximation, not an exact
image-vs-image NIST-style score. Treat the output as a prioritized candidate list
for inspection / hard-negative seeding, not as the exact NIST operating point.

Usage (two stages, run separately):

    # 1) Embed every image in the dataset (supports torchrun for multi-GPU,
    #    each rank embeds its own shard independently, no NCCL/process-group
    #    needed since ranks never communicate with each other here).
    torchrun --nproc_per_node=8 infer_hard_cases.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
        --stage embed --weight /path/to/model.pt --output-dir /path/to/hard_case_out

    # 2) Merge shards, build centroids, search for hard pairs, write CSVs
    #    (single process; the centroid search itself uses one GPU).
    python3 infer_hard_cases.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
        --stage analyze --output-dir /path/to/hard_case_out --fmr 1e-6

Outputs in --output-dir:
    false_accept.csv  : class_a,class_b,score        (impostor identity pairs above threshold)
    false_reject.csv  : file_prefix,rec_idx,label,genuine_score  (genuine images below threshold)

`file_prefix` + `rec_idx` let you pull the exact image back out via
`mx.recordio.MXIndexedRecordIO(...).read_idx(rec_idx)` on the matching .rec/.idx file.
"""
import argparse
import glob
import json
import numbers
import os

import mxnet as mx
import numpy as np
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


def stage_embed(args, cfg):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    datasets, prefixes = discover_eval_datasets(cfg)
    assert datasets, f"No .rec files found under {cfg.rec}"
    full_set = ConcatDataset(datasets)
    prefix_ids = np.concatenate([
        np.full(len(d), pi, dtype=np.int16) for pi, d in enumerate(datasets)
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
    embeddings = np.zeros((n_shard, cfg.embedding_size), dtype=np.float16)
    labels = np.zeros((n_shard,), dtype=np.int32)
    rec_idx = np.zeros((n_shard,), dtype=np.int64)
    prefix_ids_shard = prefix_ids[start:end]

    cursor = 0
    with torch.no_grad():
        for imgs, lbls, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            feat = F.normalize(net(imgs), dim=1)
            n = feat.size(0)
            embeddings[cursor:cursor + n] = feat.cpu().numpy().astype(np.float16)
            labels[cursor:cursor + n] = lbls.numpy()
            rec_idx[cursor:cursor + n] = idxs.numpy()
            cursor += n
            if rank == 0 and (cursor // args.batch_size) % 50 == 0:
                print(f"[rank 0] {cursor:,}/{n_shard:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "prefixes.json"), "w") as f:
            json.dump(prefixes, f)

    out_path = os.path.join(args.output_dir, f"shard_rank{rank}.npz")
    np.savez(out_path, embeddings=embeddings, labels=labels,
              rec_idx=rec_idx, prefix_ids=prefix_ids_shard)
    print(f"[rank {rank}] wrote {out_path} ({n_shard:,} images)")


def stage_analyze(args, cfg):
    shard_files = sorted(glob.glob(os.path.join(args.output_dir, "shard_rank*.npz")))
    assert shard_files, f"No shard_rank*.npz in {args.output_dir} - run --stage embed first"
    with open(os.path.join(args.output_dir, "prefixes.json")) as f:
        prefixes = json.load(f)

    device = torch.device(args.device)
    embed_dim, num_classes = cfg.embedding_size, cfg.num_classes

    print("Pass 1/3: accumulating per-class centroids ...")
    centroid_sum = torch.zeros((num_classes, embed_dim), dtype=torch.float64)
    centroid_count = torch.zeros((num_classes,), dtype=torch.int64)
    for sf in shard_files:
        d = np.load(sf)
        emb = torch.from_numpy(d["embeddings"].astype(np.float32)).double()
        lbl = torch.from_numpy(d["labels"].astype(np.int64))
        centroid_sum.index_add_(0, lbl, emb)
        centroid_count.index_add_(0, lbl, torch.ones_like(lbl))
    has_images = centroid_count > 0
    n_with_images = int(has_images.sum())
    print(f"  {n_with_images:,}/{num_classes:,} classes have >=1 image in this dataset view")

    centroids = torch.zeros((num_classes, embed_dim), dtype=torch.float32)
    centroids[has_images] = (centroid_sum[has_images]
                              / centroid_count[has_images].unsqueeze(1).double()).float()
    centroids = F.normalize(centroids, dim=1)

    print("Pass 2/3: searching nearest other-identity centroids (impostor candidates) ...")
    centroids_dev = centroids.to(device)
    has_images_dev = has_images.to(device)
    valid_idx = torch.nonzero(has_images, as_tuple=True)[0]
    chunk, topk = args.centroid_chunk, min(args.topk, n_with_images - 1)

    cand_a, cand_b, cand_score = [], [], []
    for start in range(0, valid_idx.numel(), chunk):
        rows = valid_idx[start:start + chunk].to(device)
        sims = centroids_dev[rows] @ centroids_dev.t()
        sims[torch.arange(rows.numel(), device=device), rows] = -2.0
        sims[:, ~has_images_dev] = -2.0
        vals, cols = sims.topk(topk, dim=1)
        cand_a.append(rows.repeat_interleave(vals.size(1)).cpu())
        cand_b.append(cols.flatten().cpu())
        cand_score.append(vals.flatten().cpu())
        if start % (chunk * 20) == 0:
            print(f"  {start:,}/{valid_idx.numel():,} classes searched")
    cand_a, cand_b, cand_score = torch.cat(cand_a), torch.cat(cand_b), torch.cat(cand_score)

    # dedup symmetric (a,b)/(b,a) duplicates
    pair_key = (torch.minimum(cand_a, cand_b) * num_classes + torch.maximum(cand_a, cand_b)).numpy()
    _, uniq_pos = np.unique(pair_key, return_index=True)
    cand_a, cand_b, cand_score = cand_a[uniq_pos], cand_b[uniq_pos], cand_score[uniq_pos]
    order = torch.argsort(cand_score, descending=True)
    cand_a, cand_b, cand_score = cand_a[order], cand_b[order], cand_score[order]

    total_impostor_pairs = n_with_images * (n_with_images - 1) // 2
    rank_at_fmr = max(1, int(round(total_impostor_pairs * args.fmr)))
    if rank_at_fmr > cand_score.numel():
        print(f"WARNING: need top-{rank_at_fmr:,} impostor pairs for FMR={args.fmr:g}, but only "
              f"{cand_score.numel():,} candidates were collected with --topk={args.topk}. "
              f"Threshold below is a LOWER BOUND - increase --topk and re-run for an exact value.")
        rank_at_fmr = cand_score.numel()
    threshold = cand_score[rank_at_fmr - 1].item()
    print(f"Target FMR={args.fmr:g} over ~{total_impostor_pairs:,} identity-pair comparisons "
          f"-> cosine threshold = {threshold:.4f}")

    false_accept = list(zip(cand_a[:rank_at_fmr].tolist(),
                             cand_b[:rank_at_fmr].tolist(),
                             cand_score[:rank_at_fmr].tolist()))

    print("Pass 3/3: scoring genuine (same-identity, leave-one-out) images against threshold ...")
    false_reject_rows = []
    for sf in shard_files:
        d = np.load(sf)
        emb = torch.from_numpy(d["embeddings"].astype(np.float32))
        lbl = torch.from_numpy(d["labels"].astype(np.int64))
        rec_idx, prefix_ids = d["rec_idx"], d["prefix_ids"]

        cnt = centroid_count[lbl]
        valid = cnt > 1
        loo_sum = centroid_sum[lbl][valid] - emb[valid].double()
        loo_centroid = F.normalize((loo_sum / (cnt[valid] - 1).unsqueeze(1).double()).float(), dim=1)
        genuine_score = (emb[valid] * loo_centroid).sum(dim=1)

        flagged_local = (genuine_score < threshold).nonzero(as_tuple=True)[0]
        valid_global_idx = valid.nonzero(as_tuple=True)[0]
        for li in flagged_local.tolist():
            gi = valid_global_idx[li].item()
            false_reject_rows.append((
                prefixes[prefix_ids[gi]], int(rec_idx[gi]), int(lbl[gi]), float(genuine_score[li]),
            ))

    fa_path = os.path.join(args.output_dir, "false_accept.csv")
    fr_path = os.path.join(args.output_dir, "false_reject.csv")
    with open(fa_path, "w") as f:
        f.write("class_a,class_b,score\n")
        for a, b, s in false_accept:
            f.write(f"{a},{b},{s:.4f}\n")
    with open(fr_path, "w") as f:
        f.write("file_prefix,rec_idx,label,genuine_score\n")
        for prefix, idx, lbl, score in false_reject_rows:
            f.write(f"{prefix},{idx},{lbl},{score:.4f}\n")

    print(f"False accept (impostor identity pairs above threshold): {len(false_accept):,} -> {fa_path}")
    print(f"False reject (genuine images below threshold): {len(false_reject_rows):,} -> {fr_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline hard-case mining at a target FMR")
    parser.add_argument("config", type=str, help="e.g. configs/wf42m_pfc03_40epoch_64gpu_vit_l")
    parser.add_argument("--stage", choices=["embed", "analyze"], required=True)
    parser.add_argument("--weight", type=str, default=None,
                         help="backbone state_dict .pt (e.g. model.pt) - required for --stage embed")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fmr", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=50,
                         help="nearest-centroid candidates kept per class for the impostor search")
    parser.add_argument("--centroid-chunk", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = get_config(args.config)

    if args.stage == "embed":
        assert args.weight, "--weight is required for --stage embed"
        stage_embed(args, cfg)
    else:
        stage_analyze(args, cfg)


if __name__ == "__main__":
    main()

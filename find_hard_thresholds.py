"""
Stage 2/3 of the offline hard-case mining pipeline: read the Parquet embeddings
written by embed_dataset.py, build per-identity centroids, derive the score
threshold for a target FMR, and write the two failure groups:

  - genuine pairs (same identity) scoring BELOW the threshold -> false reject
  - impostor pairs (different identity) scoring ABOVE the threshold -> false accept

Comparing every image against every other image is infeasible at this scale
(tens of millions of images), so identities are represented by their centroid
(mean of L2-normalized embeddings) - this is the standard template-based
approximation, not an exact image-vs-image NIST-style score. Treat the output
as a prioritized candidate list for inspection / hard-negative seeding
(see generate_hard_cases.py), not as the exact NIST operating point.

    python3 find_hard_thresholds.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
        --embeddings-dir /path/to/hard_case_out --output-dir /path/to/hard_case_out --fmr 1e-6

Outputs in --output-dir (both include the `threshold` used, alongside each
pair/image's actual score, so downstream steps don't need to recompute it):
    false_accept.csv  : class_a,class_b,score,threshold
    false_reject.csv  : file_prefix,rec_idx,label,genuine_score,threshold
"""
import argparse
import os

import numpy as np
import pyarrow.dataset as pads
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_config import get_config


def _open_embedding_dataset(embeddings_dir):
    dataset = pads.dataset(embeddings_dir, format="parquet", partitioning="hive")
    assert dataset.count_rows() > 0, f"No parquet rows found in {embeddings_dir} - run embed_dataset.py first"
    return dataset


def _iter_batches(dataset, embed_dim, batch_size):
    """Yield (embeddings[N,embed_dim] f32, identity[N] i64, rec_idx[N] i64, file_prefix[N] list[str])."""
    columns = ["identity", "file_prefix", "rec_idx", "embedding"]
    for batch in dataset.to_batches(batch_size=batch_size, columns=columns):
        identity = batch.column("identity").to_numpy().astype(np.int64)
        rec_idx = batch.column("rec_idx").to_numpy()
        file_prefix = batch.column("file_prefix").to_pylist()
        emb_flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
        embeddings = emb_flat.reshape(-1, embed_dim).astype(np.float32)
        yield embeddings, identity, rec_idx, file_prefix


def main():
    parser = argparse.ArgumentParser(description="Find hard genuine/impostor cases at a target FMR")
    parser.add_argument("config", type=str, help="e.g. configs/wf42m_pfc03_40epoch_64gpu_vit_l")
    parser.add_argument("--embeddings-dir", type=str, required=True,
                         help="output-dir of embed_dataset.py (Parquet dataset)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fmr", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=50,
                         help="nearest-centroid candidates kept per class for the impostor search")
    parser.add_argument("--centroid-chunk", type=int, default=1024)
    parser.add_argument("--read-batch-size", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = get_config(args.config)
    dataset = _open_embedding_dataset(args.embeddings_dir)
    device = torch.device(args.device)
    embed_dim, num_classes = cfg.embedding_size, cfg.num_classes
    n_rows = dataset.count_rows()
    n_read_batches = -(-n_rows // args.read_batch_size)  # ceil div

    print("Pass 1/3: accumulating per-class centroids ...")
    centroid_sum = torch.zeros((num_classes, embed_dim), dtype=torch.float64)
    centroid_count = torch.zeros((num_classes,), dtype=torch.int64)
    for embeddings, identity, _, _ in tqdm(_iter_batches(dataset, embed_dim, args.read_batch_size),
                                            total=n_read_batches, desc="pass1 centroids", unit="batch"):
        emb = torch.from_numpy(embeddings).double()
        lbl = torch.from_numpy(identity)
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
    chunk_starts = list(range(0, valid_idx.numel(), chunk))
    for start in tqdm(chunk_starts, desc="pass2 impostor search", unit="chunk"):
        rows = valid_idx[start:start + chunk].to(device)
        sims = centroids_dev[rows] @ centroids_dev.t()
        sims[torch.arange(rows.numel(), device=device), rows] = -2.0
        sims[:, ~has_images_dev] = -2.0
        vals, cols = sims.topk(topk, dim=1)
        cand_a.append(rows.repeat_interleave(vals.size(1)).cpu())
        cand_b.append(cols.flatten().cpu())
        cand_score.append(vals.flatten().cpu())
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
    for embeddings, identity, rec_idx, file_prefix in tqdm(_iter_batches(dataset, embed_dim, args.read_batch_size),
                                                             total=n_read_batches, desc="pass3 genuine scoring",
                                                             unit="batch"):
        emb = torch.from_numpy(embeddings)
        lbl = torch.from_numpy(identity)

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
                file_prefix[gi], int(rec_idx[gi]), int(lbl[gi]), float(genuine_score[li]),
            ))

    os.makedirs(args.output_dir, exist_ok=True)
    fa_path = os.path.join(args.output_dir, "false_accept.csv")
    fr_path = os.path.join(args.output_dir, "false_reject.csv")
    with open(fa_path, "w") as f:
        f.write("class_a,class_b,score,threshold\n")
        for a, b, s in false_accept:
            f.write(f"{a},{b},{s:.4f},{threshold:.4f}\n")
    with open(fr_path, "w") as f:
        f.write("file_prefix,rec_idx,label,genuine_score,threshold\n")
        for prefix, idx, lbl, score in false_reject_rows:
            f.write(f"{prefix},{idx},{lbl},{score:.4f},{threshold:.4f}\n")

    print(f"Threshold (cosine) for FMR={args.fmr:g}: {threshold:.4f}")
    print(f"False accept (impostor identity pairs above threshold): {len(false_accept):,} -> {fa_path}")
    print(f"False reject (genuine images below threshold): {len(false_reject_rows):,} -> {fr_path}")


if __name__ == "__main__":
    main()

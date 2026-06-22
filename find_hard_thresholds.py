"""
Stage 2/3 of the offline hard-case mining pipeline: read the Parquet embeddings
written by embed_dataset.py, derive the score threshold for a target FMR, and
write the two failure groups:

  - genuine pairs (same identity) scoring BELOW the threshold -> false reject
  - impostor pairs (different identity) scoring ABOVE the threshold -> false accept

Comparing every image against every other image is infeasible at this scale
(tens of millions of images), so each identity is represented by a single real
image: its "medoid" - the image whose embedding is closest to the identity's
centroid (mean of L2-normalized embeddings). The centroid itself is only ever
used internally to pick that representative image; every score that ends up in
the output is a real-image-vs-real-image cosine similarity, not an
average-vs-average or image-vs-average score. This keeps both sides of the
threshold (impostor and genuine) on the same footing - using the centroid
itself as the impostor score (as an earlier version of this script did) instead
systematically inflates impostor scores relative to genuine ones, since
averaging multiple embeddings together raises similarity for the same reason a
multi-image enrollment template scores higher than a single probe image
(see docs/hard_negative_sampling.md and the NIST/template-adaptation literature
referenced there).

This is still an approximation of the true NIST methodology (which compares
literally every enrolled image pair), since cross-identity comparisons are
limited to one representative image per identity for tractability at ~3.6M
identities. Treat the output as a prioritized candidate list for inspection /
hard-negative seeding (see generate_hard_cases.py), not as a certified metric.

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


def _list_file_prefixes(embeddings_dir):
    """Read the hive partition directory names (file_prefix=...) without scanning any rows."""
    return sorted(
        name.split("file_prefix=", 1)[1]
        for name in os.listdir(embeddings_dir)
        if name.startswith("file_prefix=")
    )


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


# Edit these to match your setup, or override per-run with the matching --flag.
DEFAULT_CONFIG = "configs/wf42m_pfc03_40epoch_64gpu_vit_l"
DEFAULT_EMBEDDINGS_DIR = "/path/to/hard_case_out"


def main():
    parser = argparse.ArgumentParser(description="Find hard genuine/impostor cases at a target FMR")
    parser.add_argument("config", type=str, nargs="?", default=DEFAULT_CONFIG,
                         help=f"default: {DEFAULT_CONFIG}")
    parser.add_argument("--embeddings-dir", type=str, default=DEFAULT_EMBEDDINGS_DIR,
                         help=f"output-dir of embed_dataset.py (Parquet dataset), default: {DEFAULT_EMBEDDINGS_DIR}")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="defaults to --embeddings-dir")
    parser.add_argument("--fmr", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=50,
                         help="nearest-medoid candidates kept per class for the impostor search")
    parser.add_argument("--centroid-chunk", type=int, default=1024)
    parser.add_argument("--read-batch-size", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.embeddings_dir

    cfg = get_config(args.config)
    dataset = _open_embedding_dataset(args.embeddings_dir)
    device = torch.device(args.device)
    embed_dim, num_classes = cfg.embedding_size, cfg.num_classes
    n_rows = dataset.count_rows()
    n_read_batches = -(-n_rows // args.read_batch_size)  # ceil div
    prefixes = _list_file_prefixes(args.embeddings_dir)
    prefix_to_id = {p: i for i, p in enumerate(prefixes)}

    print("Pass 1/4: accumulating per-class centroids (used only to pick each identity's medoid) ...")
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
    del centroid_sum

    print("Pass 2/4: finding each identity's medoid (the real image closest to its centroid) ...")
    medoid_score = torch.full((num_classes,), -2.0, dtype=torch.float32)
    medoid_embedding = torch.zeros((num_classes, embed_dim), dtype=torch.float32)
    medoid_prefix_id = torch.full((num_classes,), -1, dtype=torch.int32)
    medoid_rec_idx = torch.full((num_classes,), -1, dtype=torch.int64)
    for embeddings, identity, rec_idx, file_prefix in tqdm(_iter_batches(dataset, embed_dim, args.read_batch_size),
                                                            total=n_read_batches, desc="pass2 medoid search",
                                                            unit="batch"):
        emb = torch.from_numpy(embeddings)
        lbl = torch.from_numpy(identity)
        prefix_id = torch.tensor([prefix_to_id[p] for p in file_prefix], dtype=torch.int32)
        rec_idx_t = torch.from_numpy(rec_idx)
        sim = (emb * centroids[lbl]).sum(dim=1)

        improved = (sim > medoid_score[lbl]).nonzero(as_tuple=True)[0]
        if improved.numel() > 0:
            # ascending sort so that, for duplicate classes within this batch, the assignment
            # below (last-write-wins for repeated indices) keeps the highest sim per class
            order = torch.argsort(sim[improved])
            idx = improved[order]
            classes = lbl[idx]
            medoid_score[classes] = sim[idx]
            medoid_embedding[classes] = emb[idx]
            medoid_prefix_id[classes] = prefix_id[idx]
            medoid_rec_idx[classes] = rec_idx_t[idx]
    del centroids

    print("Pass 3/4: searching nearest other-identity medoids (impostor candidates) ...")
    medoid_dev = medoid_embedding.to(device)
    has_images_dev = has_images.to(device)
    valid_idx = torch.nonzero(has_images, as_tuple=True)[0]
    chunk, topk = args.centroid_chunk, min(args.topk, n_with_images - 1)

    cand_a, cand_b, cand_score = [], [], []
    chunk_starts = list(range(0, valid_idx.numel(), chunk))
    for start in tqdm(chunk_starts, desc="pass3 impostor search", unit="chunk"):
        rows = valid_idx[start:start + chunk].to(device)
        sims = medoid_dev[rows] @ medoid_dev.t()
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

    print("Pass 4/4: scoring genuine images (real image vs their identity's medoid) against threshold ...")
    false_reject_rows = []
    for embeddings, identity, rec_idx, file_prefix in tqdm(_iter_batches(dataset, embed_dim, args.read_batch_size),
                                                             total=n_read_batches, desc="pass4 genuine scoring",
                                                             unit="batch"):
        emb = torch.from_numpy(embeddings)
        lbl = torch.from_numpy(identity)
        prefix_id = torch.tensor([prefix_to_id[p] for p in file_prefix], dtype=torch.int32)
        rec_idx_t = torch.from_numpy(rec_idx)

        # skip images that are themselves the medoid - comparing an image to itself is meaningless
        is_medoid_itself = (prefix_id == medoid_prefix_id[lbl]) & (rec_idx_t == medoid_rec_idx[lbl])
        valid = (~is_medoid_itself).nonzero(as_tuple=True)[0]
        genuine_score = (emb[valid] * medoid_embedding[lbl[valid]]).sum(dim=1)

        flagged_local = (genuine_score < threshold).nonzero(as_tuple=True)[0]
        for li in flagged_local.tolist():
            gi = valid[li].item()
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

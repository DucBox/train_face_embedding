"""
Cluster the CFP embeddings (from embed_cfpw.py) with DBSCAN and check two
things relevant to using DBSCAN for identity discovery:

  1. Drop rate: what % of images end up unassigned (DBSCAN label == -1, "noise")
     or, after picking each identity's single best cluster, simply outside it.
  2. Pose-split: for identities that end up split across >1 cluster, is the
     split a clean frontal/profile separation (i.e. DBSCAN is treating the same
     person's frontal and profile photos as two different "identities" purely
     because of pose, not because the embeddings are actually unrelated)?

"Best cluster" per identity = the non-noise cluster with the most member images
for that identity. Everything else for that identity (other clusters + noise)
is "ignored" - this mirrors how you'd actually use DBSCAN output in practice
(keep the dominant cluster as the real identity, drop the rest).

    python3 run_dbscan_cfp.py --embeddings cfp_embeddings.parquet --eps 0.3 --min-samples 4

NOTE on --min-samples: sklearn's DBSCAN requires an integer count of points
(it is not a fraction of anything) - "min_samples=0.3" as stated is not a
valid value and will raise an error if passed literally. Pick the smallest
cluster size you'd still trust as "real" (e.g. 4, given each identity here has
10 frontal + 4 profile images) and pass it as an int via --min-samples.

Writes:
    --output (parquet)  : id_number, image_type, seq_no, cluster, in_best_cluster
    --viz-dir (PNGs)     : one grid per identity - top half = best-cluster images,
                            bottom half = ignored images (green border = frontal,
                            orange border = profile). Skip with --no-viz.
"""
import argparse
import os

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import DBSCAN

DEFAULT_EMBEDDINGS = "cfp_embeddings.parquet"
DEFAULT_OUTPUT = "dbscan_cfp_clusters.parquet"
DEFAULT_CFP_DIR = "cfp-dataset/Data/Images"
DEFAULT_VIZ_DIR = "dbscan_cfp_viz"

THUMB_SIZE = (64, 80)
BORDER = 4
TYPE_COLOR = {"frontal": (0, 160, 0), "profile": (200, 80, 0)}


def pick_best_cluster(df):
    """Per id_number, the non-noise cluster with the most member rows; None if all noise."""
    counts = (
        df.filter(pl.col("cluster") != -1)
        .group_by(["id_number", "cluster"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    best = (
        counts.group_by("id_number", maintain_order=True)
        .first()
        .select("id_number", pl.col("cluster").alias("best_cluster"))
    )
    return df.join(best, on="id_number", how="left").with_columns(
        (pl.col("cluster") == pl.col("best_cluster")).fill_null(False).alias("in_best_cluster")
    )


def summarize_splits(df):
    """For identities with >1 non-noise cluster, check whether the split is a clean
    frontal/profile separation. Returns a list of (id_number, n_clusters, is_pose_split)."""
    splits = []
    non_noise = df.filter(pl.col("cluster") != -1)
    for (id_number,), g in non_noise.group_by(["id_number"]):
        clusters_used = g["cluster"].unique()
        if clusters_used.len() <= 1:
            continue
        types_per_cluster = g.group_by("cluster").agg(pl.col("image_type").n_unique().alias("n_types"))
        is_pose_split = bool((types_per_cluster["n_types"] == 1).all() and g["image_type"].n_unique() > 1)
        splits.append((id_number, clusters_used.len(), is_pose_split))
    return sorted(splits, key=lambda r: r[0])


def summarize_retention(df):
    """Per id_number x image_type: how many images were kept in the best cluster
    vs the identity's total for that pose, e.g. id 007 has 7 frontal + 3 profile,
    best cluster keeps 6 frontal + 2 profile -> keep_ratio 0.857 frontal, 0.667 profile."""
    return (
        df.group_by(["id_number", "image_type"])
        .agg(pl.len().alias("total"), pl.col("in_best_cluster").sum().alias("kept"))
        .with_columns((pl.col("kept") / pl.col("total")).alias("keep_ratio"))
        .sort(["id_number", "image_type"])
    )


def print_retention_stats(retention):
    print("\nPer-identity retention rate (images kept in best cluster / total for that pose):")
    for image_type in ["frontal", "profile"]:
        vals = retention.filter(pl.col("image_type") == image_type)["keep_ratio"]
        if vals.len() == 0:
            continue
        print(f"  {image_type:8s}: min={vals.min():6.1%}  max={vals.max():6.1%}  "
              f"mean={vals.mean():6.1%}  median={vals.median():6.1%}  (n={vals.len()} identities)")


def _load_thumb(cfp_dir, id_number, image_type, seq_no):
    path = os.path.join(cfp_dir, f"{id_number:03d}", image_type, f"{seq_no:02d}.jpg")
    img = Image.open(path).convert("RGB").resize(THUMB_SIZE)
    return ImageOps.expand(img, border=BORDER, fill=TYPE_COLOR.get(image_type, (128, 128, 128)))


def visualize_identity(cfp_dir, id_number, best_rows, ignored_rows, out_path):
    """best_rows/ignored_rows: list of (image_type, seq_no). Top row = best cluster,
    bottom row = everything ignored for this identity."""
    cell_w, cell_h = THUMB_SIZE[0] + 2 * BORDER, THUMB_SIZE[1] + 2 * BORDER
    ncols = max(len(best_rows), len(ignored_rows), 1)
    title_h = 20
    canvas = Image.new("RGB", (ncols * cell_w, title_h + 2 * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"id {id_number:03d}  best={len(best_rows)}  ignored={len(ignored_rows)}",
              fill=(0, 0, 0))

    for i, (image_type, seq_no) in enumerate(best_rows):
        canvas.paste(_load_thumb(cfp_dir, id_number, image_type, seq_no), (i * cell_w, title_h))
    for i, (image_type, seq_no) in enumerate(ignored_rows):
        canvas.paste(_load_thumb(cfp_dir, id_number, image_type, seq_no), (i * cell_w, title_h + cell_h))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="DBSCAN clustering test on CFP embeddings")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS,
                         help=f"output of embed_cfpw.py, default: {DEFAULT_EMBEDDINGS}")
    parser.add_argument("--eps", type=float, default=0.3,
                         help="cosine DISTANCE eps, i.e. 1 - cosine_similarity (default: 0.3 -> "
                              "same-cluster requires cosine_similarity >= 0.7)")
    parser.add_argument("--min-samples", type=int, default=4,
                         help="sklearn DBSCAN requires an int point count, not a fraction - see "
                              "the note at the top of this script (default: 4)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help=f"default: {DEFAULT_OUTPUT}")
    parser.add_argument("--cfp-dir", type=str, default=DEFAULT_CFP_DIR,
                         help=f"original images, for --viz-dir output, default: {DEFAULT_CFP_DIR}")
    parser.add_argument("--viz-dir", type=str, default=DEFAULT_VIZ_DIR, help=f"default: {DEFAULT_VIZ_DIR}")
    parser.add_argument("--no-viz", action="store_true", help="skip generating the per-identity grid PNGs")
    args = parser.parse_args()

    df = pl.read_parquet(args.embeddings)
    embeddings = df["embedding"].to_numpy()

    print(f"Loaded {len(df):,} images across {df['id_number'].n_unique():,} identities")
    print(f"Running DBSCAN(eps={args.eps}, min_samples={args.min_samples}, metric=cosine) ...")
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine").fit_predict(embeddings)
    df = df.with_columns(pl.Series("cluster", labels.astype(np.int32)))
    df = pick_best_cluster(df)

    n_total = len(df)
    n_noise = int((df["cluster"] == -1).sum())
    n_ignored = int((~df["in_best_cluster"]).sum())
    n_clusters = df.filter(pl.col("cluster") != -1)["cluster"].n_unique()
    n_ids = df["id_number"].n_unique()

    print(f"\nTotal images: {n_total:,}")
    print(f"DBSCAN noise (cluster == -1): {n_noise:,} ({100 * n_noise / n_total:.2f}%)")
    print(f"Ignored after keeping only each identity's best cluster: "
          f"{n_ignored:,} ({100 * n_ignored / n_total:.2f}%)")
    print(f"Clusters found (excluding noise): {n_clusters:,} (true identity count: {n_ids:,})")

    splits = summarize_splits(df)
    n_pose_split = sum(1 for _, _, pose in splits if pose)
    print(f"\nIdentities split into >1 cluster: {len(splits):,}/{n_ids:,} "
          f"({100 * len(splits) / n_ids:.2f}%)")
    print(f"  of those, cleanly split along frontal/profile lines: {n_pose_split:,} "
          f"({100 * n_pose_split / n_ids:.2f}% of all identities)")
    if splits:
        print("\nSplit identities (id_number, n_clusters, is_pose_split):")
        for id_number, n_clusters_used, is_pose_split in splits:
            print(f"  {id_number:>4} | {n_clusters_used} clusters | pose_split={is_pose_split}")

    retention = summarize_retention(df)
    print_retention_stats(retention)

    df.sort(["id_number", "image_type", "seq_no"]).write_parquet(args.output)
    print(f"\nPer-image cluster assignment written to {args.output}")

    if not args.no_viz:
        os.makedirs(args.viz_dir, exist_ok=True)
        for (id_number,), g in df.sort(["image_type", "seq_no"]).group_by(["id_number"], maintain_order=True):
            best_rows = list(zip(g.filter(pl.col("in_best_cluster"))["image_type"],
                                  g.filter(pl.col("in_best_cluster"))["seq_no"]))
            ignored_rows = list(zip(g.filter(~pl.col("in_best_cluster"))["image_type"],
                                     g.filter(~pl.col("in_best_cluster"))["seq_no"]))
            out_path = os.path.join(args.viz_dir, f"id_{id_number:03d}.png")
            visualize_identity(args.cfp_dir, id_number, best_rows, ignored_rows, out_path)
        print(f"Per-identity grid visualizations written to {args.viz_dir}/ ({n_ids:,} PNGs)")


if __name__ == "__main__":
    main()

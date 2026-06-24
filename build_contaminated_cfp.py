"""
Build a CONTAMINATED version of the CFP set for fair DBSCAN eps calibration.

CFP is clean (each identity only has its own frontal/profile images), so it can
only measure the FRR side (do we keep a person's hard images together). To also
measure the FAR side (do other people's images leak into a cluster), we inject
HARD impostors into every identity:

  - template (medoid) embedding per identity
  - for each host id, find the N nearest OTHER ids (template-template cosine)
  - from each near id, inject the images MOST SIMILAR to the host, where
    similarity = MAX cosine over ALL host images (option b) - i.e. the single
    closest link, which is exactly what lets an impostor join the host cluster
    via DBSCAN's chaining. (Not similarity to the host medoid.)
  - configurable how many frontal / profile to inject per near id.

Saves both:
  * a folder mirroring align_cfp's layout but renamed (default
    cfp-dataset/contaminated_data/<host>/<pose>/...), host's own crops keep
    their name, injected crops are named inj_<srcid>_<seq>.jpg
  * a parquet manifest (default contaminated_cfp_embeddings.parquet) carrying
    embeddings + provenance for run_dbscan_cfp.py:
        assigned_id int32   - host id the row is grouped under (for clustering)
        src_id      int32   - the real identity the image came from (for viz)
        image_type  string  - frontal / profile
        seq_no      int32   - original seq number within src_id+pose
        is_injected bool     - False for the host's own images, True for impostors
        sim_to_host float32 - max cosine to any host image (1.0 sentinel for own)
        rel_path    string  - path of the crop inside --output-dir
        embedding   list<float32>[D]

    python3 build_contaminated_cfp.py --embeddings cfp_embeddings.parquet \
        --processed-dir cfp-dataset/processed_data \
        --output-dir cfp-dataset/contaminated_data \
        --output-parquet contaminated_cfp_embeddings.parquet \
        --n-near-ids 3 --n-frontal 2 --n-profile 1
"""
import argparse
import os
import shutil

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_embeddings(path):
    """-> (df, emb[N,D] float32 L2-normalized).

    embed_cfpw.py already L2-normalizes, nhưng vẫn normalize lại ở đây cho chắc:
    medoid / nearest-id / sim_to_host đều dựa trên dot = cosine nên embedding PHẢI
    là unit-norm thì số mới đúng.
    """
    df = pl.read_parquet(path)
    emb = np.asarray(df["embedding"].to_numpy(), dtype=np.float32)
    if emb.ndim == 1:  # object array of lists -> stack
        emb = np.stack([np.asarray(v, dtype=np.float32) for v in df["embedding"].to_list()])
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    emb = emb / norm
    return df, emb


def compute_medoids(emb, id_rows):
    """Medoid per id = the real image maximizing summed cosine to its id's images."""
    medoids = {}
    for cid, rows in id_rows.items():
        E = emb[rows]                      # (k, D)
        S = E @ E.T                        # (k, k)
        medoids[cid] = E[int(np.argmax(S.sum(axis=1)))]
    return medoids


def nearest_ids(medoids, n_near):
    """For each id -> list of n_near other ids, by template-template cosine (desc)."""
    ids = sorted(medoids.keys())
    M = np.stack([medoids[c] for c in ids])          # (G, D)
    sim = M @ M.T
    np.fill_diagonal(sim, -np.inf)
    out = {}
    for i, cid in enumerate(ids):
        order = np.argsort(-sim[i])[:n_near]
        out[cid] = [ids[j] for j in order]
    return out


def select_injections(host, near_list, emb, host_rows, pose_rows, n_frontal, n_profile):
    """For one host, pick impostor rows from each near id by MAX cosine to any host image.

    pose_rows[(src_id, pose)] -> np.array of row indices.
    Returns list of (src_row_index, src_id, pose, sim_to_host).
    """
    host_E = emb[host_rows]                           # (h, D)
    picks = []
    for src in near_list:
        for pose, n_take in (("frontal", n_frontal), ("profile", n_profile)):
            if n_take <= 0:
                continue
            cand = pose_rows.get((src, pose))
            if cand is None or len(cand) == 0:
                continue
            sims = (emb[cand] @ host_E.T).max(axis=1)  # max over host images (option b)
            order = np.argsort(-sims)[:n_take]
            for j in order:
                picks.append((int(cand[j]), src, pose, float(sims[j])))
    return picks


def parse_args():
    ap = argparse.ArgumentParser(description="Build contaminated CFP for DBSCAN eps calibration")
    ap.add_argument("--embeddings", default="cfp_embeddings.parquet", help="output of embed_cfpw.py")
    ap.add_argument("--processed-dir", default="cfp-dataset/processed_data",
                    help="aligned crops from align_cfp.py (to copy into the new folder)")
    ap.add_argument("--output-dir", default="cfp-dataset/contaminated_data",
                    help="new folder for the contaminated crops")
    ap.add_argument("--output-parquet", default="contaminated_cfp_embeddings.parquet")
    ap.add_argument("--n-near-ids", type=int, default=3, help="số id gần nhất tiêm vào mỗi host")
    ap.add_argument("--n-frontal", type=int, default=2, help="số ảnh frontal tiêm / near id")
    ap.add_argument("--n-profile", type=int, default=1, help="số ảnh profile tiêm / near id")
    ap.add_argument("--no-copy", action="store_true", help="chỉ ghi parquet, không copy ảnh")
    return ap.parse_args()


def main():
    args = parse_args()
    df, emb = load_embeddings(args.embeddings)
    D = emb.shape[1]
    ids = df["id_number"].to_numpy()
    types = df["image_type"].to_list()
    seqs = df["seq_no"].to_numpy()
    print(f"[load] {len(df):,} ảnh, {len(set(ids))} id, D={D}")

    # index: id -> rows ; (id,pose) -> rows
    id_rows, pose_rows = {}, {}
    for r in range(len(df)):
        cid = int(ids[r])
        id_rows.setdefault(cid, []).append(r)
        pose_rows.setdefault((cid, types[r]), []).append(r)
    id_rows = {k: np.array(v) for k, v in id_rows.items()}
    pose_rows = {k: np.array(v) for k, v in pose_rows.items()}

    medoids = compute_medoids(emb, id_rows)
    near = nearest_ids(medoids, args.n_near_ids)

    # assemble rows: host's own (is_injected=False) + injected (True)
    rec_assigned, rec_src, rec_type, rec_seq = [], [], [], []
    rec_inj, rec_sim, rec_path, rec_emb = [], [], [], []

    def add_row(assigned, src, pose, seq, is_inj, sim, row_idx):
        if is_inj:
            fname = f"inj_{src:03d}_{seq:02d}.jpg"
        else:
            fname = f"{seq:02d}.jpg"
        rel = os.path.join(f"{assigned:03d}", pose, fname)
        rec_assigned.append(assigned); rec_src.append(src); rec_type.append(pose)
        rec_seq.append(seq); rec_inj.append(is_inj); rec_sim.append(sim)
        rec_path.append(rel); rec_emb.append(emb[row_idx])

    for host in tqdm(sorted(id_rows.keys()), desc="build", unit="id"):
        for r in id_rows[host]:
            add_row(host, host, types[r], int(seqs[r]), False, 1.0, int(r))
        for row_idx, src, pose, sim in select_injections(
                host, near[host], emb, id_rows[host], pose_rows, args.n_frontal, args.n_profile):
            add_row(host, src, pose, int(seqs[row_idx]), True, sim, row_idx)

    n_inj = int(np.sum(rec_inj))
    print(f"[build] {len(rec_assigned):,} dòng ({n_inj:,} tiêm, "
          f"~{n_inj/len(id_rows):.1f}/id)")

    # copy crops
    if not args.no_copy:
        n_miss = 0
        for assigned, src, pose, seq, is_inj, rel in tqdm(
                list(zip(rec_assigned, rec_src, rec_type, rec_seq, rec_inj, rec_path)),
                desc="copy", unit="img"):
            src_path = os.path.join(args.processed_dir, f"{src:03d}", pose, f"{seq:02d}.jpg")
            dst_path = os.path.join(args.output_dir, rel)
            if not os.path.exists(src_path):
                n_miss += 1
                continue
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        if n_miss:
            print(f"[warn] {n_miss} ảnh nguồn không tìm thấy trong {args.processed_dir}")
        print(f"[copy] -> {args.output_dir}/")

    # write parquet manifest
    table = pa.table({
        "assigned_id": pa.array(np.array(rec_assigned, dtype=np.int32)),
        "src_id": pa.array(np.array(rec_src, dtype=np.int32)),
        "image_type": pa.array(rec_type, type=pa.string()),
        "seq_no": pa.array(np.array(rec_seq, dtype=np.int32)),
        "is_injected": pa.array(np.array(rec_inj, dtype=bool)),
        "sim_to_host": pa.array(np.array(rec_sim, dtype=np.float32)),
        "rel_path": pa.array(rec_path, type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(np.asarray(rec_emb, dtype=np.float32).reshape(-1), type=pa.float32()), D),
    })
    out_dir = os.path.dirname(os.path.abspath(args.output_parquet))
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(table, args.output_parquet)
    print(f"[done] manifest -> {args.output_parquet}  ({len(rec_assigned):,} dòng)")


if __name__ == "__main__":
    main()

import os
import numpy as np
import polars as pl
import faiss
import igraph as ig

# --- CONFIG ---
CENTERS_PATH = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_results/template_v1/centers.parquet"
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/template_results/template_v1/merge_clean"

LOWER_THR = 0.43
UPPER_THR = 0.56
DIMENSION = 512
N_CLUSTERS = 4096
N_PROBE = 64

NUM_DEBUG_IDS = None  # None = chạy toàn bộ; set int để test nhanh


def run_stats():
    # --- STEP 1: LOAD CENTERS ---
    print(f"[STEP 1] Loading Centers from {CENTERS_PATH}...")
    df = pl.read_parquet(CENTERS_PATH)

    if NUM_DEBUG_IDS is not None:
        print(f"!!! DEBUG MODE: only first {NUM_DEBUG_IDS} IDs !!!")
        df = df.head(NUM_DEBUG_IDS)

    ids = df["person_id"].to_list()
    counts = df["img_count"].to_numpy()
    vecs = np.stack(df["embedding_center"].to_numpy()).astype("float32")
    idx_to_id = {i: pid for i, pid in enumerate(ids)}

    n_total = len(ids)
    total_imgs = int(counts.sum())
    print(f" -> Loaded {n_total} centers | {total_imgs} images total.")

    # --- STEP 2: FAISS SEARCH ---
    actual_clusters = min(N_CLUSTERS, int(n_total / 30))
    if actual_clusters < 1:
        actual_clusters = 1

    print(f"[STEP 2] Building FAISS Index (IVF{actual_clusters})...")
    quantizer = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, actual_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = N_PROBE

    print(f" -> Range Search (> {LOWER_THR})...")
    lims, D, I = index.range_search(vecs, LOWER_THR)
    lims = lims.astype("int64")
    D = D.astype("float32")
    I = I.astype("int64")

    # --- STEP 3: EDGES ---
    print("[STEP 3] Building edges...")
    sources = np.repeat(np.arange(n_total), np.diff(lims))
    targets = I
    scores = D

    mask_valid = sources < targets
    sources = sources[mask_valid]
    targets = targets[mask_valid]
    scores = scores[mask_valid]

    mask_merge = scores > UPPER_THR
    mask_drop = (scores > LOWER_THR) & (scores <= UPPER_THR)

    merge_edges = np.column_stack((sources[mask_merge], targets[mask_merge]))
    drop_src = sources[mask_drop]
    drop_tgt = targets[mask_drop]

    n_merge_edges = len(merge_edges)
    n_drop_pairs = len(drop_src)
    print(f" -> {n_merge_edges} merge edges | {n_drop_pairs} drop-candidate pairs.")

    # --- STEP 4: GRAPH COMPONENTS ---
    print("[STEP 4] Solving graph components...")
    if n_merge_edges > 0:
        g = ig.Graph(n_total, edges=merge_edges, directed=False)
        components = g.connected_components(mode="weak")
    else:
        components = []

    merge_map_idx = {i: i for i in range(n_total)}
    leader_counts = counts.copy()

    cluster_sizes = []  # số ID trong mỗi cụm bị gộp (>=2)
    for comp in components:
        if len(comp) < 2:
            continue
        cluster_sizes.append(len(comp))
        leader = max(comp, key=lambda x: (counts[x], -x))
        total = sum(counts[node] for node in comp)
        leader_counts[leader] = total
        for node in comp:
            merge_map_idx[node] = leader

    # --- STEP 5: DROPS ---
    final_drop_indices = set()
    for u, v in zip(drop_src, drop_tgt):
        lu, lv = merge_map_idx[u], merge_map_idx[v]
        if lu == lv:
            continue
        if leader_counts[lu] > leader_counts[lv]:
            final_drop_indices.add(lv)
        elif leader_counts[lv] > leader_counts[lu]:
            final_drop_indices.add(lu)
        else:
            final_drop_indices.add(max(lu, lv))

    # --- RESOLVE TO REAL IDS ---
    real_drop_ids = set()
    for idx, leader in merge_map_idx.items():
        if leader in final_drop_indices:
            real_drop_ids.add(idx_to_id[idx])

    real_merge_map = {}
    for idx, leader in merge_map_idx.items():
        if idx_to_id[idx] in real_drop_ids:
            continue
        if idx != leader:
            real_merge_map[idx_to_id[idx]] = idx_to_id[leader]

    # --- SAVE RESULTS ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) ID bị DROP (loại hẳn)
    drop_path = os.path.join(OUTPUT_DIR, "drop_ids.csv")
    pl.DataFrame({"person_id": sorted(real_drop_ids)}).write_csv(drop_path)

    # 2) Mapping ID -> leader (ID bị merge gộp vào ID nào)
    map_path = os.path.join(OUTPUT_DIR, "merge_map.csv")
    pl.DataFrame({
        "person_id": list(real_merge_map.keys()),
        "merged_into": list(real_merge_map.values()),
    }).write_csv(map_path)

    # 3) Các cụm merge: leader + danh sách thành viên (gồm cả leader)
    #    bỏ qua các ID đã bị drop khỏi cụm.
    groups = {}  # leader_id -> [member_id, ...]
    for idx, leader in merge_map_idx.items():
        if idx == leader:
            continue  # leader sẽ được thêm khi xét chính nó / bên dưới
        if idx_to_id[idx] in real_drop_ids:
            continue
        groups.setdefault(idx_to_id[leader], []).append(idx_to_id[idx])
    group_rows = []
    for leader_id, members in groups.items():
        all_members = [leader_id] + members
        group_rows.append({
            "leader_id": leader_id,
            "members": " ".join(str(m) for m in all_members),
            "n_members": len(all_members),
        })
    groups_path = os.path.join(OUTPUT_DIR, "merge_groups.csv")
    pl.DataFrame(group_rows).write_csv(groups_path)

    print(f"\n[SAVE] -> {drop_path} ({len(real_drop_ids)} dropped ids)")
    print(f"[SAVE] -> {map_path} ({len(real_merge_map)} merged ids)")
    print(f"[SAVE] -> {groups_path} ({len(group_rows)} merge groups)")

    # --- STATS ---
    n_clusters = len(cluster_sizes)
    n_drop = len(real_drop_ids)
    n_merge = len(real_merge_map)
    n_leaders = n_clusters  # mỗi cụm còn 1 leader
    n_kept = n_total - n_drop - n_merge

    drop_imgs = int(sum(counts[i] for i in range(n_total) if idx_to_id[i] in real_drop_ids))

    print("\n" + "=" * 50)
    print("THỐNG KÊ MERGE / DROP (chỉ trên centers)")
    print("=" * 50)
    print(f"Tổng số ID đầu vào          : {n_total}")
    print(f"Tổng số ảnh                 : {total_imgs}")
    print("-" * 50)
    print(f"Số cụm bị gộp (merge groups): {n_clusters}")
    if cluster_sizes:
        cs = np.array(cluster_sizes)
        print(f"  - kích thước cụm: min={cs.min()} max={cs.max()} "
              f"mean={cs.mean():.2f} (tổng ID trong cụm={cs.sum()})")
    print(f"ID bị MERGE (gộp vào leader): {n_merge}")
    print(f"ID bị DROP (loại hẳn)       : {n_drop}  ({drop_imgs} ảnh)")
    print(f"ID giữ nguyên (kept)        : {n_kept}")
    print("-" * 50)
    print(f"Số ID còn lại sau clean     : {n_total - n_drop - n_merge}")
    print(f"  (kept {n_kept} + leaders {n_leaders})")
    print(f"Tỷ lệ giảm ID               : "
          f"{(n_drop + n_merge) / n_total * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_stats()

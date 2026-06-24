"""
DBSCAN trên bộ CFP đã tiêm impostor (build_contaminated_cfp.py) để calib eps.

Khác run_dbscan_cfp.py (chỉ đo FRR trên bộ sạch): bộ này có cả ảnh thật của host
(is_injected=False) lẫn ảnh người khác tiêm vào (is_injected=True), nên đo được
CẢ HAI phía:

  - RETENTION (phía FRR): ảnh thật của host (frontal/profile) còn ở cụm chính không
  - CONTAMINATION (phía FAR): ảnh tiêm có lọt vào cụm chính không (= gộp nhầm người)

Cluster PER-HOST: với mỗi assigned_id, chạy DBSCAN riêng trên pool của nó
(ảnh thật + ảnh tiêm) - đúng kịch bản "clean trong cùng 1 ID label". "Cụm chính"
= cụm non-noise có nhiều ảnh THẬT của host nhất.

    # 1 ngưỡng, in chi tiết + ghi per-row parquet (+ viz)
    python3 run_dbscan_contaminated.py --manifest contaminated_cfp_embeddings.parquet \
        --eps 0.44 --min-samples 4

    # quét nhiều eps -> bảng trade-off retention vs contamination
    python3 run_dbscan_contaminated.py --manifest contaminated_cfp_embeddings.parquet \
        --eps-sweep 0.30,0.35,0.40,0.44,0.50 --min-samples 4
"""
import argparse
import os

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN


def load_manifest(path):
    df = pl.read_parquet(path)
    emb = np.asarray(df["embedding"].to_numpy(), dtype=np.float32)
    if emb.ndim == 1:
        emb = np.stack([np.asarray(v, dtype=np.float32) for v in df["embedding"].to_list()])
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return df, emb / norm


def cluster_one_host(emb_h, eps, min_samples):
    """DBSCAN cosine cho pool 1 host -> nhãn cụm (mảng int, -1 = noise)."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb_h)


def host_report(sub, labels):
    """Thống kê 1 host. sub: dict các cột list cùng độ dài; labels: nhãn cụm.

    Trả về dict đếm: own/injected kept-in-main theo pose, drop, contamination.
    """
    is_inj = np.asarray(sub["is_injected"])
    pose = np.asarray(sub["image_type"])
    labels = np.asarray(labels)

    genuine = ~is_inj
    # cụm chính = cụm non-noise có nhiều ảnh THẬT của host nhất
    main = None
    best = -1
    for c in set(labels[labels != -1]):
        cnt = int(((labels == c) & genuine).sum())
        if cnt > best:
            best, main = cnt, c
    in_main = (labels == main) if main is not None else np.zeros(len(labels), bool)

    def counts(mask):
        return {
            "frontal": int((mask & (pose == "frontal")).sum()),
            "profile": int((mask & (pose == "profile")).sum()),
            "all": int(mask.sum()),
        }

    own_total = counts(genuine)
    own_kept = counts(genuine & in_main)            # giữ được (retention)
    inj_total = counts(is_inj)
    inj_leak = counts(is_inj & in_main)             # lọt vào cụm chính (contamination)
    return {"own_total": own_total, "own_kept": own_kept,
            "inj_total": inj_total, "inj_leak": inj_leak,
            "in_main": in_main, "main_cluster": main}


def aggregate(df, emb, eps, min_samples):
    """Chạy per-host, gộp số. Trả về (agg_dict, per_row_records)."""
    rows = {k: df[k].to_list() for k in
            ["assigned_id", "src_id", "image_type", "seq_no", "is_injected", "sim_to_host", "rel_path"]}
    order = np.argsort(np.asarray(rows["assigned_id"]), kind="stable")
    hosts = {}
    for r in order:
        hosts.setdefault(rows["assigned_id"][r], []).append(r)

    # tích luỹ
    ret_frontal, ret_profile = [], []                # per-host retention ratios
    tot = {k: 0 for k in ["own_F", "own_P", "kept_F", "kept_P",
                          "inj_F", "inj_P", "leak_F", "leak_P", "own_drop"]}
    hosts_with_contam = 0
    leak_sims, clean_sims = [], []                    # sim_to_host của ảnh tiêm: lọt vs bị loại
    per_row = []

    for host, idxs in hosts.items():
        idxs = np.array(idxs)
        sub = {k: [rows[k][i] for i in idxs] for k in rows}
        rep = host_report(sub, cluster_one_host(emb[idxs], eps, min_samples))

        ot, ok = rep["own_total"], rep["own_kept"]
        it, il = rep["inj_total"], rep["inj_leak"]
        if ot["frontal"]:
            ret_frontal.append(ok["frontal"] / ot["frontal"])
        if ot["profile"]:
            ret_profile.append(ok["profile"] / ot["profile"])
        for a, b in (("own_F", ot["frontal"]), ("own_P", ot["profile"]),
                     ("kept_F", ok["frontal"]), ("kept_P", ok["profile"]),
                     ("inj_F", it["frontal"]), ("inj_P", it["profile"]),
                     ("leak_F", il["frontal"]), ("leak_P", il["profile"])):
            tot[a] += b
        tot["own_drop"] += (ot["all"] - ok["all"])
        if il["all"] > 0:
            hosts_with_contam += 1

        in_main = rep["in_main"]
        for j, i in enumerate(idxs):
            if rows["is_injected"][i]:
                (leak_sims if in_main[j] else clean_sims).append(rows["sim_to_host"][i])
            per_row.append((rows["assigned_id"][i], rows["src_id"][i], rows["image_type"][i],
                            rows["seq_no"][i], rows["is_injected"][i], rows["sim_to_host"][i],
                            int(rep["main_cluster"]) if rep["main_cluster"] is not None else -999,
                            bool(in_main[j]), rows["rel_path"][i]))

    agg = {
        "n_hosts": len(hosts),
        "ret_frontal": np.array(ret_frontal), "ret_profile": np.array(ret_profile),
        "tot": tot, "hosts_with_contam": hosts_with_contam,
        "leak_sims": np.array(leak_sims), "clean_sims": np.array(clean_sims),
    }
    return agg, per_row


def pct(n, d):
    return f"{100*n/d:.2f}%" if d else "n/a"


def print_detail(agg, eps, min_samples):
    t = agg["tot"]
    print(f"\n{'='*64}\n  eps={eps}  min_samples={min_samples}  hosts={agg['n_hosts']}\n{'='*64}")

    print("\n[RETENTION] ảnh THẬT của host giữ được trong cụm chính:")
    for name, arr, kept, total in (("frontal", agg["ret_frontal"], t["kept_F"], t["own_F"]),
                                   ("profile", agg["ret_profile"], t["kept_P"], t["own_P"])):
        if len(arr):
            print(f"  {name:8s}: micro={pct(kept,total):>7}  | per-host "
                  f"mean={arr.mean()*100:5.1f}% median={np.median(arr)*100:5.1f}% "
                  f"min={arr.min()*100:5.1f}% (n={len(arr)} id)")
    print(f"  own dropped (noise/khác cụm): {pct(t['own_drop'], t['own_F']+t['own_P'])}")

    print("\n[CONTAMINATION] ảnh TIÊM (người khác) lọt vào cụm chính:")
    leak_all = t["leak_F"] + t["leak_P"]
    inj_all = t["inj_F"] + t["inj_P"]
    print(f"  tổng     : {pct(leak_all, inj_all)}  ({leak_all}/{inj_all} ảnh tiêm bị gộp nhầm)")
    print(f"  frontal  : {pct(t['leak_F'], t['inj_F'])}")
    print(f"  profile  : {pct(t['leak_P'], t['inj_P'])}")
    print(f"  host bị dính ít nhất 1 ảnh lẫn: {pct(agg['hosts_with_contam'], agg['n_hosts'])}")
    ls, cs = agg["leak_sims"], agg["clean_sims"]
    if len(ls):
        print(f"  sim_to_host của ảnh LỌT  : mean={ls.mean():.3f} min={ls.min():.3f} max={ls.max():.3f}")
    if len(cs):
        print(f"  sim_to_host của ảnh BỊ LOẠI: mean={cs.mean():.3f} min={cs.min():.3f} max={cs.max():.3f}")
    print("=" * 64)


def print_sweep(manifest, emb, eps_list, min_samples):
    print(f"\n{'eps':>6} | {'ret_F':>6} {'ret_P':>6} | {'contam':>7} {'cont_F':>6} {'cont_P':>6} "
          f"| {'hosts_dirty':>11} | {'own_drop':>8}")
    print("-" * 74)
    for eps in eps_list:
        agg, _ = aggregate(manifest, emb, eps, min_samples)
        t = agg["tot"]
        leak = t["leak_F"] + t["leak_P"]
        inj = t["inj_F"] + t["inj_P"]
        own = t["own_F"] + t["own_P"]
        print(f"{eps:6.3f} | {pct(t['kept_F'],t['own_F']):>6} {pct(t['kept_P'],t['own_P']):>6} | "
              f"{pct(leak,inj):>7} {pct(t['leak_F'],t['inj_F']):>6} {pct(t['leak_P'],t['inj_P']):>6} | "
              f"{pct(agg['hosts_with_contam'],agg['n_hosts']):>11} | {pct(t['own_drop'],own):>8}")
    print("-" * 74)
    print("ret_*=giữ ảnh thật (cao tốt) | contam=ảnh người khác lọt (thấp tốt) "
          "=> chọn eps lớn nhất mà contam≈0")


def parse_args():
    ap = argparse.ArgumentParser(description="DBSCAN per-host trên CFP contaminated")
    ap.add_argument("--manifest", default="contaminated_cfp_embeddings.parquet")
    ap.add_argument("--eps", type=float, default=0.44, help="cosine distance = 1 - similarity")
    ap.add_argument("--eps-sweep", type=str, default="",
                    help="danh sách eps phẩy, vd '0.3,0.4,0.44,0.5' -> chỉ in bảng trade-off")
    ap.add_argument("--min-samples", type=int, default=4)
    ap.add_argument("--output", type=str, default="", help="ghi per-row assignment ra parquet")
    return ap.parse_args()


def main():
    args = parse_args()
    df, emb = load_manifest(args.manifest)
    print(f"[load] {len(df):,} dòng | {df['assigned_id'].n_unique()} host | "
          f"tiêm={int(df['is_injected'].sum()):,}")

    if args.eps_sweep:
        eps_list = [float(x) for x in args.eps_sweep.split(",")]
        print_sweep(df, emb, eps_list, args.min_samples)
        return

    agg, per_row = aggregate(df, emb, args.eps, args.min_samples)
    print_detail(agg, args.eps, args.min_samples)

    if args.output:
        out = pl.DataFrame(per_row, schema=["assigned_id", "src_id", "image_type", "seq_no",
                                            "is_injected", "sim_to_host", "cluster",
                                            "in_main", "rel_path"], orient="row")
        out.write_parquet(args.output)
        print(f"\nper-row assignment -> {args.output}")


if __name__ == "__main__":
    main()

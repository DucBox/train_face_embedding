#!/usr/bin/env python
"""
IJB-C worst-case (per-template) 1:1 benchmark.

Hai chức năng:
  1. Build & đóng băng bộ ~46k cặp "khó nhất mỗi template" xuống disk (skip nếu đã có).
     - Với mỗi template: lấy cặp genuine có similarity THẤP nhất (dễ bị từ chối nhầm)
       và cặp impostor có similarity CAO nhất (dễ bị chấp nhận nhầm).
     - Lưu: scores.npy, labels.npy, manifest_46k.csv
  2. Nhận vào 1 ngưỡng -> in FAR/FRR trên bộ đã đóng băng.
     (tùy chọn: 2 ngưỡng vùng xám --t-lo/--t-hi cho sơ đồ reject-option)

Nguồn để build (output của protocol template-pair gốc insightface):
  --pred   : .npy điểm cosine từng cặp template          (N,)
  --label  : .npy nhãn 1=genuine / 0=impostor            (N,)
  --p1p2   : .pkl dict {'p1':..., 'p2':...} ID 2 template (N,) mỗi cái

Ví dụ:
  # build lần đầu rồi tính FAR/FRR tại 1 ngưỡng
  python ijbc_worstcase_far_frr.py --save-dir ijbc_worstcase_46k \
      --pred IJBC_result/vitl_depth36.npy --label IJBC_result/label.npy \
      --p1p2 IJBC_result/p1_p2.pkl --threshold 0.56

  # lần sau (đã có sẵn) chỉ cần đưa ngưỡng, không cần --pred/--label/--p1p2
  python ijbc_worstcase_far_frr.py --save-dir ijbc_worstcase_46k --threshold 0.432

  # vùng xám 2 ngưỡng
  python ijbc_worstcase_far_frr.py --save-dir ijbc_worstcase_46k --t-lo 0.432 --t-hi 0.56
"""
import os
import pickle
import argparse

import numpy as np


def build_frozen_set(pred_path, label_path, p1p2_path, save_dir):
    """Dựng bộ worst-case per-template và lưu xuống save_dir."""
    print(f"[build] loading {pred_path}")
    pred = np.load(pred_path)
    label = np.load(label_path)
    with open(p1p2_path, "rb") as f:
        pp = pickle.load(f)
    p1 = np.asarray(pp["p1"]).astype(np.int64)
    p2 = np.asarray(pp["p2"]).astype(np.int64)
    assert len(pred) == len(label) == len(p1) == len(p2), "độ dài pred/label/p1/p2 lệch"
    print(f"[build] {len(pred):,} template pairs")

    import pandas as pd  # chỉ cần khi build

    # Mỗi cặp đóng góp cho CẢ hai template -> nhân đôi (template, other, score, label)
    df = pd.DataFrame({
        "template": np.concatenate([p1, p2]),
        "other":    np.concatenate([p2, p1]),
        "score":    np.concatenate([pred, pred]).astype(np.float64),
        "label":    np.concatenate([label, label]).astype(np.int8),
    })

    gen = df[df.label == 1]
    imp = df[df.label == 0]
    # genuine khó nhất = score thấp nhất / impostor khó nhất = score cao nhất, mỗi template
    gmin = gen.loc[gen.groupby("template")["score"].idxmin()]
    imax = imp.loc[imp.groupby("template")["score"].idxmax()]
    print(f"[build] {len(gmin):,} genuine-hardest + {len(imax):,} impostor-hardest")

    manifest = pd.concat([gmin, imax], ignore_index=True)
    manifest = manifest.rename(columns={"other": "template_other", "score": "score_old"})
    manifest = manifest[["template", "template_other", "label", "score_old"]]

    scores = manifest["score_old"].to_numpy(dtype=np.float32)
    labels = manifest["label"].to_numpy(dtype=np.int8)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "scores.npy"), scores)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    manifest.to_csv(os.path.join(save_dir, "manifest_46k.csv"), index=False)
    print(f"[build] saved -> {save_dir}/ (scores.npy, labels.npy, manifest_46k.csv)  total={len(scores):,}")
    return scores, labels


def load_frozen_set(save_dir):
    scores = np.load(os.path.join(save_dir, "scores.npy"))
    labels = np.load(os.path.join(save_dir, "labels.npy"))
    return scores, labels


def far_frr_at(threshold, scores, labels):
    """FAR/FRR cho 1 ngưỡng. accept khi score >= t, reject khi score < t."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    gen, imp = s[y == 1], s[y == 0]
    far = float((imp >= threshold).mean())  # impostor bị chấp nhận nhầm
    frr = float((gen < threshold).mean())   # genuine bị từ chối nhầm
    print(f"  thr={threshold:.4f}  ->  FAR={far*100:.4f}%   FRR={frr*100:.4f}%"
          f"   (#gen={len(gen):,}, #imp={len(imp):,})")
    return far, frr


def gray_zone(t_lo, t_hi, scores, labels):
    """Sơ đồ 2 ngưỡng: >=t_hi accept, <=t_lo reject, ở giữa -> review tay."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    gen, imp = s[y == 1], s[y == 0]
    P, N = len(gen), len(imp)
    fa = int((imp >= t_hi).sum())                       # impostor auto-accept (lỗi)
    fr = int((gen <= t_lo).sum())                       # genuine auto-reject (lỗi)
    gz_g = int(((gen > t_lo) & (gen < t_hi)).sum())     # genuine vào vùng xám
    gz_i = int(((imp > t_lo) & (imp < t_hi)).sum())     # impostor vào vùng xám
    print(f"  T_lo={t_lo:.4f}  T_hi={t_hi:.4f}")
    print(f"    FAR_auto (impostor auto-accept) = {fa/N*100:.4f}%")
    print(f"    FRR_auto (genuine  auto-reject) = {fr/P*100:.4f}%")
    print(f"    Review genuine  = {gz_g/P*100:.2f}%")
    print(f"    Review impostor = {gz_i/N*100:.2f}%")
    print(f"    Review TONG     = {(gz_g+gz_i)/(P+N)*100:.2f}%  ({gz_g+gz_i:,} cap)")


def full_table(scores, labels,
               far_targets=(1e-1, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6),
               frr_targets=(1e-1, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4)):
    """In 2 bảng như notebook, nhưng chọn ngưỡng theo ĐÚNG ràng buộc (không 'gần nhất').

    accept khi score >= threshold. Khi hạ ngưỡng: FAR tăng, FRR giảm (đơn điệu).
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    order = np.argsort(-s)                 # ngưỡng cao -> thấp
    ss, yy = s[order], y[order]
    P, N = int((y == 1).sum()), int((y == 0).sum())
    TP = np.cumsum(yy == 1)
    FP = np.cumsum(yy == 0)
    FAR = FP / N                           # tại ngưỡng = ss[i]
    FRR = (P - TP) / P

    print("\n" + "=" * 58)
    print("  FRR @ FAR cố định  (nới ngưỡng tới khi FAR vừa <= target)")
    print("=" * 58)
    print(f"{'Target FAR':>12} | {'Actual FAR':>11} | {'FRR':>9} | {'Threshold':>10}")
    print("-" * 58)
    for tgt in far_targets:
        ok = np.where(FAR <= tgt)[0]       # FAR tăng theo i -> giữ i lớn nhất (FRR nhỏ nhất)
        i = ok[-1] if len(ok) else 0
        print(f"{tgt:12.1e} | {FAR[i]*100:10.4f}% | {FRR[i]*100:8.4f}% | {ss[i]:10.4f}")

    print("\n" + "=" * 58)
    print("  FAR @ FRR cố định  (siết ngưỡng tới khi FRR vừa <= target)")
    print("=" * 58)
    print(f"{'Target FRR':>12} | {'Actual FRR':>11} | {'FAR':>9} | {'Threshold':>10}")
    print("-" * 58)
    for tgt in frr_targets:
        ok = np.where(FRR <= tgt)[0]       # FRR giảm theo i -> giữ i nhỏ nhất (FAR nhỏ nhất)
        i = ok[0] if len(ok) else len(ss) - 1
        print(f"{tgt:12.1e} | {FRR[i]*100:10.4f}% | {FAR[i]*100:8.4f}% | {ss[i]:10.4f}")
    print("=" * 58 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="IJB-C worst-case 46k FAR/FRR")
    ap.add_argument("--save-dir", default="ijbc_worstcase_46k",
                    help="thư mục lưu/đọc bộ đóng băng")
    ap.add_argument("--pred", default="", help=".npy điểm cosine từng cặp template (chỉ cần khi build)")
    ap.add_argument("--label", default="", help=".npy nhãn 1/0 (chỉ cần khi build)")
    ap.add_argument("--p1p2", default="", help=".pkl dict p1/p2 template id (chỉ cần khi build)")
    ap.add_argument("--rebuild", action="store_true", help="build lại dù đã tồn tại")
    ap.add_argument("--threshold", type=float, default=None, help="1 ngưỡng -> FAR/FRR")
    ap.add_argument("--t-lo", type=float, default=None, help="ngưỡng dưới vùng xám")
    ap.add_argument("--t-hi", type=float, default=None, help="ngưỡng trên vùng xám")
    ap.add_argument("--table", action="store_true", help="in 2 bảng FAR@FRR / FRR@FAR như notebook")
    return ap.parse_args()


def main():
    args = parse_args()
    scores_path = os.path.join(args.save_dir, "scores.npy")
    labels_path = os.path.join(args.save_dir, "labels.npy")
    exists = os.path.exists(scores_path) and os.path.exists(labels_path)

    if exists and not args.rebuild:
        print(f"[skip build] đã có {args.save_dir}/ -> load lại")
        scores, labels = load_frozen_set(args.save_dir)
    else:
        if not (args.pred and args.label and args.p1p2):
            raise SystemExit(
                "Chưa có bộ đóng băng. Cần --pred --label --p1p2 để build lần đầu "
                "(hoặc dùng --rebuild)."
            )
        scores, labels = build_frozen_set(args.pred, args.label, args.p1p2, args.save_dir)

    print(f"[set] total={len(scores):,}  genuine={(labels==1).sum():,}  impostor={(labels==0).sum():,}")

    did_eval = False
    if args.table:
        full_table(scores, labels)
        did_eval = True
    if args.threshold is not None:
        print("[FAR/FRR @ 1 ngưỡng]")
        far_frr_at(args.threshold, scores, labels)
        did_eval = True
    if args.t_lo is not None and args.t_hi is not None:
        print("[Vùng xám 2 ngưỡng]")
        gray_zone(args.t_lo, args.t_hi, scores, labels)
        did_eval = True
    if not did_eval:
        print("(không truyền --table/--threshold/--t-lo+--t-hi nên chỉ build/load)")


if __name__ == "__main__":
    main()

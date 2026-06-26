#!/usr/bin/env python
"""
Sinh điểm số (pred) cho bộ IJB-C custom 15M-cặp, cho MỘT model.

Tái dùng đúng tiền xử lý của eval_ijbc.py (align 5-điểm -> 112x112, flip-test,
nhân detector score), nhưng đọc các file META CUSTOM của bạn:
  - image_template_media.csv : img_path,template,media          (ảnh -> template, media)
  - template_label.csv       : identity1,template1,identity2,template2,label   (15M cặp)
  - ijbc_name_5pts_score.txt  : "<name> x1 y1 ... x5 y5 score"   (landmark để align)

Pipeline: embed loose_crop -> pool ảnh/media/template -> chấm các cặp template.

Output (đưa thẳng vào ijbc_worstcase_far_frr.py):
  <out-dir>/pred.npy     (float32, 1 điểm cosine / cặp)
  <out-dir>/label.npy    (int8, 1=genuine / 0=impostor)
  <out-dir>/p1_p2.pkl    ({'p1':..., 'p2':...} template id mỗi cặp)

Ví dụ:
  python build_ijbc_scores.py \
      --model /path/model_moi.pt --network vit_l \
      --image-dir /data/.../IJBC/loose_crop \
      --landmark-file /data/.../meta/ijbc_name_5pts_score.txt \
      --itm /data/.../image_template_media.csv \
      --pairs /data/.../template_label.csv \
      --out-dir IJBC_result_v2 --batch-size 256

  # nếu ảnh ĐÃ align sẵn 112x112 (không có landmark) thì thêm --no-align
  # khi đó không cần --landmark-file, det-score = 1.0
"""
import os
import pickle
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import transform as trans

from backbones import get_model

try:
    from tqdm import tqdm
except ImportError:  # fallback nếu chưa cài tqdm
    def tqdm(it, **kw):
        return it


# Đích align chuẩn ArcFace (giống eval_ijbc.py)
_SRC = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)
_SRC[:, 0] += 8.0


class TorchBackend:
    """Chạy model PyTorch trên GPU + autocast fp16 (y hệt flow cũ)."""
    name = "torch"

    def __init__(self, model_path, network, fp16=True):
        print(f"[model] torch: loading {network} <- {model_path}")
        net = get_model(network, dropout=0, fp16=False)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        self.net = net.cuda().eval()
        self.fp16 = fp16

    @torch.no_grad()
    def infer(self, blob):
        """blob: torch.Tensor (M,3,112,112) raw [0,255] -> feat (M,512) numpy."""
        t = blob.cuda(non_blocking=True)
        t.div_(255).sub_(0.5).div_(0.5)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            feat = self.net(t)
        return feat.float().cpu().numpy()


class OnnxBackend:
    """Chạy model ONNX qua onnxruntime. Tiền xử lý GIỐNG torch: (x/255-0.5)/0.5, RGB CHW."""
    name = "onnx"

    def __init__(self, onnx_path, cpu=False):
        import onnxruntime as ort
        providers = (["CPUExecutionProvider"] if cpu
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"[model] onnx: loading {onnx_path}  providers={providers}")
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[model] onnx in={self.in_name}{self.sess.get_inputs()[0].shape} "
              f"out={self.out_name} actual_providers={self.sess.get_providers()}")

    def infer(self, blob):
        """blob: torch.Tensor/np (M,3,112,112) raw [0,255] -> feat (M,512) numpy."""
        x = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
        x = ((x.astype(np.float32) / 255.0) - 0.5) / 0.5
        return self.sess.run([self.out_name], {self.in_name: x})[0]


def align_112(img, lmk5):
    tform = trans.SimilarityTransform()
    tform.estimate(lmk5, _SRC)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


def preprocess(img, lmk5, no_align, flip):
    """Trả về list các view CHW RGB (1 nếu không flip, 2 nếu có flip)."""
    if no_align:
        face = cv2.resize(img, (112, 112))
    else:
        face = align_112(img, lmk5)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    views = [np.transpose(face, (2, 0, 1))]
    if flip:
        views.append(np.transpose(np.fliplr(face), (2, 0, 1)))
    return views


def read_landmarks(path):
    """name -> (lmk5 [5,2], det_score)."""
    info = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ')
            name = parts[0]
            lmk = np.array([float(x) for x in parts[1:11]], dtype=np.float32).reshape(5, 2)
            score = float(parts[11]) if len(parts) > 11 else 1.0
            info[name] = (lmk, score)
    return info


class _ImgDataset(torch.utils.data.Dataset):
    """Đọc + align ảnh trong worker (song song nhiều core)."""

    def __init__(self, image_dir, names, lmk_info, no_align, flip, use_det_score):
        self.image_dir = image_dir
        self.names = names
        self.lmk_info = lmk_info
        self.no_align = no_align
        self.flip = flip
        self.use_det_score = use_det_score

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        nm = self.names[i]
        img = cv2.imread(os.path.join(self.image_dir, nm))
        if img is None:
            return nm, None, 1.0
        if self.no_align:
            lmk, score = None, 1.0
        else:
            if nm not in self.lmk_info:
                return nm, None, 1.0
            lmk, score = self.lmk_info[nm]
        views = preprocess(img, lmk, self.no_align, self.flip)      # list (3,112,112) uint8
        blob = np.stack(views).astype(np.float32)                  # (n_view,3,112,112)
        return nm, blob, (score if self.use_det_score else 1.0)


def _collate(batch):
    batch = [b for b in batch if b[1] is not None]                 # bỏ ảnh hỏng/thiếu landmark
    if not batch:
        return [], None, []
    names = [b[0] for b in batch]
    scores = [b[2] for b in batch]
    blob = np.concatenate([b[1] for b in batch], 0)                # (n_view*B,3,112,112)
    return names, torch.from_numpy(blob), scores


def embed_all(backend, image_dir, names, lmk_info, no_align, batch_size, flip,
              use_det_score, workers=8):
    """Embed list ảnh -> dict name -> feat512. DataLoader đa worker; forward qua backend (torch/onnx)."""
    name2feat = {}
    n_view = 2 if flip else 1
    ds = _ImgDataset(image_dir, names, lmk_info, no_align, flip, use_det_score)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=workers, collate_fn=_collate,
        pin_memory=True, persistent_workers=workers > 0)

    for nms, blob, scores in tqdm(loader, desc="embed", unit="batch"):
        if blob is None:
            continue
        feat = backend.infer(blob)                                  # (n_view*B, 512) numpy
        feat = feat.reshape(len(nms), n_view, -1).sum(axis=1)       # gộp orig+flip
        for nm, sc, fe in zip(nms, scores, feat):
            name2feat[nm] = fe * sc
    print(f"[embed] xong {len(name2feat)} ảnh"
          + (f" ({len(names)-len(name2feat)} ảnh hỏng/thiếu landmark bị bỏ)"
             if len(name2feat) != len(names) else ""))
    return name2feat


def image2template_feature(feats, templates, medias):
    """Pool: trung bình theo media -> cộng các media -> chuẩn hoá L2 (giống eval_ijbc)."""
    uniq_t = np.unique(templates)
    out = np.zeros((len(uniq_t), feats.shape[1]), dtype=np.float32)
    for ci, t in enumerate(tqdm(uniq_t, desc="pool", unit="tmpl")):
        (idx,) = np.where(templates == t)
        fts = feats[idx]
        med = medias[idx]
        um, uc = np.unique(med, return_counts=True)
        chunks = []
        for u, c in zip(um, uc):
            (im,) = np.where(med == u)
            chunks.append(fts[im] if c == 1 else fts[im].mean(0, keepdims=True))
        out[ci] = np.concatenate(chunks, 0).sum(0)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return out / norm, uniq_t


def score_pairs(tfeat, uniq_t, p1, p2, batch=200000):
    t2id = {int(t): i for i, t in enumerate(uniq_t)}
    id1 = np.array([t2id[int(x)] for x in p1])
    id2 = np.array([t2id[int(x)] for x in p2])
    score = np.empty(len(p1), dtype=np.float32)
    for s in tqdm(range(0, len(p1), batch), desc="score", unit="batch"):
        e = min(s + batch, len(p1))
        score[s:e] = np.sum(tfeat[id1[s:e]] * tfeat[id2[s:e]], axis=1)
    return score


def parse_args():
    ap = argparse.ArgumentParser(description="Sinh pred/label/p1_p2 cho IJB-C custom")
    ap.add_argument("--model", default="", help="checkpoint PyTorch (.pt) - dùng nếu không có --onnx")
    ap.add_argument("--network", default="vit_l", help="kiến trúc cho backend torch")
    ap.add_argument("--onnx", default="", help="file .onnx - bật backend ONNX thay cho torch")
    ap.add_argument("--onnx-cpu", action="store_true", help="ép ONNX chạy CPU (mặc định ưu tiên CUDA)")
    ap.add_argument("--image-dir", required=True, help="thư mục loose_crop")
    ap.add_argument("--landmark-file", default="", help="ijbc_name_5pts_score.txt (cần nếu align)")
    ap.add_argument("--itm", required=True, help="image_template_media.csv")
    ap.add_argument("--pairs", required=True, help="template_label.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8, help="số worker đọc/align ảnh song song")
    ap.add_argument("--no-fp16", action="store_true", help="tắt fp16 autocast (mặc định BẬT)")
    ap.add_argument("--no-align", action="store_true",
                    help="ảnh đã 112x112 căn sẵn -> chỉ resize, không dùng landmark")
    ap.add_argument("--no-flip", action="store_true",
                    help="tắt flip-test (mặc định BẬT: cộng feature ảnh gốc + ảnh lật)")
    ap.add_argument("--no-det-score", action="store_true",
                    help="tắt nhân detector score (mặc định BẬT, theo eval_ijbc.py)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- meta ---
    itm = pd.read_csv(args.itm)                       # img_path, template, media
    pairs = pd.read_csv(args.pairs)                   # ..., template1, template2, label
    p1 = pairs["template1"].to_numpy()
    p2 = pairs["template2"].to_numpy()
    label = pairs["label"].to_numpy().astype(np.int8)
    print(f"[meta] {len(itm):,} ảnh-template | {len(pairs):,} cặp")

    lmk_info = {} if args.no_align else read_landmarks(args.landmark_file)
    if not args.no_align and not args.landmark_file:
        raise SystemExit("Cần --landmark-file để align (hoặc --no-align nếu ảnh đã 112x112).")

    # --- chọn backend: onnx nếu có --onnx, ngược lại torch ---
    if args.onnx:
        backend = OnnxBackend(args.onnx, cpu=args.onnx_cpu)
    elif args.model:
        torch.backends.cudnn.benchmark = True
        backend = TorchBackend(args.model, args.network, fp16=not args.no_fp16)
    else:
        raise SystemExit("Cần --model (PyTorch) hoặc --onnx.")

    # --- embed (theo đúng danh sách ảnh trong itm) ---
    names = itm["img_path"].astype(str).tolist()
    print(f"[cfg] backend={backend.name}  align={not args.no_align}  flip_test={not args.no_flip}  "
          f"det_score={not args.no_det_score}  workers={args.workers}")
    name2feat = embed_all(backend, args.image_dir, names, lmk_info, args.no_align,
                          args.batch_size, flip=not args.no_flip,
                          use_det_score=not args.no_det_score, workers=args.workers)

    # --- gom feats/templates/medias theo thứ tự itm (bỏ ảnh thiếu) ---
    feats, templates, medias = [], [], []
    miss = 0
    for nm, t, m in zip(itm["img_path"].astype(str), itm["template"], itm["media"]):
        fe = name2feat.get(nm)
        if fe is None:
            miss += 1
            continue
        feats.append(fe); templates.append(int(t)); medias.append(int(m))
    if miss:
        print(f"[warn] {miss} ảnh thiếu feature, đã bỏ qua")
    feats = np.asarray(feats, dtype=np.float32)
    templates = np.asarray(templates)
    medias = np.asarray(medias)

    # --- pool -> template features -> chấm cặp ---
    tfeat, uniq_t = image2template_feature(feats, templates, medias)
    print(f"[pool] {len(uniq_t):,} template")
    score = score_pairs(tfeat, uniq_t, p1, p2)

    # --- lưu ---
    np.save(os.path.join(args.out_dir, "pred.npy"), score.astype(np.float32))
    np.save(os.path.join(args.out_dir, "label.npy"), label)
    with open(os.path.join(args.out_dir, "p1_p2.pkl"), "wb") as f:
        pickle.dump({"p1": p1, "p2": p2}, f)
    print(f"[done] -> {args.out_dir}/ (pred.npy, label.npy, p1_p2.pkl)  pairs={len(score):,}")
    print("Tiếp: python ijbc_worstcase_far_frr.py --save-dir ijbc_worstcase_46k_v2 "
          f"--pred {args.out_dir}/pred.npy --label {args.out_dir}/label.npy "
          f"--p1p2 {args.out_dir}/p1_p2.pkl --rebuild --threshold 0.5")


if __name__ == "__main__":
    main()

"""
VERIFY the synthetic-rec layout assumption used by write_rec.py, AND count the
synthetic images so you can compare against your ground-truth number.

Claim to check (your description): `train.rec` is PURE webface, and
`train_synthetic.rec` == the same pure records FOLLOWED BY the synthetic images
appended at the end. write_rec.py relies on this to pick synthetic-only records
as those with key > max(pure key), keeping their parent webface label.

Pass the two recs explicitly (a prefix under CFG.rec_root, or a full .rec path):

    python verify_synthetic_layout.py \
        --pure /data/train.rec --synthetic /data/train_synthetic.rec \
        --sample 5000 --expected-synthetic 6000000

What it checks / prints:
  1. record counts of both recs (banner).
  2. prefix check: a random sample of pure keys must match train_synthetic at the
     same key in BOTH label and image bytes -> proves the pure prefix is intact.
  3. SYNTHETIC COUNT = records with key > max(pure key); compared to
     --expected-synthetic if given (PASS/FAIL).
  4. sample of synthetic-only parent labels (should be existing webface ids).
"""
from __future__ import annotations

import argparse
import numbers
import os
import random

from tqdm import tqdm

from config import CFG, WEBFACE


def _resolve(arg, default_prefix):
    """Accept a full .rec path or a bare prefix under CFG.rec_root."""
    if arg is None:
        arg = default_prefix
    if arg.endswith(".rec") or os.path.sep in arg:
        rec = arg if arg.endswith(".rec") else arg + ".rec"
        return rec, rec[:-4] + ".idx", os.path.basename(rec)[:-4]
    return (os.path.join(CFG.rec_root, f"{arg}.rec"),
            os.path.join(CFG.rec_root, f"{arg}.idx"), arg)


def _open(rec_path, idx_path):
    import mxnet as mx
    if not (os.path.exists(rec_path) and os.path.exists(idx_path)):
        raise FileNotFoundError(f"missing {rec_path} / {idx_path}")
    return mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")


def _label(header):
    lb = header.label
    return int(lb if isinstance(lb, numbers.Number) else lb[0])


def main():
    import mxnet as mx
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", default=None, help="pure webface rec (prefix or .rec path)")
    ap.add_argument("--synthetic", default=None, help="synthetic rec (prefix or .rec path)")
    ap.add_argument("--sample", type=int, default=5000, help="pure keys to byte-compare")
    ap.add_argument("--expected-synthetic", type=int, default=None,
                    help="your ground-truth synthetic image count to assert against")
    args = ap.parse_args()

    p_rec, p_idx, p_name = _resolve(args.pure, CFG.rec_prefix[WEBFACE][0])
    s_rec, s_idx, s_name = _resolve(args.synthetic, CFG.synthetic_prefix)
    print(f"[paths] pure      = {p_rec}")
    print(f"[paths] synthetic = {s_rec}\n")

    pure, syn = _open(p_rec, p_idx), _open(s_rec, s_idx)
    # IMPORTANT: insightface .rec keys include per-identity HEADER records at the
    # end (one per class), not just images. So len(keys) = images + num_classes + 1.
    # The real IMAGE boundary is header.label[0]: images live at indices 1..label[0]-1.
    hp0, _ = mx.recordio.unpack(pure.read_idx(0))
    hs0, _ = mx.recordio.unpack(syn.read_idx(0))
    assert hp0.flag > 0 and hs0.flag > 0, "expected header-format recs (flag>0)"
    Hp, Hs = int(hp0.label[0]), int(hs0.label[0])   # image boundary (images = 1..H-1)
    Np, Ns = Hp - 1, Hs - 1                          # actual image counts

    print("=" * 60)
    print(f"[counts] {p_name}: images={Np:,}  (raw idx records={len(list(pure.keys)):,} "
          f"= images + per-id headers)")
    print(f"[counts] {s_name}: images={Ns:,}  (raw idx records={len(list(syn.keys)):,})")
    print("=" * 60)
    assert Ns >= Np, "synthetic has fewer images than pure (!)"

    # step 2: prefix integrity (label + bytes) over IMAGE indices 1..Np
    sample = random.sample(range(1, Hp), min(args.sample, Np))
    mismatch_label = mismatch_bytes = 0
    for k in tqdm(sample, desc="prefix-check (label+bytes)", unit="img"):
        hp, ip = mx.recordio.unpack(pure.read_idx(k))
        hs, is_ = mx.recordio.unpack(syn.read_idx(k))
        mismatch_label += _label(hp) != _label(hs)
        mismatch_bytes += ip != is_
    print(f"[prefix-check] sampled {len(sample):,} image idxs: "
          f"label_mismatch={mismatch_label} bytes_mismatch={mismatch_bytes}")
    assert mismatch_label == 0 and mismatch_bytes == 0, \
        "train_synthetic is NOT (pure + appended synthetic) — revisit write_rec.syn_keys"

    # step 3: synthetic count = images beyond the pure boundary (idx Hp .. Hs-1)
    n_syn = Ns - Np
    print("\n" + "#" * 60)
    print(f"#  SYNTHETIC IMAGE COUNT = {n_syn:,}   (image idxs {Hp}..{Hs - 1})")
    if args.expected_synthetic is not None:
        ok = n_syn == args.expected_synthetic
        print(f"#  ground-truth expected = {args.expected_synthetic:,}  -> "
              f"{'PASS ✓' if ok else 'MISMATCH ✗'}")
    print("#" * 60 + "\n")

    # step 4: sample synthetic parent labels (should be existing webface ids)
    syn_idxs = range(Hp, Hs)
    tail_sample = random.sample(list(syn_idxs), min(20, n_syn)) if n_syn else []
    labels = sorted({_label(mx.recordio.unpack(syn.read_idx(k))[0]) for k in tail_sample})
    print(f"[tail] sample parent labels: {labels}")
    print(f"\n[RESULT] OK — pure prefix intact, {n_syn:,} synthetic at image idxs "
          f"[{Hp}, {Hs}). write_rec synthetic = read_idx over range({Hp}, {Hs}).")


if __name__ == "__main__":
    main()

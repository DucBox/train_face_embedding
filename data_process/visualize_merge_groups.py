"""
Visualize the merge groups produced by merge_id_ivf_stats.py.

Each row of `merge_groups.csv` (columns: leader_id, members, n_members) is one
set of person IDs that the IVF stats decided to merge into a single identity.
This tool pulls real images for each member ID straight out of the training
`.rec`/`.idx` files and lays them out as ONE image per group so you can eyeball
whether the merged IDs are actually the same person.

Layout of a group image (top -> bottom):

    ID 12345  (20 imgs)
    [img][img][img][img][img]
    [img][img][img][img][img]
    ...
                                <- gap
    ID 67890  (18 imgs)
    [img][img][img][img][img]
    ...

`cols` images per row (default 5), up to `per_id` images per ID (default 20),
IDs stacked with a gap + a text label between them.

Usage:
    python visualize_merge_groups.py \
        --groups-csv .../merge_clean/merge_groups.csv \
        --root-dir   /workspace/FaceNist/Data \
        --out-dir    .../merge_clean/viz

Header detection is automatic per .rec file (same rule as embed_dataset.py):
train_synthetic / train_public carry a RecordIO header (flag > 0); the crawl
shards (train_1, train_2, ...) do not, so their indices are enumerated from
`imgrec.keys` instead.
"""
import argparse
import numbers
import os
from collections import defaultdict

import mxnet as mx
import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def discover_prefixes(root_dir):
    """Return the .rec file prefixes present under root_dir, in load order:
    train_synthetic, train_public, then train_1, train_2, ... (crawl shards)."""
    prefixes = []
    for p in ("train_synthetic", "train_public", "train"):
        if os.path.exists(os.path.join(root_dir, f"{p}.rec")):
            prefixes.append(p)
    i = 1
    while os.path.exists(os.path.join(root_dir, f"train_{i}.rec")):
        prefixes.append(f"train_{i}")
        i += 1
    return prefixes


def open_rec(root_dir, prefix):
    path_imgrec = os.path.join(root_dir, f"{prefix}.rec")
    path_imgidx = os.path.join(root_dir, f"{prefix}.idx")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        # train_synthetic / train_public: dense range from the header.
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        # crawl shards: no usable header, enumerate the real keys.
        imgidx = np.array(list(imgrec.keys))
    return imgrec, imgidx


def label_of(header):
    label = header.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    return int(label)


def collect_images(root_dir, prefixes, wanted_ids, per_id, thumb):
    """Scan the .rec files once each and grab up to `per_id` images for every ID
    in `wanted_ids`. Returns dict[id] -> list of (thumb x thumb x 3) uint8 RGB.

    Stops scanning a file early once every still-needed ID is full (cheap when
    images of one ID are contiguous, which they usually are in these recs)."""
    images = defaultdict(list)
    wanted = set(int(i) for i in wanted_ids)
    # IDs that still need more images; drop an ID once it reaches per_id.
    needed = set(wanted)

    for prefix in prefixes:
        if not needed:
            break
        imgrec, imgidx = open_rec(root_dir, prefix)
        pbar = tqdm(imgidx, desc=f"scan {prefix}", unit="img")
        for idx in pbar:
            if not needed:
                break
            idx = int(idx)
            s = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            lbl = label_of(header)
            if lbl not in needed:
                continue
            arr = mx.image.imdecode(img).asnumpy()  # RGB
            im = Image.fromarray(arr).resize((thumb, thumb))
            images[lbl].append(np.asarray(im))
            if len(images[lbl]) >= per_id:
                needed.discard(lbl)
                pbar.set_postfix(remaining=len(needed))

    return images


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_group(member_ids, images, cols, thumb, gap, title_h, pad, font):
    """Compose one image: each member ID = a title line + a grid of its images,
    stacked vertically with `gap` px between IDs."""
    canvas_w = cols * thumb + 2 * pad

    # Pre-compute the height each ID block needs.
    blocks = []  # (mid, imgs, n_rows, block_h)
    total_h = pad
    for mid in member_ids:
        imgs = images.get(int(mid), [])
        n_rows = max(1, (len(imgs) + cols - 1) // cols)
        block_h = title_h + n_rows * thumb
        blocks.append((mid, imgs, n_rows, block_h))
        total_h += block_h + gap
    total_h += pad

    canvas = Image.new("RGB", (canvas_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for mid, imgs, n_rows, block_h in blocks:
        title = f"ID {mid}  ({len(imgs)} imgs)"
        draw.text((pad, y + 4), title, fill=(255, 220, 80), font=font)
        gy = y + title_h
        for k, arr in enumerate(imgs):
            r, c = divmod(k, cols)
            x = pad + c * thumb
            canvas.paste(Image.fromarray(arr), (x, gy + r * thumb))
        # separator line under the block
        sep_y = y + block_h + gap // 2
        draw.line([(pad, sep_y), (canvas_w - pad, sep_y)], fill=(80, 80, 80), width=1)
        y += block_h + gap

    return canvas


def parse_groups(groups_csv):
    """Yield (group_name, [member_id, ...]) from a merge CSV.

    Supports merge_groups.csv (leader_id, members) where `members` is a
    space-separated id list, or merge_map.csv (person_id, merged_into) which is
    grouped by merged_into on the fly."""
    df = pl.read_csv(groups_csv)
    cols = df.columns
    if "members" in cols:
        for row in df.iter_rows(named=True):
            members = [int(x) for x in str(row["members"]).split()]
            leader = row.get("leader_id", members[0])
            yield str(leader), members
    elif "merged_into" in cols and "person_id" in cols:
        grouped = defaultdict(list)
        for row in df.iter_rows(named=True):
            grouped[int(row["merged_into"])].append(int(row["person_id"]))
        for leader, members in grouped.items():
            yield str(leader), [leader] + members
    else:
        raise ValueError(
            f"Unrecognized CSV columns {cols}; expected merge_groups.csv "
            f"(leader_id, members) or merge_map.csv (person_id, merged_into)."
        )


def main():
    ap = argparse.ArgumentParser(description="Visualize merge groups as one image per group.")
    ap.add_argument("--groups-csv", required=True, help="merge_groups.csv (or merge_map.csv)")
    ap.add_argument("--root-dir", required=True, help="folder containing the .rec/.idx files")
    ap.add_argument("--out-dir", required=True, help="where to write the group PNGs")
    ap.add_argument("--prefixes", default=None,
                    help="comma list of .rec prefixes to scan; default = auto-discover")
    ap.add_argument("--per-id", type=int, default=20, help="max images per ID (default 20)")
    ap.add_argument("--cols", type=int, default=5, help="images per row (default 5)")
    ap.add_argument("--thumb", type=int, default=112, help="thumbnail size in px (default 112)")
    ap.add_argument("--gap", type=int, default=28, help="vertical gap between IDs in px")
    ap.add_argument("--limit", type=int, default=None, help="only render first N groups (debug)")
    args = ap.parse_args()

    groups = list(parse_groups(args.groups_csv))
    if args.limit is not None:
        groups = groups[: args.limit]
    print(f"[1] {len(groups)} merge group(s) from {args.groups_csv}")

    wanted_ids = set()
    for _, members in groups:
        wanted_ids.update(members)
    print(f"    -> {len(wanted_ids)} unique IDs to fetch")

    prefixes = args.prefixes.split(",") if args.prefixes else discover_prefixes(args.root_dir)
    assert prefixes, f"No .rec files found under {args.root_dir}"
    print(f"[2] Scanning prefixes: {prefixes}")
    images = collect_images(args.root_dir, prefixes, wanted_ids, args.per_id, args.thumb)

    missing = [i for i in wanted_ids if not images.get(int(i))]
    if missing:
        print(f"    !! {len(missing)} ID(s) found 0 images (not in scanned files): "
              f"{missing[:20]}{' ...' if len(missing) > 20 else ''}")

    os.makedirs(args.out_dir, exist_ok=True)
    font = load_font(max(14, args.thumb // 6))
    title_h = max(20, args.thumb // 4)
    print(f"[3] Rendering -> {args.out_dir}")
    for gi, (name, members) in enumerate(tqdm(groups, desc="render", unit="grp")):
        canvas = render_group(members, images, args.cols, args.thumb, args.gap, title_h, pad=8, font=font)
        canvas.save(os.path.join(args.out_dir, f"group_{gi:05d}_leader{name}.png"))

    print(f"[done] wrote {len(groups)} group image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

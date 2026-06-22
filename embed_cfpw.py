"""
Embed the CFP (Celebrities in Frontal-Profile) dataset for the DBSCAN
clustering test in run_dbscan_cfp.py.

Walks cfp-dataset/Data/Images/{id:03d}/{frontal|profile}/{seq:02d}.jpg directly
off disk (plain JPGs, not MXNet RecordIO) and writes a single Parquet file with
one row per image:
    id_number   int32   - the identity folder (1-500)
    image_type  string  - "frontal" or "profile"
    seq_no      int32   - the image number within that identity+pose (the "01" in 01.jpg)
    embedding   fixed_size_list<float32>[embedding_size] - L2-normalized embedding

    python3 embed_cfpw.py configs/wf42m_pfc03_40epoch_64gpu_vit_l \
        --weight /path/to/model.pt --cfp-dir cfp-dataset/Data/Images --output cfp_embeddings.parquet
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from embed_dataset import DEFAULT_WEIGHT, build_backbone
from utils.utils_config import get_config


class CFPImageDataset(Dataset):
    """Reads cfp-dataset/Data/Images/{id}/{frontal|profile}/{seq}.jpg directly off disk."""

    def __init__(self, root_dir):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*", "*", "*.jpg")))
        assert self.paths, f"No .jpg files found under {root_dir}"
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image_type = os.path.basename(os.path.dirname(path))                          # "frontal" / "profile"
        id_number = int(os.path.basename(os.path.dirname(os.path.dirname(path))))      # "001" -> 1
        seq_no = int(os.path.splitext(os.path.basename(path))[0])                      # "01.jpg" -> 1

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, id_number, image_type, seq_no


DEFAULT_CONFIG = "configs/wf42m_pfc03_40epoch_64gpu_vit_l"
DEFAULT_CFP_DIR = "cfp-dataset/Data/Images"
DEFAULT_OUTPUT = "cfp_embeddings.parquet"


def main():
    parser = argparse.ArgumentParser(description="Embed the CFP dataset for DBSCAN clustering tests")
    parser.add_argument("config", type=str, nargs="?", default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    parser.add_argument("--weight", type=str, default=DEFAULT_WEIGHT, help=f"default: {DEFAULT_WEIGHT}")
    parser.add_argument("--cfp-dir", type=str, default=DEFAULT_CFP_DIR, help=f"default: {DEFAULT_CFP_DIR}")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help=f"default: {DEFAULT_OUTPUT}")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = get_config(args.config)
    device = torch.device(args.device)

    dataset = CFPImageDataset(args.cfp_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    net = build_backbone(cfg)
    net.load_state_dict(torch.load(args.weight, map_location="cpu"))
    net.eval().to(device)

    embeddings, id_numbers, image_types, seq_nos = [], [], [], []
    with torch.no_grad():
        for imgs, ids, types, seqs in tqdm(loader, desc="embed cfp", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            feat = F.normalize(net(imgs), dim=1)
            embeddings.append(feat.cpu().numpy().astype(np.float32))
            id_numbers.append(ids.numpy().astype(np.int32))
            image_types.extend(types)
            seq_nos.append(seqs.numpy().astype(np.int32))

    embeddings = np.concatenate(embeddings, axis=0)
    id_numbers = np.concatenate(id_numbers, axis=0)
    seq_nos = np.concatenate(seq_nos, axis=0)

    table = pa.table({
        "id_number": pa.array(id_numbers, type=pa.int32()),
        "image_type": pa.array(image_types, type=pa.string()),
        "seq_no": pa.array(seq_nos, type=pa.int32()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()), cfg.embedding_size),
    })

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"wrote {len(id_numbers):,} rows ({len(set(id_numbers))} identities) to {args.output}")


if __name__ == "__main__":
    main()

"""
Stage 3/3 of the offline hard-case mining pipeline: turn the two failure
groups from find_hard_thresholds.py into actionable artifacts.

    python3 generate_hard_cases.py --input-dir /path/to/hard_case_out --output-dir /path/to/hard_case_out

Reads:
    false_accept.csv  : class_a,class_b,score,threshold   (impostor identity pairs)
    false_reject.csv  : file_prefix,rec_idx,label,genuine_score,threshold  (genuine images)

Writes:
    hard_class_neighbors.csv : class_id,neighbor_class_id,score
        Each false_accept pair (a,b) is symmetric (a is confusable with b AND
        b is confusable with a), so it is expanded into both directed rows,
        sorted by class_id then score descending. This is the "ground-truth"
        global hard-negative list discussed in docs/hard_negative_sampling.md -
        meant to seed PartialFC_V2's neighbor_cache/confusion_queue for the next
        training run (each rank would filter this down to its own class shard),
        instead of relying only on the FC-weight-based proxy computed online.

    hard_images_review.csv : file_prefix,rec_idx,label,genuine_score
        Same rows as false_reject.csv, sorted worst-score-first - a prioritized
        list for a human to inspect (possible mislabeled / very hard images),
        not meant to be fed back into sampling directly.
"""
import argparse
import csv
import os


def generate_hard_class_neighbors(input_dir, output_dir):
    in_path = os.path.join(input_dir, "false_accept.csv")
    out_path = os.path.join(output_dir, "hard_class_neighbors.csv")

    rows = []  # (class_id, neighbor_class_id, score)
    with open(in_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a, b, score = int(row["class_a"]), int(row["class_b"]), float(row["score"])
            rows.append((a, b, score))
            rows.append((b, a, score))

    rows.sort(key=lambda r: (r[0], -r[2]))

    with open(out_path, "w") as f:
        f.write("class_id,neighbor_class_id,score\n")
        for class_id, neighbor_id, score in rows:
            f.write(f"{class_id},{neighbor_id},{score:.4f}\n")

    print(f"hard_class_neighbors: {len(rows):,} directed rows ({len(rows)//2:,} unique pairs) -> {out_path}")


def generate_hard_images_review(input_dir, output_dir):
    in_path = os.path.join(input_dir, "false_reject.csv")
    out_path = os.path.join(output_dir, "hard_images_review.csv")

    with open(in_path) as f:
        reader = csv.DictReader(f)
        rows = [(row["file_prefix"], int(row["rec_idx"]), int(row["label"]), float(row["genuine_score"]))
                for row in reader]

    rows.sort(key=lambda r: r[3])  # worst (lowest) genuine_score first

    with open(out_path, "w") as f:
        f.write("file_prefix,rec_idx,label,genuine_score\n")
        for file_prefix, rec_idx, label, genuine_score in rows:
            f.write(f"{file_prefix},{rec_idx},{label},{genuine_score:.4f}\n")

    print(f"hard_images_review: {len(rows):,} rows, sorted worst-first -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate hard-case artifacts from find_hard_thresholds.py output")
    parser.add_argument("--input-dir", type=str, required=True,
                         help="output-dir of find_hard_thresholds.py (contains false_accept.csv / false_reject.csv)")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="defaults to --input-dir")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    generate_hard_class_neighbors(args.input_dir, output_dir)
    generate_hard_images_review(args.input_dir, output_dir)


if __name__ == "__main__":
    main()

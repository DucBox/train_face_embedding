"""
Preprocessing stage for the CFP DBSCAN test: detect the face in every raw CFP
image with yolov6n_face.onnx (bbox + 5-point landmarks: left eye, right eye,
nose tip, left mouth corner, right mouth corner), then warp it to a 112x112
crop with the SAME similarity-transform alignment + reference template
eval_ijbc.py uses for real evaluation data. Saves the result to
cfp-dataset/processed_data/ (sibling to cfp-dataset/Data/). embed_cfpw.py then
just loads these - no detection/alignment logic at embed time.

Why not CFP's own 30-point fiducials (see an earlier version of this script,
and docs/offline_hard_case_mining.md history): nothing in the dataset documents
which of the 30 indices is which landmark, and a visual check showed the scheme
deliberately only marks ONE side of the face in detail (so it still applies to
profile shots, which only show one side) - not a clean 5-point eye/nose/mouth
set we could safely map without guessing. Running an actual face detector
sidesteps that entirely: yolov6n_face.onnx's landmark outputs ARE already the
standard 5-point layout, confirmed by inspecting raw output values against a
sample image (see PR/commit notes) - left/right eye y-coordinates close
together, nose below+between them, two mouth corners below that.

    python3 align_cfp.py --cfp-dir cfp-dataset/Data/Images \
        --detector yolov6n_face.onnx --output-dir cfp-dataset/processed_data
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort
from skimage import transform as trans
from tqdm import tqdm

DEFAULT_CFP_DIR = "cfp-dataset/Data/Images"
DEFAULT_DETECTOR = "yolov6n_face.onnx"
DEFAULT_OUTPUT_DIR = "cfp-dataset/processed_data"
DETECTOR_INPUT_SIZE = 640
MIN_CONFIDENCE = 0.3

# Same 5-point ArcFace reference template as eval_ijbc.py's Embedding class -
# aligning to this (not a CFP-specific template) keeps these crops on the exact
# same footing as the real evaluation/training-time face alignment.
REFERENCE_5PT = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)
REFERENCE_5PT[:, 0] += 8.0


class FaceDetector:
    """yolov6n_face.onnx wrapper. Raw output is (1, 8400, 16) - per anchor:
    [cx, cy, w, h, lm1x, lm1y, ..., lm5x, lm5y, obj, conf]. Each CFP image has
    exactly one face, so this just returns the single highest-confidence anchor
    (no NMS needed) instead of the usual multi-face decode+NMS pipeline."""

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, img):
        """Returns (landmarks[5,2] in original image coords, confidence) or
        (None, confidence) if the best detection is below MIN_CONFIDENCE."""
        H0, W0 = img.shape[:2]
        resized = cv2.resize(img, (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE))
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        blob = blob.transpose(2, 0, 1)[None]

        out = self.session.run([self.output_name], {self.input_name: blob})[0][0]  # (8400, 16)
        best = out[np.argmax(out[:, 15])]
        conf = float(best[15])
        if conf < MIN_CONFIDENCE:
            return None, conf

        landmarks = best[4:14].reshape(5, 2)
        scale = np.array([W0 / DETECTOR_INPUT_SIZE, H0 / DETECTOR_INPUT_SIZE], dtype=np.float32)
        return landmarks * scale, conf


def align_image(img, landmarks, image_size=112):
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks.astype(np.float32), REFERENCE_5PT)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0)


def main():
    parser = argparse.ArgumentParser(description="Detect+align CFP faces to 112x112 via yolov6n_face.onnx")
    parser.add_argument("--cfp-dir", type=str, default=DEFAULT_CFP_DIR, help=f"default: {DEFAULT_CFP_DIR}")
    parser.add_argument("--detector", type=str, default=DEFAULT_DETECTOR, help=f"default: {DEFAULT_DETECTOR}")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                         help=f"skip images whose best detection scores below this, default: {MIN_CONFIDENCE}")
    args = parser.parse_args()

    detector = FaceDetector(args.detector)
    image_paths = sorted(glob.glob(os.path.join(args.cfp_dir, "*", "*", "*.jpg")))
    assert image_paths, f"No .jpg files found under {args.cfp_dir}"

    n_done, n_failed = 0, 0
    failed_paths = []
    for img_path in tqdm(image_paths, desc="align", unit="img"):
        image_type = os.path.basename(os.path.dirname(img_path))
        id_str = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        seq_str = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path)
        landmarks, conf = detector.detect(img)
        if landmarks is None:
            n_failed += 1
            failed_paths.append((img_path, conf))
            continue

        aligned = align_image(img, landmarks, image_size=args.image_size)
        out_dir = os.path.join(args.output_dir, id_str, image_type)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{seq_str}.jpg"), aligned)
        n_done += 1

    print(f"Aligned {n_done:,} images -> {args.output_dir} ({n_failed:,} failed: "
          f"best detection below confidence {args.min_confidence})")
    for path, conf in failed_paths[:20]:
        print(f"  FAILED conf={conf:.3f}: {path}")
    if len(failed_paths) > 20:
        print(f"  ... and {len(failed_paths) - 20:,} more")


if __name__ == "__main__":
    main()

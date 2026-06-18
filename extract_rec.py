import os
import numbers
import cv2
import mxnet as mx
import numpy as np
from tqdm import tqdm
import argparse


def extract_mxnet_to_imagefolder(mxnet_path, output_path):
    """
    Extract MXNet RecordIO format to ImageFolder structure
    
    Args:
        mxnet_path: Path to folder containing train.rec and train.idx
        output_path: Output folder for ImageFolder structure
    """
    
    # Setup paths
    path_imgrec = os.path.join(mxnet_path, 'train.rec')
    path_imgidx = os.path.join(mxnet_path, 'train.idx')
    
    if not os.path.exists(path_imgrec) or not os.path.exists(path_imgidx):
        raise FileNotFoundError(f"train.rec or train.idx not found in {mxnet_path}")
    
    print(f"Reading from: {path_imgrec}")
    print(f"Output to: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Open MXNet RecordIO
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    
    # Get header and indices
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
        print(f"Found {len(imgidx)} images, {header0[1]} identities")
    else:
        imgidx = np.array(list(imgrec.keys))
        print(f"Found {len(imgidx)} images")
    
    # Count images per identity for progress tracking
    identity_counts = {}
    
    # First pass: count identities
    print("Counting identities...")
    for i, idx in enumerate(tqdm(imgidx[:1000])):  # Sample first 1000 for quick count
        s = imgrec.read_idx(idx)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        identity_counts[int(label)] = identity_counts.get(int(label), 0) + 1
    
    print(f"Sample shows ~{len(identity_counts)} identities")
    
    # Extract images
    print("Extracting images...")
    identity_image_counts = {}
    
    for i, idx in enumerate(tqdm(imgidx, desc="Processing images")):
        try:
            # Read record
            s = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            
            # Get label
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            identity_id = int(label)
            
            # Create identity folder
            identity_folder = os.path.join(output_path, f"identity_{identity_id:06d}")
            os.makedirs(identity_folder, exist_ok=True)
            
            # Decode image
            img_np = mx.image.imdecode(img).asnumpy()
            
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Generate filename
            if identity_id not in identity_image_counts:
                identity_image_counts[identity_id] = 0
            identity_image_counts[identity_id] += 1
            
            img_filename = f"img_{identity_image_counts[identity_id]:04d}.jpg"
            img_path = os.path.join(identity_folder, img_filename)
            
            # Save image
            cv2.imwrite(img_path, img_bgr)
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    print(f"\nExtraction completed!")
    print(f"Total identities: {len(identity_image_counts)}")
    print(f"Total images: {sum(identity_image_counts.values())}")
    print(f"Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract MXNet RecordIO to ImageFolder format')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input path containing train.rec and train.idx')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output path for ImageFolder structure')
    
    args = parser.parse_args()
    
    extract_mxnet_to_imagefolder(args.input, args.output)


if __name__ == "__main__":
    main()
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

MERGED_ROOT = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/person_id_merged_by_bin"
IMG_SIZE = 112  # Original size
GRID_COLS = 20

def create_merged_image(person_dir, output_path):
    img_paths = []
    for subdir in os.listdir(person_dir):
        sub_path = os.path.join(person_dir, subdir)
        if os.path.isdir(sub_path):
            img_paths.extend(Path(sub_path).glob("*.jpg"))
    
    if not img_paths:
        return False
    
    images = []
    for img_path in sorted(img_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except:
            continue
    
    if not images:
        return False
    
    n_images = len(images)
    n_rows = (n_images + GRID_COLS - 1) // GRID_COLS
    height = n_rows * IMG_SIZE
    width = GRID_COLS * IMG_SIZE
    merged = Image.new('RGB', (width, height), color=(240, 240, 240))
    
    for i, img in enumerate(images):
        row, col = divmod(i, GRID_COLS)
        merged.paste(img, (col * IMG_SIZE, row * IMG_SIZE))
    
    merged.save(output_path, quality=95)
    return True

def main():
    visual_root = "/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/clean_data/visual_v1_ivf_real_debug"
    os.makedirs(MERGED_ROOT, exist_ok=True)
    
    processed = 0
    for bin_folder in os.listdir(visual_root):
        bin_path = os.path.join(visual_root, bin_folder)
        if not os.path.isdir(bin_path):
            continue
            
        # Tạo bin folder trong merged
        bin_merged_dir = os.path.join(MERGED_ROOT, bin_folder)
        os.makedirs(bin_merged_dir, exist_ok=True)
        
        print(f"\n--- {bin_folder} ---")
        for person_folder in tqdm(os.listdir(bin_path)):
            person_path = os.path.join(bin_path, person_folder)
            if not os.path.isdir(person_path):
                continue
            
            person_id = person_folder.replace("person_", "")
            output_path = os.path.join(bin_merged_dir, f"{person_id}_merged.jpg")
            
            if os.path.exists(output_path):
                continue
                
            if create_merged_image(person_path, output_path):
                processed += 1
    
    print(f"\nDONE! {processed} merged images")
    print(f"Structure: {MERGED_ROOT}/bin_xxx/person_yyy_merged.jpg")

if __name__== "__main__":
    main()
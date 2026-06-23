import ast
import io
import os
import boto3
import polars as pl
from PIL import Image, ImageDraw, ImageFont

# --- CONFIG ---
S3_BUCKET = "ttnt"
RAW_PREFIX = "cv/crawled-datasets/face-google-search/"
OUTPUT_DIR = "/workspace/FaceNist/raw_data_processing/visualize_face_bbox_by_count"

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://s3-data.cyberspace.vn",
        aws_access_key_id="ttnt",
        aws_secret_access_key="H?3o0nn4Irej",
    )
def parse_s3_path(s3_path: str):
    """Tách s3_path thành tar_key và member_name bên trong tar."""
    if ".tar/" in s3_path:
        parts = s3_path.split(".tar/", 1)
        tar_key = parts[0] + ".tar"
        member_name = parts[1]
    else:
        tar_key = None
        member_name = s3_path
    return tar_key, member_name

def download_tar_to_memory(s3_client, tar_key: str) -> io.BytesIO:
    """Download file tar từ S3 vào memory."""
    print(f"  📦 Downloading tar: {tar_key}")
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=tar_key)
    return io.BytesIO(response["Body"].read())

def extract_image_from_tar(tar_buffer: io.BytesIO, member_name: str) -> Image.Image:
    """Extract 1 ảnh cụ thể từ tar buffer trong memory."""
    tar_buffer.seek(0)
    with tarfile.open(fileobj=tar_buffer, mode="r:") as tar:
        # Thử tìm member theo tên file (có thể có subfolder trong tar)
        members = tar.getnames()
        matched = next(
            (m for m in members if m.endswith(member_name) or os.path.basename(m) == member_name),
            None
        )
        if matched is None:
            raise FileNotFoundError(f"Member '{member_name}' not found in tar. Available: {members[:5]}...")
        
        f = tar.extractfile(matched)
        return Image.open(io.BytesIO(f.read())).convert("RGB")

def draw_bboxes(image: Image.Image, bboxs_str: str) -> Image.Image:
    """Parse bbox string và vẽ lên ảnh."""
    draw = ImageDraw.Draw(image)
    try:
        bboxs = ast.literal_eval(bboxs_str)
    except Exception as e:
        print(f"  ⚠️  Cannot parse bboxs: {e}")
        return image
    
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        score = bbox[4] if len(bbox) > 4 else None
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if score is not None:
            draw.text((x1, max(0, y1 - 15)), f"{score:.2f}", fill="red")
    
    return image

def sample_by_num_faces(df: pl.DataFrame) -> dict:
    """Filter và sample 3 ảnh cho mỗi nhóm num_faces."""
    groups = {
        "num_faces_1":      df.filter(pl.col("num_faces") == 1).head(3),
        "num_faces_3":      df.filter(pl.col("num_faces") == 3).head(3),
        "num_faces_5":      df.filter(pl.col("num_faces") == 5).head(3),
        "num_faces_10":     df.filter(pl.col("num_faces") == 10).head(3),
        "num_faces_20plus": df.filter(pl.col("num_faces") >= 20).head(3),
    }
    print("📊 Sample counts:")
    for name, sub_df in groups.items():
        print(f"  {name}: {len(sub_df)} ảnh")
    return groups

def main():
    s3_client = get_s3_client()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    groups = sample_by_num_faces(df_faces)
    
    # Cache tar buffers: tar_key → BytesIO (tránh download lại cùng 1 tar)
    tar_cache = {}
    
    for group_name, sub_df in groups.items():
        group_dir = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n📁 Processing group: {group_name}")
        
        for i, row in enumerate(sub_df.iter_rows(named=True)):
            s3_path  = row["s3_path"]
            bboxs_str = row["bboxs"]
            num_faces = row["num_faces"]
            
            tar_key, member_name = parse_s3_path(s3_path)
            print(f"  [{i+1}/{len(sub_df)}] {member_name}  (tar: {os.path.basename(tar_key)})")
            
            try:
                # Download tar 1 lần, cache lại
                if tar_key not in tar_cache:
                    tar_cache[tar_key] = download_tar_to_memory(s3_client, tar_key)
                
                image = extract_image_from_tar(tar_cache[tar_key], member_name)
                image = draw_bboxes(image, bboxs_str)
                
                filename = f"{i+1:02d}_faces{num_faces}_{member_name}"
                save_path = os.path.join(group_dir, filename)
                image.save(save_path)
                print(f"  ✅ Saved: {save_path}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print(f"\n🎉 Done! Images saved to: {OUTPUT_DIR}/")
    print(f"📦 Tar files cached: {len(tar_cache)}")

main()
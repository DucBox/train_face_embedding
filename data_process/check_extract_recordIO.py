import mxnet as mx
import os
import shutil

# --- CONFIG ---
REC_PATH = '/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M/train.rec'
IDX_PATH = '/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/WebFace42M/train.idx'
DEBUG_DIR = './debug_output'
NUM_PERSONS = 10  # Số lượng person ID muốn test


def main():
    # Reset debug folder
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)

    print(f"Reading RecordIO: {REC_PATH}")
    imgrec = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, 'r')

    # Get index range
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        idx_list = range(1, int(header.label[0]))
    else:
        idx_list = imgrec.keys

    processed_persons = set()

    for i in idx_list:
        s = imgrec.read_idx(i)
        header, img_bytes = mx.recordio.unpack(s)

        # Get label
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        person_id = int(label)

        # Stop condition
        if person_id not in processed_persons:
            if len(processed_persons) >= NUM_PERSONS:
                print(f"Reached limit of {NUM_PERSONS} persons. Stopping.")
                break
            processed_persons.add(person_id)
            print(f"Extracting Person ID: {person_id}")

        # Save to local
        person_dir = os.path.join(DEBUG_DIR, str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        
        with open(os.path.join(person_dir, f"{i}.jpg"), "wb") as f:
            f.write(img_bytes)

    print(f"Debug extraction complete at: {os.path.abspath(DEBUG_DIR)}")

if __name__ == "__main__":
    main()
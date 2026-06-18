# import os
# import shutil
# from pathlib import Path

# def organize_facepix(input, output):
#     input_path = Path(input)
#     output_path = Path(output)
#     output_path.mkdir(parents=True, exist_ok =True)

#     for img_file in input_path.glob("*.jpg"):
#         filename = img_file.stem
#         person_id = filename.split("(")[0]

#         person_folder = output_path / f"person_{person_id}"
#         person_folder.mkdir(exist_ok=True)

#         shutil.copy2(img_file, person_folder/img_file.name)
#         print(f"Copied {img_file.name} to {person_folder}")

# if __name__ == "__main__":
#     input = "/workspace/FaceNist/Data/FacePix"
#     output = "/workspace/FaceNist/Data/FacePix_organized"

#     organize_facepix(input, out

import os
import shutil

source_folder = "/workspace/FaceNist/Data/HR_128"

destination = "/workspace/FaceNist/Data/HR_128/MultiPie"

os.makedirs(destination, exist_ok=True)
for filename in os.listdir(source_folder):
    if filename.endswith(".png"):
        person_id = filename.split("_")[0]
        person_folder=os.path.join(destination, f"person_multipie_{person_id}")
        os.makedirs(person_folder, exist_ok=True)

        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(person_folder, filename)
        
        shutil.move(source_path, destination_path)

print("Done")
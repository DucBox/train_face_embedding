import torch

<<<<<<< HEAD
model_paths = ["/workspace/FaceNist/Models/vit_l_60epoch/model_epoch_56.pt", "/workspace/FaceNist/Models/vit_l_60epoch/model_epoch_50.pt", "/workspace/FaceNist/Models/vit_l_60epoch/model_epoch_55.pt"]
=======
model_paths = ["/workspace/FaceNist/arcface_torch/Models/2609/model_epoch_36.pt", "/workspace/FaceNist/arcface_torch/Models/2609/model_epoch_37.pt", "/workspace/FaceNist/arcface_torch/Models/2609/model_epoch_38.pt", "/workspace/FaceNist/arcface_torch/Models/2609/model.pt"]

>>>>>>> ca206dcebfc48520679cc258ac9575de89b5466a

all_state_dicts = []

for path in model_paths:
    checkpoint = torch.load(path, map_location = "cpu")

    if isinstance(checkpoint, dict) and ("state_dict_backbone" in checkpoint or "state _dict_softmax_fc" in checkpoint):
<<<<<<< HEAD

=======
>>>>>>> ca206dcebfc48520679cc258ac9575de89b5466a
        merged_state_dict = {}
        for key, value in checkpoint.items:
            for k, v in value.items():
                merged_state_dict[f"{key}.{k}"] = v

        all_state_dicts.append(merged_state_dict)

    else:
        all_state_dicts.append(checkpoint)

keys = all_state_dicts[0].keys()

avg_state_dict = {}

for key in keys:
    avg_tensor = sum(state_dict[key] for state_dict in all_state_dicts)/len(all_state_dicts)
    avg_state_dict[key] = avg_tensor

<<<<<<< HEAD
torch.save(avg_state_dict, "/workspace/FaceNist/Models/Soups/vit_60epoch_50_55_56.pt")
=======
torch.save(avg_state_dict, "/workspace/FaceNist/arcface_torch/Models/2609/average_36_39.pt")
>>>>>>> ca206dcebfc48520679cc258ac9575de89b5466a

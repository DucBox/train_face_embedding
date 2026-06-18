import argparse

import cv2
import numpy as np
import torch

from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity

def inference(weight, backbone, img1, img2):
    list_embedding = []
    list_img = [img1, img2]
    for img in list_img:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        net = get_model(backbone, fp16=False)
        net.load_state_dict(torch.load(weight))
        net.eval()
        feat = net(img).detach().cpu().numpy()
        list_embedding.append(feat)
    print(f"List Embeddings shape: {list_embedding[0].shape} and Dtype: {list_embedding[0].dtype}")

    similarity = cosine_similarity(list_embedding[0], list_embedding[1])

    print(f"Cosine similarity between 2 imgs: {similarity[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='vit_l_dp005_mask_005')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--img1', type=str, default=None)
    parser.add_argument('--img2', type=str, default=None)
    args = parser.parse_args()
    inference(args.weights, args.network, args.img1, args.img2)
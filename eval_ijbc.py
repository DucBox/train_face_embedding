# coding: utf-8

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc

# THAY THẾ 2 dòng này:
# from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
# from prettytable import PrettyTable

from pathlib import Path

import sys
import warnings

sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--image-path', default='', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        print(f"Loading model from {prefix}")
        print(f"File exists: {os.path.exists(prefix)}")
        weight = torch.load(prefix)
        resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        print(f"Model architecture: {args.network}")
        resnet.load_state_dict(weight)
        print(f"Weights loaded")
        # model = torch.nn.DataParallel(resnet)

        print(f"DataParallel wrapped")

        self.model = resnet
        self.model.eval()
        print(f"Model set tpo eval mosde")
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        print(f"Input {batch_data.shape} {batch_data.dtype}")
        imgs = torch.Tensor(batch_data).cuda()
        print(f"Moved to CUDA {imgs.device}")
        imgs.div_(255).sub_(0.5).div_(0.5)
        print("Normalied")
        print("Calling model.forward")
        feat = self.model(imgs)
        print(f"Model output shape: {feat.shape}")
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        print(f"Reshape to {feat.shape}")
        return feat.cpu().numpy()


def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0

    img_feats = np.empty((len(files), 1024), dtype=np.float32)


    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        if img_index % 1000 == 0:
            print(f"Process image {img_index}/{len(files)}: {name_lmk_score[0]}")
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print(f"batch {batch} - Processing image {batch * batch_size} to {(batch+1) * batch_size}")
            import time
            start_time = time.time()
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
            elapsed = time.time() - start_time
            print(f"  -> Forward time: {elapsed:.2f}s, Speed: {batch_size/elapsed:.1f} img/s")
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)

        # if count_template % 2000 == 0:
        #     # print('Finish Calculating {} template features.'.format(
        #         count_template))

    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))

    total_pairs = np.array(range(len(p1)))

    # print(f"Total pairs: {len(p1)}")

    batchsize = 100000
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()

        # if c % 10 == 0:
            # print('Finish {}/{} pairs.'.format(c, total_sublists))

    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# ============ THAY THẾ HÀM TẠO MÀU ============
def generate_colors(n):
    """Tạo n màu khác nhau"""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(rgb)
    return colors


# ============ THAY THẾ HÀM IN BẢNG ============
def print_table(headers, rows):
    """In bảng đơn giản thay thế PrettyTable"""
    # Tính độ rộng mỗi cột
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Tạo format string
    row_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])
    
    # In bảng
    print("\n" + row_format.format(*headers))
    print(separator)
    for row in rows:
        print(row_format.format(*row))
    print()


# ===============================================
# Step 1: Load Meta Data
# ===============================================

assert target == 'IJBC' or target == 'IJBB'

start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# ===============================================
# Step 2: Get Image Features
# ===============================================

start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
files_list = files

img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                               model_path, 0, gpu_id)
stop = timeit.default_timer()

# print('Time: %.2f s. ' % (stop - start))
# print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
#                                           img_feats.shape[1]))

# ===============================================
# Step 3: Get Template Features
# ===============================================

start = timeit.default_timer()

if use_flip_test:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()

# print('Time: %.2f s. ' % (stop - start))


# ===============================================
# Step 4: Get Template Similarity Scores
# ===============================================

start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()

# print('Time: %.2f s. ' % (stop - start))


save_path = os.path.join(result_dir, args.job)

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)

# ===============================================
# Step 5: Get ROC Curves and TPR@FPR Table
# ===============================================

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))

# THAY THẾ: Tạo màu đơn giản
colours = dict(zip(methods, generate_colors(len(methods))))

x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

# THAY THẾ: Tạo bảng đơn giản
table_headers = ['Methods'] + [str(x) for x in x_labels]
table_rows = []

fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    
    # Tạo row cho bảng
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    table_rows.append(tpr_fpr_row)

plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))

model_path_obj = Path(model_path)
log_dir = model_path_obj.parent
model_filename = model_path_obj.name
log_filename = model_filename.replace(model_path_obj.suffix, '.txt')
log_file_path = log_dir/log_filename
log_dir.mkdir(parents=True, exist_ok=True)


# THAY THẾ: In bảng đơn giản
print(f"Result of model {model_path}")
print_table(table_headers, table_rows)

with open(log_file_path, 'w') as f:
    f.write(str(table_headers))
    f.write(f"\n")
    f.write(str(table_rows))
    f.write(f"\n")


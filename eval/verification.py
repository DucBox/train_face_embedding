import datetime
import os
import pickle
import numpy as np
import sklearn
import torch
import cv2
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from PIL import Image


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
        # Remove duplicates for interpolation
            unique_indices = np.unique(far_train, return_index=True)[1]
            far_train_unique = far_train[np.sort(unique_indices)]
            thresholds_unique = thresholds[np.sort(unique_indices)]

            # Check if we have enough unique points for interpolation
            if len(far_train_unique) > 1:
                f = interpolate.interp1d(far_train_unique, thresholds_unique,
                    kind='linear', fill_value='extrapolate')
                threshold = f(far_target)
            else:
                # Fallback: find closest threshold
                idx = np.argmin(np.abs(far_train - far_target))
                threshold = thresholds[idx]
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size):
    """Load verification dataset - Pure PyTorch version without MXNet"""
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        
        # Decode image using OpenCV instead of MXNet
        img = cv2.imdecode(np.frombuffer(_bin, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if img.shape[0] != image_size[0]:
            img = cv2.resize(img, (image_size[1], image_size[0]))
        
        # Convert to CHW format
        img = img.transpose(2, 0, 1)  # HWC to CHW
        
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2).copy()  # Horizontal flip
            data_list[flip][idx][:] = torch.from_numpy(img)
        
        if idx % 1000 == 0:
            print('loading bin', idx)
    
    print(data_list[0].shape)
    return data_list, issame_list


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            
            time0 = datetime.datetime.now()
            
            # Normalize: [0, 255] -> [-1, 1]
            img = ((_data.float() / 255) - 0.5) / 0.5
            img = img.cuda()
            
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt
    
    # Without flip
    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc1, std1 = np.mean(accuracy), np.std(accuracy)
    
    # With flip
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Verification')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to model.pt file')
    parser.add_argument('--network', type=str, default='vit_l_depth36',
                        help='backbone network')
    parser.add_argument('--data-dir', type=str, default='/workspace/FaceNist/Data',
                        help='path to verification datasets directory')
    parser.add_argument('--target', type=str, default='lfw,agedb_30,calfw,cfp_ff,cfp_fp,cplfw,vgg2_fp',
                        help='verification datasets to test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for inference')
    parser.add_argument('--nfolds', type=int, default=10,
                        help='number of folds for cross-validation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id')
    args = parser.parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Image size
    image_size = [112, 112]
    print('image_size', image_size)
    
    # Load model
    print(f'Loading model from {args.model_path}')
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backbones import get_model
    
    backbone = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    backbone.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    backbone = backbone.cuda()
    backbone.eval()
    print('Model loaded successfully')
    
    # Load verification datasets
    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print(f'Loading {name}...')
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
        else:
            print(f'Warning: {path} not found, skipping...')
    
    if len(ver_list) == 0:
        print('Error: No verification datasets found!')
        exit(1)
    
    # Run verification
    print('\n' + '='*50)
    print('Starting verification tests...')
    print('='*50 + '\n')
    
    results = []
    for i in range(len(ver_list)):
        print(f'\n--- Testing on {ver_name_list[i]} ---')
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(
            ver_list[i], backbone, args.batch_size, args.nfolds)
        
        print(f'[{ver_name_list[i]}] XNorm: {xnorm:.5f}')
        print(f'[{ver_name_list[i]}] Accuracy: {acc1:.5f}+-{std1:.5f}')
        print(f'[{ver_name_list[i]}] Accuracy-Flip: {acc2:.5f}+-{std2:.5f}')
        results.append(acc2)
    
    # Summary
    print('\n' + '='*50)
    print('VERIFICATION RESULTS SUMMARY')
    print('='*50)
    for i, name in enumerate(ver_name_list):
        print(f'{name}: {results[i]:.5f}')
    if len(results) > 0:
        print(f'\nBest accuracy: {np.max(results):.5f}')
        print(f'Average accuracy: {np.mean(results):.5f}')
    print('='*50)
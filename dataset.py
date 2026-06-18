import cv2

from tqdm import tqdm
import numbers
import os
import queue as Queue
import threading
import random
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from PIL import Image
import albumentations as A

def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    use_albumentations = True,
    num_rec_files = 2,
    use_public_dataset=False,
    use_synthetic_dataset=True,
    save_images=False,
    save_dir=None,
    num_save=1000,
    ) -> Iterable:

    # rec = os.path.join(root_dir, 'train.rec')
    # idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False
    
    # Mxnet RecordIO
    else:
        print(f"Check {num_rec_files} RecordIO files")
        datasets = []
        rec_found = False

        if use_synthetic_dataset == False:
            rec_main = os.path.join(root_dir, 'train.rec')
            idx_main = os.path.join(root_dir, 'train.idx')
            if os.path.exists(rec_main) and os.path.exists(idx_main):
                print(f"Loading data from {rec_main} and {idx_main}")
                datasets.append(MXFaceDataset(root_dir=root_dir, file_prefix='train', local_rank=local_rank, use_albumentations=use_albumentations, save_images=save_images, save_dir=save_dir, num_save=num_save))
                rec_found = True
        else:
            rec_main = os.path.join(root_dir, 'train_synthetic.rec')
            idx_main = os.path.join(root_dir, 'train_synthetic.idx')
            if os.path.exists(rec_main) and os.path.exists(idx_main):
                print(f"Loading data from {rec_main} and {idx_main}")
                datasets.append(MXFaceDataset(root_dir=root_dir, file_prefix='train_synthetic', local_rank=local_rank, use_albumentations=use_albumentations, save_images=save_images, save_dir=save_dir, num_save=num_save))
                rec_found = True          

        if use_public_dataset==True:
            print("Using public5m dataset")
            rec_public = os.path.join(root_dir, 'train_public.rec')
            idx_public = os.path.join(root_dir, 'train_public.idx')
            if os.path.exists(rec_public) and os.path.exists(idx_public):
                print(f"Loading Public: {rec_public}")
                datasets.append(MXFaceDataset(root_dir=root_dir, file_prefix='train_public', local_rank=local_rank, use_albumentations=use_albumentations, save_images=save_images, save_dir=save_dir, num_save=num_save))
                rec_found = True

        for i in range(1, num_rec_files):
            file_prefix = f'train_{i}'
            rec_i = os.path.join(root_dir, f'{file_prefix}.rec')
            idx_i = os.path.join(root_dir, f'{file_prefix}.idx')

            if os.path.exists(rec_i) and os.path.exists(idx_i):
                if local_rank == 0:
                    print(f"Loading data from {file_prefix}.rec/.idx")
                datasets.append(MXFaceDataset(root_dir=root_dir, file_prefix=file_prefix, local_rank=local_rank, use_albumentations=use_albumentations, save_images=save_images, save_dir=save_dir, num_save=num_save))
                rec_found = True            
        if rec_found: 
            if len(datasets) > 1:
                print(f"Multi RecordIO files found: {len(datasets)} files")
                train_set = torch.utils.data.ConcatDataset(datasets)
            else:
                print("Single RecordIO files found")
                train_set = datasets[0]
            
            #DEBUG
            if local_rank == 0:
                total_images = len(train_set)
            #     max_id = -1

                # for ds in (datasets if len(datasets) > 1 else [datasets[0]]):
                #     last_idx = ds.imgidx[-1]
                #     s = ds.imgrec.read_idx(last_idx)
                #     header, _ = mx.recordio.unpack(s)
                #     label = header.label
                #     if not isinstance(label, numbers.Number):
                #         label = label[0]
                #     max_id = max(max_id, int(label))

                # print("-" * 30)
                # print(f"DATASET VERIFICATION DONE")
                # print(f"Total Combined Images: {total_images:,}")
                # print(f"Detected Max ID      : {max_id}")
                # print(f"Suggested num_classes: {max_id + 1}")
                # print("-" * 30)

                # actual_ids = set()
                # for ds in datasets:
                #     for idx in tqdm(ds.imgidx, desc=f"Verifying {ds.root_dir}"):
                #         s = ds.imgrec.read_idx(idx)
                #         header, _ = mx.recordio.unpack(s)
                #         label = header.label if isinstance(header.label, numbers.Number) else header.label[0]
                #         actual_ids.add(int(label))
                print(f"Total Combined Images: {total_images:,}")
                # print(f"Absolute Max ID: {max(actual_ids)}")
                # print(f"Absolute Unique Persons: {len(actual_ids)}")

        else:
            print(f"No RecordIO files found")

        if train_set is None:
            print(f"Load from ImageFolder")
            transform = transforms.Compose([
                #  transforms.Resize((224,224)),
                transforms.Resize((112,112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            try:
                train_set = ImageFolder(root_dir, transform)
            except Exception as e:
                print(f"Error while loadimg ImageFolder from {root_dir}: {e}")

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, file_prefix='train', use_albumentations=False, save_images=False, save_dir=None, num_save=1000):
        super(MXFaceDataset, self).__init__()
        if use_albumentations == True:
            # print(f"Using albumentations")
            self.transform = create_transform(use_albumentations)
        else:
            print(f"No albumentations")
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        # print(f"Open mxnet data")
        path_imgrec = os.path.join(root_dir, f"{file_prefix}.rec")
        path_imgidx = os.path.join(root_dir, f"{file_prefix}.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            if local_rank == 0:
                print(f"File {file_prefix} header flag > 0")
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            if local_rank == 0:
                print(f"File {file_prefix} header flag < 0")
            self.imgidx = np.array(list(self.imgrec.keys))
        # print(f"REC 1 loaded. Total images: {len(self.imgidx)}")

        # Print for debug
        # rec1_samples, rec1_classes = self.get_class(self.imgrec, self.imgidx)
        # self.rec1_offset = rec1_classes
        # print(f"Total images rec1: {rec1_samples}")
        # print(f"Total rec1 classes: {rec1_classes}")

        self.save_images = save_images
        self.save_dir = save_dir
        self.num_save = num_save
        self.saved_count = 0

        # if save_images and save_dir and local_rank==0:
        #     os.makedirs(save_dir, exist_ok=True)
        #     print(f"Saving {num_save} images to {save_dir}")

        # if save_images and save_dir and local_rank==0:
        #     self.og_save_dir = os.path.join(save_dir, 'original_images')
        #     self.aug_save_dir = os.path.join(save_dir, 'augmented_images')

        #     os.makedirs(self.og_save_dir, exist_ok=True)
        #     os.makedirs(self.aug_save_dir, exist_ok=True)
        #     print(f"Saving {num_save} pairs to {self.og_save_dir} and {self.aug_save_dir}")

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.float32)
        sample = mx.image.imdecode(img).asnumpy()

        if self.transform is not None:
            sample_transformed = self.transform(sample)
        else:
            sample_transformed=sample

        # filename = f"{self.saved_count:06d}_label{int(label)}.jpg"
        # cv2.imwrite(os.path.join(self.og_save_dir, filename), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

        # if self.save_images and self.saved_count < self.num_save and self.local_rank==0:
        #     if isinstance(sample_transformed, torch.Tensor):
        #         img_to_save = sample_transformed.mul(0.5).add(0.5).mul(255.0).permute(1, 2, 0).byte().cpu().numpy()
        #     else:
        #         img_to_save = sample_transformed

        #     cv2.imwrite(os.path.join(self.aug_save_dir, filename), cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        #     self.saved_count += 1
        #     if self.saved_count == self.num_save:
        #         print("Saved done")

        return sample_transformed, label

    def __len__(self):
        return len(self.imgidx)

    def get_class(self, imgrec, imgidx):
        if not imgidx.size:
            return 0,0

        unique_labels = set()

        for idx in imgidx:
            if self.local_rank == 0 and idx % 1000 == 0:
                print(f"idx: {idx}")
            s = imgrec.read_idx(idx)
            header, _ = mx.recordio.unpack(s)

            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            
            unique_labels.add(int(label))
        
        actual_max_label = max(unique_labels)
        actual_max_num_classes = actual_max_label + 1

        return len(imgidx), actual_max_num_classes
    
class HalfFill127(A.ImageOnlyTransform):
    def __init__(self, always_apply=False,p=0.1):
        super().__init__(always_apply = always_apply, p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            img[:, :w//2]=127
        else:
            img[:, w//2:]=127
        return img

def get_albumentations_transform():
    """
    Create albumentations transform pipeline to replace torchvision transforms
    """
    transform = A.Compose([
        # FLIP
        A.HorizontalFlip(p=0.5),

        #CoarseDropout
        # A.CoarseDropout(num_holes_range=[5,10], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.2], fill=[127, 127, 127], p =0.1),
        
        #FillHalf
        # HalfFill127(always_apply=False, p=0.1),
        
        # BLUR EFFECTS 
        A.OneOf([
            A.AdvancedBlur(
                blur_limit=(3, 5),
                sigma_x_limit=(0.2, 1.0),
                sigma_y_limit=(0.2, 1.0),
                rotate_limit=90,
                beta_limit=(0.5, 8.0),
                noise_limit=(0.9, 1.1),
                p=0.6,
            ),
            A.MedianBlur(blur_limit=(3, 5), p=0.1),
            A.MotionBlur(
                blur_limit=(3, 5),
                allow_shifted=True,
                p=0.3,
            ),
        ], p=0.2),
        
        # NOISE EFFECTS 
        A.OneOf([
            A.GaussNoise(
                var_limit=(10.0, 20.0),
                mean=0,
                per_channel=True,
                p=0.5,
            ),
            A.ISONoise(
                color_shift=(0.00, 0.00),
                intensity=(0.1, 0.4),
                p=0.5,
            ),
        ], p=0.2),
        
        # DOWNSCALE EFFECTS 
        A.OneOf([
            A.Downscale(
                scale_min=0.4,
                scale_max=0.9,
                interpolation=cv2.INTER_CUBIC,
                p=0.25,
            ),
            A.Downscale(
                scale_min=0.4,
                scale_max=0.9,
                interpolation=dict(
                    upscale=cv2.INTER_CUBIC, downscale=cv2.INTER_AREA
                ),
                p=0.25,
            ),
            A.Downscale(
                scale_min=0.4,
                scale_max=0.9,
                interpolation=dict(
                    upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA
                ),
                p=0.25,
            ),
            A.Downscale(
                scale_min=0.4,
                scale_max=0.9,
                interpolation=dict(
                    upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR
                ),
                p=0.25,
            ),
        ], p=0.2),
        
        # COMPRESSION EFFECTS 
        A.OneOf([
            A.ImageCompression(
                quality_lower=60,
                quality_upper=90,
                compression_type="jpeg",
                p=0.5,
            ),
            A.ImageCompression(
                quality_lower=60,
                quality_upper=90,
                compression_type="webp",
                p=0.5,
            ),
        ], p=0.2),
        
        # COLOR ADJUSTMENTS 
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.4),
                contrast_limit=(-0.3, 0.3),
                brightness_by_max=True,
                p=0.2,
            ),
            A.RandomToneCurve(scale=0.1, per_channel=True, p=0.2),
            A.ColorJitter(
               brightness=(0.6, 1.4),
                contrast=(0.6, 1.4),
                saturation=(0.6, 1.4),
                hue=(-0.01, 0.01),
                p=0.2,
            ),
            A.PlanckianJitter(
                mode="blackbody",
                temperature_limit=[4000, 10000],
                sampling_method="uniform",
                p=0.1,
            ),
        ], p=0.5),
        
        # NORMALIZATION 
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
    ])
    
    return transform

class AlbumentationsWrapper:
    """
    Wrapper to make albumentations compatible with MXNet -> PyTorch pipeline
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        # MXNet image is already numpy array (H,W,C) uint8 format
        # Apply albumentations transform directly
        transformed = self.transform(image=image)
        augmented_image = transformed['image']
        
        # Convert to PyTorch tensor (C,H,W) format
        if isinstance(augmented_image, np.ndarray):
            # albumentations Normalize outputs float64, ensure float32
            augmented_image = torch.from_numpy(augmented_image).permute(2, 0, 1).float()
        
        return augmented_image

def get_torchvision_transform():
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform

def create_transform(use_albumentations):
    if use_albumentations:
        # print(f"Using albumentations for augmentation")
        album_transform = get_albumentations_transform()
        return AlbumentationsWrapper(album_transform)
    else:
        # print(f"Using torchvision for augmentation")
        return get_torchvision_transform()


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

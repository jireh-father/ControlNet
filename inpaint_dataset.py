import json
import random
import traceback

import cv2
import numpy as np

from torch.utils.data import Dataset
import os
import copy
import albumentations as albu

import random
from torch.utils.data.sampler import Sampler


class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for cluster_indices in self.data_source.cluster_indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class SizeClusterInpaintDataset(Dataset):
    def __init__(self, data_root, label_path, target_size=512, divisible_by=64, use_transform=False,
                 max_size=768):
        self.data = []
        self.data_root = data_root
        self.target_size = target_size
        self.divisible_by = divisible_by
        self.max_size = max_size

        self.transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # albu.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1,0.02), contrast_limit=0.2),
            albu.HueSaturationValue(p=0.5, hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=15),
            albu.RandomGamma(p=0.5, gamma_limit=(100, 150)),
        ])
        self.use_transform = use_transform
        self.cluster_indices = []
        self.make_cluster_indices(label_path)

    def __len__(self):
        return len(self.data)

    def make_cluster_indices(self, label_path):
        cluster_dict = {}
        with open(label_path, 'rt') as f:
            for line in f:
                item = json.loads(line)

                source_filename = item['source']

                source = cv2.imread(os.path.join(self.data_root, source_filename))

                h, w, _ = source.shape
                if self.target_size > h or self.target_size > w:
                    print("this image is too small", source_filename, "width", w, "height", h)
                    continue

                target_h, target_w = self.calc_divisible_size(source)

                if self.max_size < target_h or self.max_size < target_w:
                    print("this image is too big", source_filename, "width", target_w, "height", target_h, "ori width", w, "ori height", h)
                    continue

                key = (target_h, target_w)
                self.data.append(item)
                cur_idx = len(self.data) - 1
                if key in cluster_dict:
                    cluster_dict[key].append(cur_idx)
                else:
                    cluster_dict[key] = [cur_idx]
        for key in cluster_dict:
            print(key, len(cluster_dict[key]))
        self.cluster_indices = list(cluster_dict.values())

    def calc_divisible_size(self, source):
        h, w, _ = source.shape

        if w > h:
            target_h = self.target_size
            target_w = int(target_h / h * w)
            if target_w + self.divisible_by > self.max_size:
                remain = target_w - self.max_size
                num_height_divisible = remain // self.divisible_by // 2
                target_h = target_h - num_height_divisible * self.divisible_by
                target_w = int(target_h / h * w)

            if target_w % self.divisible_by != 0:
                target_w = (target_w // self.divisible_by) * self.divisible_by
        elif w < h:
            target_w = self.target_size
            target_h = int(target_w / w * h)
            if target_h + self.divisible_by > self.max_size:
                remain = target_h - self.max_size
                num_width_divisible = remain // self.divisible_by // 2
                target_w = target_w - num_width_divisible * self.divisible_by
                target_h = int(target_w / w * h)

            if target_h % self.divisible_by != 0:
                target_h = (target_h // self.divisible_by) * self.divisible_by
        else:
            target_w = self.target_size
            target_h = self.target_size

        return target_h, target_w

    def resize_image(self, source, target):
        h, w, _ = source.shape

        if w > h:
            target_h = self.target_size
            target_w = int(target_h / h * w)
            if target_w + self.divisible_by > self.max_size:
                remain = target_w - self.max_size
                num_height_divisible = remain // self.divisible_by // 2
                target_h = target_h - num_height_divisible * self.divisible_by
                target_w = int(target_h / h * w)

            source = cv2.resize(source, (target_w, target_h))
            target = cv2.resize(target, (target_w, target_h))
            if target_w % self.divisible_by != 0:
                # crop remaining both side
                left_remaining = target_w % self.divisible_by // 2
                right_remaining = target_w % self.divisible_by - left_remaining

                source = source[:, left_remaining:-right_remaining]
                target = target[:, left_remaining:-right_remaining]
        elif w < h:
            target_w = self.target_size
            target_h = int(target_w / w * h)
            if target_h + self.divisible_by > self.max_size:
                remain = target_h - self.max_size
                num_width_divisible = remain // self.divisible_by // 2
                target_w = target_w - num_width_divisible * self.divisible_by
                target_h = int(target_w / w * h)
            source = cv2.resize(source, (target_w, target_h))
            target = cv2.resize(target, (target_w, target_h))
            if target_h % self.divisible_by != 0:
                # crop remaining both side
                top_remaining = target_h % self.divisible_by // 2
                bottom_remaining = target_h % self.divisible_by - top_remaining

                source = source[top_remaining:-bottom_remaining, :]
                target = target[top_remaining:-bottom_remaining, :]

        else:
            source = cv2.resize(source, (self.target_size, self.target_size))
            target = cv2.resize(target, (self.target_size, self.target_size))

        return source, target

    def __getitem__(self, idx):
        while True:
            item = self.data[idx]

            source_filename = item['source']
            target_filename = item['target']
            prompt = item['prompt']

            source = cv2.imread(os.path.join(self.data_root, source_filename))

            h, w, _ = source.shape
            if self.target_size > h or self.target_size > w:
                idx = random.randint(0, len(self.data) - 1)
                print("this image is too small", source_filename, "width", w, "height", h)
                continue

            target = cv2.imread(os.path.join(self.data_root, target_filename))

            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

            try:
                source, target = self.resize_image(source, target)
            except Exception as e:
                print("error file path", source_filename, target_filename)
                traceback.print_exc()
                idx = random.randint(0, len(self.data) - 1)
                continue

            if self.use_transform:
                target = self.transform(image=target)['image']

            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

            # deep copy target
            inpaint_source = copy.deepcopy(target)
            inpaint_source[source > 0.5] = -1.0

            return dict(jpg=target, txt=prompt, hint=inpaint_source)


class InpaintDataset(Dataset):
    def __init__(self, data_root, label_path, use_multi_aspect_ratio=False, target_size=512, divisible_by=None,
                 use_transform=False):
        self.data = []
        self.data_root = data_root
        self.use_multi_aspect_ratio = use_multi_aspect_ratio
        self.target_size = target_size
        self.divisible_by = divisible_by
        with open(label_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = albu.Compose([
            # albu.HorizontalFlip(p=0.5),
            # albu.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1,0.02), contrast_limit=0.2),
            albu.HueSaturationValue(p=0.5, hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=15),
            albu.RandomGamma(p=0.5, gamma_limit=(100, 150)),
        ])
        self.use_transform = use_transform

    def __len__(self):
        return len(self.data)

    def resize_image(self, source, target):
        h, w, _ = source.shape

        if not self.use_multi_aspect_ratio:
            if h > w:
                pad = (h - w) // 2
                source = cv2.copyMakeBorder(source, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                target = cv2.copyMakeBorder(target, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                pad = (w - h) // 2
                source = cv2.copyMakeBorder(source, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                target = cv2.copyMakeBorder(target, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            source = cv2.resize(source, (self.target_size, self.target_size))
            target = cv2.resize(target, (self.target_size, self.target_size))
        else:
            if self.divisible_by:
                if w > h:
                    target_h = self.target_size
                    target_w = int(target_h / h * w)
                    source = cv2.resize(source, (target_w, target_h))
                    target = cv2.resize(target, (target_w, target_h))
                    if target_w % self.divisible_by != 0:
                        # crop remaining both side
                        left_remaining = target_w % self.divisible_by // 2
                        right_remaining = target_w % self.divisible_by - left_remaining

                        source = source[:, left_remaining:-right_remaining]
                        target = target[:, left_remaining:-right_remaining]
                elif w < h:
                    target_w = self.target_size
                    target_h = int(target_w / w * h)
                    source = cv2.resize(source, (target_w, target_h))
                    target = cv2.resize(target, (target_w, target_h))
                    if target_h % self.divisible_by != 0:
                        # crop remaining both side
                        top_remaining = target_h % self.divisible_by // 2
                        bottom_remaining = target_h % self.divisible_by - top_remaining

                        source = source[top_remaining:-bottom_remaining, :]
                        target = target[top_remaining:-bottom_remaining, :]
                else:
                    source = cv2.resize(source, (self.target_size, self.target_size))
                    target = cv2.resize(target, (self.target_size, self.target_size))
            else:
                # resize to min side 512 and other side bigger than 512 keeping aspect ratio
                if h > w:
                    target_h = self.target_size + self.target_size // 2
                    tmp_h = int(h / w * self.target_size)
                    source = cv2.resize(source, (self.target_size, tmp_h))
                    target = cv2.resize(target, (self.target_size, tmp_h))

                    if tmp_h > target_h:
                        # crop
                        source = source[tmp_h // 2 - target_h // 2:tmp_h // 2 + target_h // 2, :]
                        target = target[tmp_h // 2 - target_h // 2:tmp_h // 2 + target_h // 2, :]

                    elif tmp_h < target_h:
                        # pad h
                        pad = (target_h - tmp_h) // 2
                        source = cv2.copyMakeBorder(source, pad, pad if (target_h - tmp_h) % 2 == 0 else pad + 1, 0, 0,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        target = cv2.copyMakeBorder(target, pad, pad if (target_h - tmp_h) % 2 == 0 else pad + 1, 0, 0,
                                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                elif h < w:
                    target_w = self.target_size + self.target_size // 2
                    tmp_w = int(w / h * self.target_size)
                    source = cv2.resize(source, (tmp_w, self.target_size))
                    target = cv2.resize(target, (tmp_w, self.target_size))
                    if tmp_w > target_w:
                        # crop
                        source = source[:, tmp_w // 2 - target_w // 2:tmp_w // 2 + target_w // 2]
                        target = target[:, tmp_w // 2 - target_w // 2:tmp_w // 2 + target_w // 2]
                    elif tmp_w < target_w:
                        # pad w
                        pad = (target_w - tmp_w) // 2
                        source = cv2.copyMakeBorder(source, 0, 0, pad, pad if (target_w - tmp_w) % 2 == 0 else pad + 1,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        target = cv2.copyMakeBorder(target, 0, 0, pad, pad if (target_w - tmp_w) % 2 == 0 else pad + 1,
                                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                else:
                    source = cv2.resize(source, (self.target_size, self.target_size))
                    target = cv2.resize(target, (self.target_size, self.target_size))

        return source, target

    def __getitem__(self, idx):
        while True:
            item = self.data[idx]

            source_filename = item['source']
            target_filename = item['target']
            prompt = item['prompt']

            source = cv2.imread(os.path.join(self.data_root, source_filename))

            h, w, _ = source.shape
            if self.target_size > h or self.target_size > w:
                idx = random.randint(0, len(self.data) - 1)
                print("this image is too small", source_filename, "width", w, "height", h)
                continue

            target = cv2.imread(os.path.join(self.data_root, target_filename))

            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

            try:
                source, target = self.resize_image(source, target)
            except Exception as e:
                print("error file path", source_filename, target_filename)
                traceback.print_exc()
                idx = random.randint(0, len(self.data) - 1)
                continue

            if self.use_transform:
                target = self.transform(image=target)['image']

            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

            # deep copy target
            inpaint_source = copy.deepcopy(target)
            inpaint_source[source > 0.5] = -1.0

            return dict(jpg=target, txt=prompt, hint=inpaint_source)


if __name__ == '__main__':
    dataset = InpaintDataset(data_root='E:/dataset/fill50k',
                             label_path='E:\dataset/fill50k/prompt.json')
    print(len(dataset))
    import torch

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataset_loader:
        print(batch['jpg'].shape, batch['txt'], batch['hint'].shape)
        # convert to pil image
        from PIL import Image
        import matplotlib.pyplot as plt


        # unnormalize to 0~255
        def to_pil(x):
            # x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            x = x.squeeze(0).cpu().numpy()
            x = (x + 1) / 2 * 255
            x = x.clip(0, 255).astype(np.uint8)
            return Image.fromarray(x)


        print(batch['jpg'].max(), batch['jpg'].min())
        img = to_pil(batch['jpg'][0])
        plt.imshow(img)
        plt.show()
        img = to_pil(batch['hint'][0])
        plt.imshow(img)
        plt.show()

        break

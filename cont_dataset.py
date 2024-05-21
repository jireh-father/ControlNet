import json
import os
import traceback

import cv2
import numpy as np

from torch.utils.data import Dataset
import albumentations as albu

from inpaint_dataset import SizeClusterInpaintDataset

class SizeClusterContDataset(SizeClusterInpaintDataset):
    def __init__(self, data_root, label_path, target_size=512, divisible_by=64, use_transform=False, max_size=768, source_invert=False):
        self.source_invert = source_invert
        super().__init__(data_root=data_root, label_path=label_path, target_size=target_size, divisible_by=divisible_by, use_transform=use_transform, max_size=max_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_root, source_filename))

        cluster_height, cluster_width = self.calc_divisible_size(source)
        self.current_cluster_key = (cluster_height, cluster_width)

        h, w, _ = source.shape
        if self.target_size > h or self.target_size > w:
            print("this image is too small", source_filename, "width", w, "height", h)
            raise Exception("this image is too small")

        target = cv2.imread(os.path.join(self.data_root, target_filename))

        target_h, target_w, _ = target.shape
        if h != target_h or w != target_w:
            print("source and target size mismatch", source_filename, target_filename)
            raise Exception("source and target size mismatch")

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)


        try:
            source, target, source_guide, avail_mask = self.resize_image(source, target)
        except Exception as e:
            print("error file path", source_filename, target_filename)
            traceback.print_exc()
            raise Exception("error file path")

        source_h, source_w, _ = source.shape

        if cluster_height != source_h or cluster_width != source_w:
            print("size mismatch", source_filename, target_filename)
            raise Exception("size mismatch")

        if source_h != target.shape[0] or source_w != target.shape[1]:
            print("size mismatch", source_filename, target_filename)
            raise Exception("size mismatch")

        if self.source_invert:
            source = 255 - source

        if self.use_transform:
            transformed = self.transform(image=target, mask=source)
            target, source = transformed['image'], transformed['mask']

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = prompt[prompt.index('1girl'):]
        prompt_list = prompt.split(', ')

        # drop tags 10~80%
        rand_drop_ratio = np.random.uniform(0.2, 0.8)
        drop_count = int(len(prompt_list) * rand_drop_ratio)
        drop_indices = np.random.choice(len(prompt_list), drop_count, replace=False)
        prompt_list = [prompt_list[i] for i in range(len(prompt_list)) if i not in drop_indices]
        prompt = ', '.join(prompt_list)

        return dict(jpg=target, txt=prompt, hint=source)


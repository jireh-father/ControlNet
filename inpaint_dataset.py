import json
import random
import traceback

import cv2
import numpy as np
import re

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
    def __init__(self, data_root, label_path, guide_mask_dir_name=None, avail_mask_dir_name=None,
                 avail_mask_file_prefix='_reverse_face_mask_00001_.png', target_size=512, divisible_by=64,
                 use_transform=False,
                 max_size=768, inpaint_mode='reverse_face_mask',
                 # inpaint_mode: reverse_face_mask, reverse_face_mask_and_lineart, random_mask_and_lineart
                 min_mask_dilation_range=1,
                 max_mask_dilation_range=70,
                 use_hair_mask_prob=0.3,
                 use_long_hair_mask_prob=0.3,
                 use_bottom_hair_prob=0.2
                 ):
        self.data = []
        self.data_root = data_root
        self.target_size = target_size
        self.divisible_by = divisible_by
        self.max_size = max_size
        self.guide_mask_dir_name = guide_mask_dir_name
        self.avail_mask_dir_name = avail_mask_dir_name
        self.avail_mask_file_prefix = avail_mask_file_prefix
        self.use_long_hair_mask_prob = use_long_hair_mask_prob

        transform_list = [
            albu.HorizontalFlip(p=0.5),
            # albu.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1,0.02), contrast_limit=0.2),
            albu.HueSaturationValue(p=0.5, hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=15),
            albu.RandomGamma(p=0.5, gamma_limit=(100, 120)),
        ]
        if inpaint_mode == "reverse_face_mask":
            self.transform = albu.Compose(transform_list)
        elif inpaint_mode == "reverse_face_mask_and_lineart":
            self.transform = albu.Compose(transform_list, additional_targets={'mask1': 'mask'})
        elif inpaint_mode == "random_mask_and_lineart":
            self.transform = albu.Compose(transform_list, additional_targets={'mask1': 'mask', 'mask2': 'mask'})

        self.use_transform = use_transform
        self.cluster_indices = []
        self.make_cluster_indices(label_path)
        self.inpaint_mode = inpaint_mode
        self.mask_dilation_range = (min_mask_dilation_range, max_mask_dilation_range)
        self.use_hair_mask_prob = use_hair_mask_prob
        self.use_bottom_hair_prob = use_bottom_hair_prob

    def __len__(self):
        return len(self.data)

    def get_random_mask(self, hair_mask, guide_mask, avail_mask):
        if random.random() < self.use_hair_mask_prob:
            rand_mask = hair_mask.copy()
        else:
            if random.random() < self.use_bottom_hair_prob:
                # cv2.imshow("hair mask", hair_mask)
                hair_y_indexes = np.where(hair_mask > 0)
                from_y = np.min(hair_y_indexes)
                to_y = np.max(hair_y_indexes)
                from_y = from_y + int((to_y - from_y) * 0.1)
                to_y = to_y - int((to_y - from_y) * 0.1)
                start_y = random.randint(from_y, to_y)
                rand_mask = hair_mask.copy()
                rand_mask[:start_y] = 0
                # cv2.imshow("rand mask", rand_mask)
            else:
                # draw and fill random ellipse as white color
                ellipse_image = np.zeros(guide_mask.shape[:2], dtype=np.uint8)
                center_x = random.randint(0, guide_mask.shape[1])
                center_y = random.randint(0, guide_mask.shape[0])
                axes_x = random.randint(1, guide_mask.shape[1] // 2)
                axes_y = random.randint(1, guide_mask.shape[0] // 2)
                angle = random.randint(0, 360)
                color = 255
                thickness = -1
                cv2.ellipse(ellipse_image, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, color, thickness)
                # cv2.imshow("random mask", ellipse_image)
                # cv2.waitKey(0)
                # cv2.imshow("hair mask", hair_mask)
                # cv2.waitKey(0)

                rand_mask = cv2.bitwise_and(hair_mask, hair_mask, mask=ellipse_image)
                # cv2.imshow("rand mask", rand_mask)
                # cv2.waitKey(0)
                # check mask empty
                if rand_mask.max() == 0:
                    return False
        dilation = random.randint(self.mask_dilation_range[0], self.mask_dilation_range[1])
        rand_mask = cv2.dilate(rand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)))

        # get highest y coordinate of mask when pixel > 0
        # print("rand_mask shape", rand_mask.shape)

        # extract height axis
        height_axis = np.sum(rand_mask, axis=1)
        height_axis = np.sum(height_axis, axis=1)
        # print("height_axis", height_axis.shape)

        y_indexes = np.where(height_axis > 0)
        if y_indexes is None or not y_indexes or len(y_indexes) == 0:
            return False
        # print(y_indexes)
        try:
            y = np.max(y_indexes)
        except Exception as e:
            print("error y_indexes", y_indexes)
            return False
        # print("Y", y)
        # print("height", rand_mask.shape[0])
        if y < rand_mask.shape[0] * 0.8 and random.random() < self.use_long_hair_mask_prob:
            margin = int((rand_mask.shape[0] - y) * 0.1)
            target_y = y + random.randint(margin, min(rand_mask.shape[0] - y - 1 + margin, rand_mask.shape[0] - 1))
            clone_rand_mask = rand_mask.copy()
            # shift clone_rand_mask to target_y by window sliding 3pixels
            # cv2.imshow("1", clone_rand_mask)
            for idx in range(5, target_y - y, 5):
                tmp_rand_mask = clone_rand_mask.copy()
                tmp_rand_mask = np.roll(tmp_rand_mask, idx, axis=0)
                tmp_rand_mask[:idx] = 0
                # dilate
                tmp_prob = random.random()
                if tmp_prob < 0.3:
                    tmp_dilation = random.randint(1, 5)
                    tmp_rand_mask = cv2.dilate(tmp_rand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                        (tmp_dilation, tmp_dilation)))
                elif tmp_prob < 0.6:
                    tmp_shift = random.randint(-5, 5)
                    tmp_rand_mask = np.roll(tmp_rand_mask, tmp_shift, axis=1)

                rand_mask = cv2.bitwise_or(rand_mask, tmp_rand_mask)
            # merge clone_rand_mask and rand_mask
            # cv2.imshow("3", rand_mask)

        rand_guide_mask = guide_mask.copy()
        rand_guide_mask[rand_mask < 1] = 0
        # rand_guide_mask = rand_mask & guide_mask
        if rand_guide_mask.max() == 0:
            return False
        # cv2.imshow("guide_mask", guide_mask)
        # cv2.waitKey(0)
        # cv2.imshow("rand guide mask", rand_guide_mask)
        # cv2.waitKey(0)
        # cv2.imshow("rand mask", rand_mask)
        # cv2.waitKey(0)

        rand_mask[avail_mask < 1] = 0
        rand_guide_mask[avail_mask < 1] = 0
        if rand_mask.max() == 0:
            return False
        if rand_guide_mask.max() == 0:
            return False
        # cv2.imshow("last", rand_mask)
        # cv2.waitKey(0)

        return rand_mask, rand_guide_mask

    def make_cluster_indices(self, label_path):
        cluster_dict = {}
        num_skip = 0
        with open(label_path, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                source_filename = item['source']

                source = cv2.imread(os.path.join(self.data_root, source_filename))
                # if source is None:
                #     continue
                h, w, _ = source.shape
                if self.target_size > h or self.target_size > w:
                    num_skip += 1
                    # print("this image is too small", source_filename, "width", w, "height", h)
                    continue

                # if self.max_size < h or self.max_size < w:
                #     num_skip += 1
                #     # print("this image is too big", source_filename, "width", w, "height", h)
                #     continue

                target_h, target_w = self.calc_divisible_size(source)
                if self.max_size < target_h or self.max_size < target_w:
                    num_skip += 1
                    # print("this image is too big after resize", source_filename, "width", target_w, "height", target_h)
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
        print("skip", num_skip, "images")
        print("data", len(self.data))
        self.cluster_indices = list(cluster_dict.values())

    def calc_divisible_size(self, source):
        h, w, _ = source.shape

        if w > h:
            target_h = self.target_size
            target_w = int(target_h / h * w)
            # if self.max_size + self.divisible_by < target_w:
            #     pass
            if target_w > self.max_size:
                target_w = self.max_size
            elif target_w % self.divisible_by != 0:
                target_w = (target_w // self.divisible_by) * self.divisible_by

        elif w < h:
            target_w = self.target_size
            target_h = int(target_w / w * h)

            if target_h > self.max_size:
                target_h = self.max_size
            elif target_h % self.divisible_by != 0:
                target_h = (target_h // self.divisible_by) * self.divisible_by
        else:
            target_w = self.target_size
            target_h = self.target_size

        return target_h, target_w

    def resize_image(self, source, target, source_guide=None, avail_mask=None):
        h, w, _ = source.shape

        if w > h:
            target_h = self.target_size
            target_w = int(target_h / h * w)

            source = cv2.resize(source, (target_w, target_h))
            target = cv2.resize(target, (target_w, target_h))
            if source_guide is not None:
                source_guide = cv2.resize(source_guide, (target_w, target_h))
            if avail_mask is not None:
                avail_mask = cv2.resize(avail_mask, (target_w, target_h))
            if target_w > self.max_size or target_w % self.divisible_by != 0:
                if target_w > self.max_size:
                    left_remaining = (target_w - self.max_size) // 2
                    right_remaining = target_w - self.max_size - left_remaining
                else:
                    # crop remaining both side
                    left_remaining = target_w % self.divisible_by // 2
                    right_remaining = target_w % self.divisible_by - left_remaining

                source = source[:, left_remaining:-right_remaining]
                target = target[:, left_remaining:-right_remaining]
                if source_guide is not None:
                    source_guide = source_guide[:, left_remaining:-right_remaining]
                if avail_mask is not None:
                    avail_mask = avail_mask[:, left_remaining:-right_remaining]

        elif w < h:
            target_w = self.target_size
            target_h = int(target_w / w * h)
            source = cv2.resize(source, (target_w, target_h))
            target = cv2.resize(target, (target_w, target_h))
            if source_guide is not None:
                source_guide = cv2.resize(source_guide, (target_w, target_h))
            if avail_mask is not None:
                avail_mask = cv2.resize(avail_mask, (target_w, target_h))
            if target_h > self.max_size or target_h % self.divisible_by != 0:
                if target_h > self.max_size:
                    bottom_remaining = target_h - self.max_size
                else:
                    bottom_remaining = target_h % self.divisible_by

                source = source[:-bottom_remaining, :]
                target = target[:-bottom_remaining, :]
                if source_guide is not None:
                    source_guide = source_guide[:-bottom_remaining, :]
                if avail_mask is not None:
                    avail_mask = avail_mask[:-bottom_remaining, :]
        else:
            source = cv2.resize(source, (self.target_size, self.target_size))
            target = cv2.resize(target, (self.target_size, self.target_size))
            if source_guide is not None:
                source_guide = cv2.resize(source_guide, (self.target_size, self.target_size))
            if avail_mask is not None:
                avail_mask = cv2.resize(avail_mask, (self.target_size, self.target_size))

        return source, target, source_guide, avail_mask

    def _getitem(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        if self.inpaint_mode == "reverse_face_mask_and_lineart" or self.inpaint_mode == "random_mask_and_lineart":
            if self.guide_mask_dir_name:
                source_guide_filename = os.path.join(self.guide_mask_dir_name, os.path.basename(item['target']))
            else:
                source_guide_filename = item['source_guide']
            if self.inpaint_mode == "random_mask_and_lineart":
                if self.avail_mask_dir_name:
                    avail_mask_filename = os.path.join(self.avail_mask_dir_name, os.path.splitext(
                        os.path.basename(item['target']))[0] + self.avail_mask_file_prefix)
                else:
                    avail_mask_filename = item['source_avail_mask']
                # --inpaint_mode random_mask_and_lineart --guide_mask_dir_name hair_lineart_mask &
        else:
            source_guide_filename = None
            avail_mask_filename = None
        prompt = item['prompt']
        if self.inpaint_mode == "reverse_face_mask_and_lineart" or self.inpaint_mode == "random_mask_and_lineart":
            prompt = prompt[prompt.index('1girl'):]

        source = cv2.imread(os.path.join(self.data_root, source_filename))

        h, w, _ = source.shape
        if self.target_size > h or self.target_size > w:
            print("this image is too small", source_filename, "width", w, "height", h)
            raise Exception("this image is too small")

        target = cv2.imread(os.path.join(self.data_root, target_filename))

        target_h, target_w, _ = target.shape
        if h != target_h or w != target_w:
            print("source and target size mismatch", source_filename, target_filename)
            raise Exception("source and target size mismatch")

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if source_guide_filename:
            source_guide = cv2.imread(os.path.join(self.data_root, source_guide_filename))
            source_guide_h, source_guide_w, _ = source_guide.shape
            if h != source_guide_h or w != source_guide_w:
                print("source and source_guide size mismatch", source_filename, source_guide_filename)
                raise Exception("source and source_guide size mismatch")
            source_guide = cv2.cvtColor(source_guide, cv2.COLOR_BGR2RGB)
        else:
            source_guide = None

        if avail_mask_filename:
            avail_mask = cv2.imread(os.path.join(self.data_root, avail_mask_filename))
            avail_mask_h, avail_mask_w, _ = avail_mask.shape
            if h != avail_mask_h or w != avail_mask_w:
                print("source and avail_mask size mismatch", source_filename, avail_mask_filename)
                raise Exception("source and avail_mask size mismatch")
            avail_mask = cv2.cvtColor(avail_mask, cv2.COLOR_BGR2RGB)
        else:
            avail_mask = None

        try:
            source, target, source_guide, avail_mask = self.resize_image(source, target, source_guide, avail_mask)
        except Exception as e:
            print("error file path", source_filename, target_filename, source_guide_filename)
            traceback.print_exc()
            raise Exception("error file path")

        source_h, source_w, _ = source.shape
        if source_h != target.shape[0] or source_w != target.shape[1]:
            print("size mismatch", source_filename, target_filename)
            raise Exception("size mismatch")

        if source_guide is not None and (source_h != source_guide.shape[0] or source_w != source_guide.shape[1]):
            print("size mismatch", source_filename, source_guide_filename)
            raise Exception("size mismatch")

        if avail_mask is not None and (source_h != avail_mask.shape[0] or source_w != avail_mask.shape[1]):
            print("size mismatch", source_filename, avail_mask_filename)
            raise Exception("size mismatch")

        if self.use_transform:
            if self.inpaint_mode == "reverse_face_mask_and_lineart":
                transformed = self.transform(image=target, mask=source, mask1=source_guide)
                target, source, source_guide = transformed['image'], transformed['mask'], transformed['mask1']
            elif self.inpaint_mode == "random_mask_and_lineart":
                transformed = self.transform(image=target, mask=source, mask1=source_guide, mask2=avail_mask)
                target, source, source_guide, avail_mask = transformed['image'], transformed['mask'], transformed[
                    'mask1'], transformed['mask2']
            else:
                transformed = self.transform(image=target, mask=source)
                target, source = transformed['image'], transformed['mask']

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        if self.inpaint_mode == "reverse_face_mask_and_lineart":
            source_guide = source_guide.astype(np.float32) / 255.0
        elif self.inpaint_mode == "random_mask_and_lineart":
            source_guide = source_guide.astype(np.float32) / 255.0
            avail_mask = avail_mask.astype(np.float32) / 255.0

        # Normalize tarƒet images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # deep copy target
        inpaint_source = copy.deepcopy(target)

        if self.inpaint_mode == "reverse_face_mask_and_lineart":
            inpaint_source[source > 0.5] = -1.0
            inpaint_source[source_guide > 0.5] = 1.0
        elif self.inpaint_mode == "random_mask_and_lineart":
            num_try = 0
            is_failed = False
            while True:
                masks = self.get_random_mask(source, source_guide, avail_mask)
                if masks is not False:
                    break
                num_try += 1
                if num_try > 5:
                    print("generating random mask failed")
                    is_failed = True
                    break
            if is_failed:
                raise Exception("generating random mask failed")
            rand_mask, rand_guide_mask = masks
            inpaint_source[rand_mask > 0.5] = -1.0
            inpaint_source[rand_guide_mask > 0.5] = 1.0
        elif self.inpaint_mode == "reverse_face_mask":
            inpaint_source[source > 0.5] = -1.0

        if source_h != target.shape[0] or source_w != target.shape[1]:
            print("size mismatch", source_filename, target_filename)
            raise Exception("size mismatch")

        if source_h != inpaint_source.shape[0] or source_w != inpaint_source.shape[1]:
            print("size mismatch", source_filename, target_filename)
            raise Exception("size mismatch")

        return dict(jpg=target, txt=prompt, hint=inpaint_source)

    def __getitem__(self, idx):
        while True:
            try:
                return self._getitem(idx)
            except Exception as e:
                print("error idx", idx)
                traceback.print_exc()
                idx = random.randint(0, len(self.data) - 1)


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
            albu.HorizontalFlip(p=0.5),
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
                transformed = self.transform(image=target, mask=source)
                target, source = transformed['image'], transformed['mask']

            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

            # deep copy target
            inpaint_source = copy.deepcopy(target)
            inpaint_source[source > 0.5] = -1.0

            return dict(jpg=target, txt=prompt, hint=inpaint_source)


if __name__ == '__main__':
    # dataset = SizeClusterInpaintDataset("D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet",
    #                                     "D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet/reverse_face_mask_prompt.json",
    #                                     guide_mask_dir_name="D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet/hair_lineart_mask",
    #                                     target_size=512, divisible_by=64, use_transform=False,
    #                                     max_size=768, inpaint_mode='reverse_face_mask_and_lineart')
    dataset = SizeClusterInpaintDataset("D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet",
                                        "D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet/exact_hair_mask_prompt.json",
                                        guide_mask_dir_name="D:\dataset\hair_style\hairshop_sample_from_gisu\controlnet/hair_lineart_mask",
                                        target_size=512, divisible_by=64, use_transform=False,
                                        max_size=768, inpaint_mode='random_mask_and_lineart',  # use_hair_mask_prob=0.,
                                        avail_mask_dir_name='reverse_face_mask_source',
                                        avail_mask_file_prefix='_reverse_face_mask_00001_.png',
                                        # use_long_hair_mask_prob=1.0,
                                        # use_bottom_hair_prob=1.0
                                        )
    # dataset = InpaintDataset(data_root='E:/dataset/fill50k',
    #                          label_path='E:\dataset/fill50k/prompt.json')
    print(len(dataset))
    import torch

    sampler = ClusterRandomSampler(dataset, 1, True)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False)
    i = 0
    for batch in dataloader:
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
        # img = to_pil(batch['jpg'][0])
        # plt.imshow(img)
        # plt.show()
        img = to_pil(batch['hint'][0])
        img.save(f"hint_{i}.jpg")
        # plt.imshow(img)
        # plt.show()

        i += 1

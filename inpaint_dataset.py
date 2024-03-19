import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os
import copy


class InpaintDataset(Dataset):
    def __init__(self, data_root, label_path):
        self.data = []
        self.data_root = data_root
        with open(label_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_root, source_filename))
        target = cv2.imread(os.path.join(self.data_root, target_filename))

        # resize to 512x512 keeping the aspect ratio with padding aligned to the center
        h, w, _ = source.shape
        if h > w:
            pad = (h - w) // 2
            source = cv2.copyMakeBorder(source, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            target = cv2.copyMakeBorder(target, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            pad = (w - h) // 2
            source = cv2.copyMakeBorder(source, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            target = cv2.copyMakeBorder(target, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # deep copy target
        inpaint_source = copy.deepcopy(target)
        inpaint_source[source > 0.5] = -1.0

        return dict(jpg=target, txt=prompt, hint=inpaint_source)

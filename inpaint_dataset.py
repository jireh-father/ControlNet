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

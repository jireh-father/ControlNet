import json
import random
import traceback

import cv2
import numpy as np

from torch.utils.data import Dataset
import os
import copy


class InpaintDataset(Dataset):
    def __init__(self, data_root, label_path, use_multi_aspect_ratio=False, target_size=512):
        self.data = []
        self.data_root = data_root
        self.use_multi_aspect_ratio = use_multi_aspect_ratio
        self.target_size = target_size
        with open(label_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

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
            # resize to min side 512 and other side bigger than 512 keeping aspect ratio
            if h > w:
                target_h = self.target_size + self.target_size // 2
                tmp_h = int(h / w * self.target_size)
                source = cv2.resize(source, (self.target_size, tmp_h))
                target = cv2.resize(target, (self.target_size, tmp_h))

                if tmp_h > target_h:
                    #crop
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

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from inpaint_dataset import InpaintDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse


def main(args):
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = InpaintDataset(args.data_root, args.label_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--model_config', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--data_root', type=str, help='Root directory of the dataset')
    parser.add_argument('--label_path', type=str, help='Path to the label file')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=4)
    # logger_freq
    parser.add_argument('--logger_freq', type=int, default=300)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    args = parser.parse_args()
    main(args)

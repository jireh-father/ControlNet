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
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=args.precision, callbacks=[logger], max_epochs=args.max_epochs,
                         min_epochs=args.max_epochs, default_root_dir=args.default_root_dir)

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
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--precision', type=int, default=16)
    # max_epochs
    parser.add_argument('--max_epochs', type=int, default=10)
    # logger_freq
    parser.add_argument('--logger_freq', type=int, default=300)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # default_root_dir
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    args = parser.parse_args()
    main(args)

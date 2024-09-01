import os

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cont_dataset import SizeClusterContDataset
from inpaint_dataset import ClusterRandomSampler
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
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

    if args.every_n_train_steps:
        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=args.every_n_train_steps,
            save_top_k=args.save_top_k,
            save_weights_only=True,
            monitor="global_step",
            # mode="max",
            dirpath=args.default_root_dir,
            filename=os.path.basename(args.default_root_dir) + "-model-{epoch:02d}-{global_step}",
        )
    elif args.every_n_epochs:
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=args.every_n_epochs,
            save_top_k=args.save_top_k,
            save_weights_only=True,
            monitor="global_step",
            dirpath=args.default_root_dir,
            filename=os.path.basename(args.default_root_dir) + "-model-{epoch:02d}-{global_step}",
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=args.save_top_k,
            save_weights_only=True,
            monitor="global_step",
            mode="max",
            dirpath=args.default_root_dir,
            filename=os.path.basename(args.default_root_dir) + "-model-{epoch:02d}-{global_step}",
        )

    # Misc
    dataset = SizeClusterContDataset(args.data_root, args.label_path, target_size=args.input_target_size,
                                     divisible_by=args.divisible_by,
                                     use_transform=args.use_transform,
                                     max_size=args.input_max_size,
                                     source_invert=args.source_invert,
                                     hori_flip_prob=args.hori_flip_prob
                                     )

    sampler = ClusterRandomSampler(dataset, args.batch_size, True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=args.precision, callbacks=[logger, checkpoint_callback],
                         max_epochs=args.max_epochs,
                         min_epochs=args.max_epochs, default_root_dir=args.default_root_dir,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         auto_lr_find=args.auto_lr_find,
                         )
    # Train!
    trainer.fit(model, dataloader)

    # model.save_weights(os.path.join(args.default_root_dir, "final.ckpt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--model_config', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--data_root', type=str, help='Root directory of the dataset')
    parser.add_argument('--label_path', type=str, help='Path to the label file')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    # save_top_k
    parser.add_argument('--save_top_k', type=int, default=10)
    parser.add_argument('--precision', type=int, default=16)
    # max_epochs
    parser.add_argument('--max_epochs', type=int, default=10)
    # logger_freq
    parser.add_argument('--logger_freq', type=int, default=300)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # default_root_dir
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    # auto_lr_find
    parser.add_argument('--auto_lr_find', action='store_true', default=False)

    # input_target_size
    parser.add_argument('--input_target_size', type=int, default=512)

    parser.add_argument('--divisible_by', type=int, default=None)
    # use_transform
    parser.add_argument('--use_transform', action='store_true', default=False)
    # use_size_cluster
    parser.add_argument('--use_size_cluster', action='store_true', default=False)
    # input_max_size
    parser.add_argument('--input_max_size', type=int, default=768)
    # source_invert
    parser.add_argument('--source_invert', action='store_true', default=False)
    # hori_flip_prob
    parser.add_argument('--hori_flip_prob', type=float, default=0.5)

    # every_n_train_steps
    parser.add_argument('--every_n_train_steps', type=int, default=None)

    # every_n_epochs
    parser.add_argument('--every_n_epochs', type=int, default=None)

    args = parser.parse_args()
    main(args)

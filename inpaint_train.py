import os

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from inpaint_dataset import SizeClusterInpaintDataset, InpaintDataset, ClusterRandomSampler
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

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k,
        save_weights_only=True,
        monitor="global_step",
        mode="max",
        dirpath=args.default_root_dir,
        filename=os.path.basename(args.default_root_dir) + "-model-{epoch:02d}-{global_step}",
    )

    if args.use_size_cluster:
        # Misc
        dataset = SizeClusterInpaintDataset(args.data_root, args.label_path, target_size=args.input_target_size,
                                            divisible_by=args.divisible_by,
                                            use_transform=args.use_transform,
                                            max_size=args.input_max_size,
                                            inpaint_mode=args.inpaint_mode,
                                            guide_mask_dir_name=args.guide_mask_dir_name,
                                            avail_mask_dir_name=args.avail_mask_dir_name,
                                            avail_mask_file_prefix=args.avail_mask_file_prefix,
                                            use_long_hair_mask_prob=args.use_long_hair_mask_prob,
                                            use_hair_mask_prob=args.use_hair_mask_prob,
                                            min_mask_dilation_range=args.min_mask_dilation_range,
                                            max_mask_dilation_range=args.max_mask_dilation_range,
                                            use_bottom_hair_prob=args.use_bottom_hair_prob,
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
    else:
        dataset = InpaintDataset(args.data_root, args.label_path, target_size=args.input_target_size,
                                 use_multi_aspect_ratio=args.use_multi_aspect_ratio,
                                 divisible_by=args.divisible_by,
                                 use_transform=args.use_transform)

        dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
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
    parser.add_argument('--guide_mask_dir_name', type=str, default='hair_lineart_mask')
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
    # use_multi_aspect_ratio
    parser.add_argument('--use_multi_aspect_ratio', action='store_true', default=False)

    parser.add_argument('--divisible_by', type=int, default=None)
    # use_transform
    parser.add_argument('--use_transform', action='store_true', default=False)
    # use_size_cluster
    parser.add_argument('--use_size_cluster', action='store_true', default=False)
    # input_max_size
    parser.add_argument('--input_max_size', type=int, default=768)
    #inpaint_mode
    parser.add_argument('--inpaint_mode', type=str, default='reverse_face_mask') # reverse_face_mask, reverse_face_mask_and_lineart, random_mask_and_lineart
    # avail_mask_dir_name
    parser.add_argument('--avail_mask_dir_name', type=str, default='reverse_face_mask_source')
    # avail_mask_file_prefix
    parser.add_argument('--avail_mask_file_prefix', type=str, default='_reverse_face_mask_00001_.png')
    # use_long_hair_mask_prob
    parser.add_argument('--use_long_hair_mask_prob', type=float, default=0.3)
    # use_hair_mask_prob
    parser.add_argument('--use_hair_mask_prob', type=float, default=0.3)
    # use_bottom_hair_prob
    parser.add_argument('--use_bottom_hair_prob', type=float, default=0.2)
    # min_mask_dilation_range
    parser.add_argument('--min_mask_dilation_range', type=int, default=1)
    # max_mask_dilation_range
    parser.add_argument('--max_mask_dilation_range', type=int, default=70)

    args = parser.parse_args()
    main(args)

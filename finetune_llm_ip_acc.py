from torch.utils.data import DataLoader
from inpaint_dataset import SizeClusterInpaintDataset, ClusterRandomSampler
from cldm.model import create_model, load_state_dict
import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator



def main(args):
    sd_locked = True
    only_mid_control = False

    accelerator = Accelerator()

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control




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
                                        another_source_prob=args.another_source_prob,
                                        another_source_key_postfix=args.another_source_key_postfix,
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

    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(args.max_epochs):
        epoch += 1
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            model.on_train_batch_start(batch, step, 0)
            loss = model.training_step(batch, step)
            model.on_train_batch_end()

            accelerator.backward(loss)

            optimizer.step()
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')


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
    # another_source_prob
    parser.add_argument('--another_source_prob', type=float, default=None)
    # another_source_key_postfix
    parser.add_argument('--another_source_key_postfix', type=str, default='_over_eyes')
    # hori_flip_prob
    parser.add_argument('--hori_flip_prob', type=float, default=0.5)

    args = parser.parse_args()
    main(args)

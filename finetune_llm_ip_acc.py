import math

from torch.utils.data import DataLoader
from inpaint_dataset import SizeClusterInpaintDataset, ClusterRandomSampler
from cldm.model import create_model, load_state_dict
import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import os
from tqdm import tqdm

def main(args):
    sd_locked = True
    only_mid_control = False

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )


    # weight_dtype = torch.float32
    # if args.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     os.environ["ATTN_PRECISION"] = "fp16"
    # elif args.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     os.environ["ATTN_PRECISION"] = "bf16"

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(accelerator.device)#, dtype=weight_dtype)


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

    max_train_steps = args.max_epochs * math.ceil(
        len(dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
    )
    progress_bar = tqdm(range(max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                        desc="steps")

    optimizer = model.configure_optimizers()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(args.max_epochs):
        if args.save_init_model and epoch == 0:
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.default_root_dir, "init.ckpt"))
                print("Init model saved")

        epoch += 1
        loss_total = 0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate():
                optimizer.zero_grad()

                loss, _ = model(batch)

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

            if accelerator.sync_gradients:
                progress_bar.update(1)

            current_loss = loss.detach().item()

            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)


        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                #save model every n epochs
                if epoch % args.save_every_n_epochs == 0:
                    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.default_root_dir, f"epoch_{epoch}.ckpt"))
                    print(f"Model saved at epoch {epoch}")

    accelerator.end_training()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.default_root_dir, "final.ckpt"))
        print("Training finished")

    del accelerator


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
    # save_every_n_epochs
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # save_top_k
    parser.add_argument('--save_top_k', type=int, default=10)
    # max_epochs
    parser.add_argument('--max_epochs', type=int, default=10)
    # logger_freq
    parser.add_argument('--logger_freq', type=int, default=300)
    # mixed_precision
    parser.add_argument('--mixed_precision', type=str, default='fp16') #float, fp16
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)#1e-05
    # default_root_dir
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # auto_lr_find
    parser.add_argument('--auto_lr_find', action='store_true', default=False)

    # input_target_size
    parser.add_argument('--input_target_size', type=int, default=512)
    # use_multi_aspect_ratio
    parser.add_argument('--use_multi_aspect_ratio', action='store_true', default=False)
    # save init model
    parser.add_argument('--save_init_model', action='store_true', default=False)

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

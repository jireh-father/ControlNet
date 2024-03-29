import argparse
from PIL import Image
import os
import shutil
import glob


def calc_divisible_size(image, target_size=512, divisible_by=64):
    w, h = image.size

    if w > h:
        target_h = target_size
        target_w = int(target_h / h * w)

        if target_w % divisible_by != 0:
            target_w = (target_w // divisible_by) * divisible_by
    elif w < h:
        target_w = target_size
        target_h = int(target_w / w * h)

        if target_h % divisible_by != 0:
            target_h = (target_h // divisible_by) * divisible_by
    else:
        target_w = target_size
        target_h = target_size

    return target_h, target_w


def main(args):
    image_files = glob.glob(args.image_pattern)
    os.makedirs(args.output_dir, exist_ok=True)

    for image_file in image_files:
        image = Image.open(image_file)
        h, w = calc_divisible_size(image)
        if h > 640 or w > 640:
            print(f"skip {image_file}")
            shutil.copy(image_file, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_pattern', type=str, default='D:\dataset\hair_controlnet_images_no_filter\images/*.jpg')
    parser.add_argument('--output_dir', type=str, default='./data/output')
    main(parser.parse_args())

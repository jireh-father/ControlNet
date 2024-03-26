import argparse
import os
import torch
from cldm.model import create_model, load_state_dict
import glob


def main(args):
    model_paths = glob.glob(args.model_path_pattern)
    for model_path in model_paths:
        model = create_model(args.model_config).cpu()
        model.load_state_dict(load_state_dict(model_path, location='cpu'))
        model_fname, ext = os.path.splitext(os.path.basename(model_path))
        output_path = os.path.join(os.path.dirname(model_path), model_fname + "_weights" + ext)
        torch.save(model.state_dict(), output_path)
        if args.remove_original:
            os.remove(model_path)
        print(f'Weights saved to [{output_path}]')
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract only weights from a model')
    parser.add_argument('--model_path_pattern', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--model_config', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--remove_original', default=False, action='store_true')
    main(parser.parse_args())

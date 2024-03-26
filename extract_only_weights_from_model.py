import argparse
import os
import torch
from cldm.model import create_model, load_state_dict


def main(args):
    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)

    print(f'Weights saved to [{args.output_path}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract only weights from a model')
    parser.add_argument('--model_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--output_path', type=str, default='./models/control_sd15_ini_weights.ckpt')
    parser.add_argument('--model_config', type=str, default='./models/cldm_v15.yaml')
    main(parser.parse_args())

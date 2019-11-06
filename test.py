import os
import argparse
import torch
from eval.final_evaluator import UnetEvaluator
import model.unet_model as module_arch
from train import get_instance


def main(config, resume):

    # build model architecture
    model = get_instance(module_arch, 'generator_arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume, map_location='cpu')
    if 'GAN' in config["name"] or 'PG' in config["name"] or 'Real' in config["name"]:
        state_dict = checkpoint['generator_state_dict']
    else:
        state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    evaluator = UnetEvaluator(model, config)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

    if args.resume:
        config = torch.load(args.resume, map_location=device)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)

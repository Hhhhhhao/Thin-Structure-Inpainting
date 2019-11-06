import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.unet_model as module_g_arch
import model.patch_discriminator as module_dl_arch
import model.global_discriminator as module_dg_arch
from trainer import GANTrainer
from utils import Logger
from eval.final_evaluator import UnetEvaluator


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    train_data_loader = get_instance(module_data, 'train_data_loader', config)
    valid_data_loader = get_instance(module_data, 'valid_data_loader', config)

    # build model architecture
    model = {}
    model["generator"] = get_instance(module_g_arch, 'generator_arch', config)
    print(model["generator"])
    model["local_discriminator"] = get_instance(module_dl_arch, 'local_discriminator_arch', config)
    print(model["local_discriminator"])
    model["global_discriminator"] = get_instance(module_dg_arch, 'global_discriminator_arch', config)
    print(model["global_discriminator"])

    # get function handles of loss and metrics
    loss = {}
    loss["vanilla_gan"] = torch.nn.BCELoss()
    loss["lsgan"] = torch.nn.MSELoss()
    loss["ce"] = module_loss.cross_entropy2d
    loss["pg"] = module_loss.PG_Loss()
    loss["mask_ce"] = module_loss.Masked_CrossEntropy()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = {}
    generator_trainable_params = filter(lambda p: p.requires_grad, model["generator"].parameters())
    local_discriminator_trainable_params = filter(lambda p: p.requires_grad, model["local_discriminator"].parameters())
    global_discriminator_trainable_params = filter(lambda p: p.requires_grad, model["global_discriminator"].parameters())
    optimizer["generator"] = get_instance(torch.optim, 'generator_optimizer', config, generator_trainable_params)
    optimizer["local_discriminator"] = get_instance(torch.optim, 'discriminator_optimizer', config, local_discriminator_trainable_params)
    optimizer["global_discriminator"] = get_instance(torch.optim, 'discriminator_optimizer', config,
                                                    global_discriminator_trainable_params)
    # lr_scheduler = None # get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = GANTrainer(model, optimizer, loss, metrics,
                         resume=resume,
                         config=config,
                         data_loader=train_data_loader,
                         valid_data_loader=valid_data_loader,
                         train_logger=train_logger)
    print("pretrain models")
    trainer.pre_train()
    print("training")
    trainer.train()
    evaluator = UnetEvaluator(trainer.generator, trainer.config)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(config, args.resume)

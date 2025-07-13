"""_main_training.py"""
"""to train the geometry autoencoder model"""

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.cuda.set_device(0)

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    net = Solver(args)

    if args.train:
        net.train()
        if args.train_then_test:
            net._test_model()
    else:
        net._test_model()
    print('****well done****')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='control beta_VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--train_then_test', default=True, type=str2bool, help='train then test')
    parser.add_argument('--train_with_validation', default=True, type=str2bool, help='train with validation set')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
   
    parser.add_argument('--max_epoch', default=500, type=float, help='maximum training epoch')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')

    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE (B)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--is_PID', default=True, type=str2bool, help='if use pid or not')
    parser.add_argument('--beta', default=3, type=float, help='beta parameter for KL-term in original beta-VAE')

    parser.add_argument('--pid_fixed', default=False, type=str2bool, help='if fixed PID or dynamic')
    parser.add_argument('--KL_loss', default=32, type=float, help='KL_divergence max target')
    parser.add_argument('--C_start', default=0.5, type=float, help='start value of KL target if dynamic PI-control')
    parser.add_argument('--step_val', default=0.05, type=float, help='step value of increasing KL target')

    parser.add_argument('--dset_dir', default='_data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='bldgview_top', type=str, help='dataset name')
    parser.add_argument('--image_size', default=128, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--ratio_train', default=0.8, type=float, help='ratio of training data')
    parser.add_argument('--ratio_val', default=0.1, type=float, help='ratio of validation data')
    
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visualisation')
    parser.add_argument('--viz_name', default='main', type=str, help='all files and models base name')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--num_traverse', default=10, type=int, help='number of traverse images to save')
    parser.add_argument('--create_gif', default=True, type=str2bool, help='create traverse gif during testing')

    parser.add_argument('--gather_epoch', default=1, type=int, help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_epoch', default=1, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_epoch', default=20, type=int, help='number of iterations after which a checkpoint is saved')
    
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')

    parser.add_argument('--encoding_only', default=False, type=str2bool, help='only run encoding for doppelGANger attributes')
    
    args = parser.parse_args()

    assert args.encoding_only == False, 'encoding_only should not be run from _main_training.py, see _main_encoding.py'
    if not args.train_with_validation and args.ratio_val > 0:
        print("Warning: train_with_validation is False, but ratio_val is greater than 0. Setting ratio_val to 0.")
        args.ratio_val = 0
    
    main(args)
    
    

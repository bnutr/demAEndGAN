"""_main_reconstructing.py"""
"""to reconstruct images from the latent representation z"""
import os
import time
import torch
import argparse
import warnings
import numpy as np

from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

from utils import cuda, str2bool, traverse_gif
from model import BetaVAE_H, BetaVAE_B, reparametrize
from dataset import return_data
from traverse import *

warnings.filterwarnings("ignore")


# region Reconstructing images
class Reconstructing(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.z_dim = args.z_dim
        self.model = args.model
        self.model_name = args.model_name
        self.dataset = args.dataset
        
        dataset_config = {
            'bldgview_top': (3, 'gaussian'),
            'bldgview_allparallel': (3, 'gaussian'),
            'bldgview_ne': (3, 'gaussian'),
            'bldgview_se': (3, 'gaussian'),
            'bldgview_sw': (3, 'gaussian'),
            'bldgview_nw': (3, 'gaussian'),
            'traverse_nw': (3, 'gaussian'),
            'traverse_se': (3, 'gaussian'),
            'traverse_top': (3, 'gaussian'),
        }

        if self.dataset.lower() in dataset_config:
            self.nc, self.decoder_dist = dataset_config[self.dataset.lower()]
        else:
            raise NotImplementedError

        if self.model == 'H':
            net = BetaVAE_H
        elif self.model == 'B':
            raise NotImplementedError
            # net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        
        self.model_dir = os.path.join('trained_model_', self.model_name)
        if os.path.exists(self.model_dir):
            self.load_checkpoint(self.model_dir)
        else:
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist.")
            
        self.save_reconstructed_dir = 'reconstructed_results_'
        if not os.path.exists(self.save_reconstructed_dir):
            os.makedirs(self.save_reconstructed_dir, exist_ok=True)

        self.data_all = return_data(args)


    def _reconstructing(self):
        print('******--reconstructing now--****')
        self.net_mode(train=False)

        reconstructed_subpath = os.path.join(self.save_reconstructed_dir, self.dataset)
        if not os.path.exists(reconstructed_subpath):
            os.makedirs(reconstructed_subpath, exist_ok=True)
        ids = 0
        batch = 0

        for x in self.data_all:
            batch += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, _, _ = self.net(x)
            x_recon = F.sigmoid(x_recon)
            for i in range(x_recon.size(0)):
                save_image(x_recon.data[i], os.path.join(reconstructed_subpath, f'reconstructed_{ids}.png'))
                ids += 1

        print('******--reconstructing finished--****')             


    # region Misc.
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()


    def load_checkpoint(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.net.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded model '{})'".format(file_path))
        else:
            print("=> no model found at '{}'".format(file_path))
    # endregion
# endregion


# region Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='control beta_VAE producing reconstructed images')

    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--z_dim', default=64, type=int, help='dimension of the representation z')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model_name', default='controlvae_se_z64_KL50', type=str, help='model name')
    parser.add_argument('--dataset', default='bldgview_se', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')

    parser.add_argument('--dset_dir', default='_data2', type=str, help='dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--image_size', default=128, type=int, help='image size. now only (128,128) is supported')

    parser.add_argument('--encoding_only', default=True, type=str2bool, help='not for training or testing')
    
    args = parser.parse_args()

    assert args.encoding_only == True, 'training and testing should not be run from _main_reconstructing.py, see _main_training.py'
    
    start_time = time.time()
    reconstructing_init = Reconstructing(args)
    reconstructing_init._reconstructing()
    end_time = time.time()
    print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")
# endregion

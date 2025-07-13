"""_main_encoding.py"""
"""to encode images and get latent representation for the timeseries generator attributes"""

import os
import time
import torch
import random
import argparse
import warnings
import numpy as np

from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

from utils import cuda, str2bool
from model import BetaVAE_H, BetaVAE_B, reparametrize
from dataset import return_data

warnings.filterwarnings("ignore")


# region Encoding attributes
class Encoding_Attr(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.z_dim = args.z_dim
        self.model = args.model
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.z_useful = args.z_useful
        self.save_filename_suffix = args.save_filename_suffix

        if self.z_useful is None:
            self.z_useful = np.arange(self.z_dim)
        
        dataset_config = {
            'bldgview_top': (3, 'gaussian'),
            'bldgview_allparallel': (3, 'gaussian'),
            'bldgview_ne': (3, 'gaussian'),
            'bldgview_se': (3, 'gaussian'),
            'bldgview_sw': (3, 'gaussian'),
            'bldgview_nw': (3, 'gaussian')
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
            
        self.save_encoding_dir = 'encoding_results_'
        if not os.path.exists(self.save_encoding_dir):
            os.makedirs(self.save_encoding_dir, exist_ok=True)

        self.data_all = return_data(args)


    def _encoding_attr(self):
        print('******--encoding now--****')
        self.net_mode(train=False)

        all_mu_z = []
        all_logvar_z = []
        img_names = []
        k = 0
        random_ids = np.random.random_integers(0, len(self.data_all)-1, 5)
        print(f"Random IDs to check decoder fidelity: {random_ids}")
        save_recon_path = os.path.join(self.save_encoding_dir, self.model_name+'_recon_'+self.save_filename_suffix)
        if not os.path.exists(save_recon_path):
            os.makedirs(save_recon_path, exist_ok=True)

        for i, x in enumerate(self.data_all):
            img_name = self.data_all.dataset.imgs[i*self.batch_size:(i+1)*self.batch_size]
            img_name = [os.path.basename(name[0]) for name in img_name]
            img_name = [name.split('_')[0].replace('ID', '') for name in img_name]

            x = Variable(cuda(x, self.use_cuda))
            mu_z = self.net.encoder(x)[:, :self.z_dim]
            logvar_z = self.net.encoder(x)[:, self.z_dim:]

            if i in random_ids:
                x_recon, _, _ = self.net(x[0:1])
                x_recon = F.sigmoid(x_recon)
                comparison = torch.cat([x[0].unsqueeze(0), x_recon])
                save_image(comparison, os.path.join(save_recon_path, f'ori_vs_recon_{k}.png'))
                k += 1
                
            mu_z = mu_z.data.cpu().numpy()
            mu_z = mu_z[:, self.z_useful]
            logvar_z = logvar_z.data.cpu().numpy()
            logvar_z = logvar_z[:, self.z_useful]

            img_names.append(img_name)
            all_mu_z.append(mu_z)
            all_logvar_z.append(logvar_z)
        
        img_names = np.array(img_names).flatten().astype(int).reshape(-1, 1)
        all_mu_z = np.concatenate(all_mu_z, axis=0).astype(float).round(4)
        all_logvar_z = np.concatenate(all_logvar_z, axis=0).astype(float).round(4)

        save_npz_path = os.path.join(self.save_encoding_dir, f'encoded_z_{self.model_name}_{self.save_filename_suffix}.npz')
        np.savez_compressed(save_npz_path, img_names=img_names, all_mu_z=all_mu_z, all_logvar_z=all_logvar_z)
        print('******--encoding finished--****')             


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
    parser = argparse.ArgumentParser(description='control beta_VAE encoding attributes')

    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model_name', default='controlvae_top_z32_KL50', type=str, help='model name')
    parser.add_argument('--dataset', default='bldgview_top', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--z_useful', default=None, type=int, nargs='+', help='useful z dimensions')

    parser.add_argument('--dset_dir', default='_data', type=str, help='dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--image_size', default=128, type=int, help='image size. now only (128,128) is supported')
    parser.add_argument('--save_filename_suffix', default='', type=str, help='suffix added to the save filename')

    parser.add_argument('--encoding_only', default=True, type=str2bool, help='only run encoding for timeseries generator attributes')
    
    args = parser.parse_args()

    assert args.encoding_only == True, 'training and testing should not be run from _main_encoding.py, see _main_training.py'
    
    start_time = time.time()
    encoding_init = Encoding_Attr(args)
    encoding_init._encoding_attr()
    end_time = time.time()
    print(f"Encoding completed in {end_time - start_time:.2f} seconds.")
# endregion

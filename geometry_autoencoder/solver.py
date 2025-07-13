"""solver.py"""
# This code is based on the controlVAE
# from the original repository: https://github.com/HuajieShao/ControlVAE-ICML2020

import torch
import warnings
warnings.filterwarnings("ignore")

import os
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, traverse_gif
from model import BetaVAE_H, BetaVAE_B, reparametrize
from dataset import return_data
from PID import PIDControl
from traverse import traverse_latent_space
from metric import cal_fid, cal_ssim

import matplotlib.pyplot as plt


# region Losses
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss
    

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld
# endregion

# region DataGather
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(epoch=[],
                    iter=[],
                    recon_loss_train_record=[],
                    total_kld_train_record=[],
                    total_kld_train_stored=[],
                    bvae_loss_train_record=[],
                    recon_loss_val_record=[],
                    total_kld_val_record=[],
                    bvae_loss_val_record=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[], beta=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
# endregion

# region Solver
class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.epoch = 0
        self.iter = 0
        self.max_epoch = args.max_epoch
        
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.KL_loss = args.KL_loss
        self.pid_fixed = args.pid_fixed
        self.is_PID = args.is_PID
        self.step_value = args.step_val
        self.C_start = args.C_start
        self.model_name = args.viz_name
        self.train_with_validation = args.train_with_validation
        
        dataset_config = {
            'dsprites': (1, 'bernoulli'),
            'celeba': (3, 'gaussian'),
            'bldgview_top': (3, 'gaussian'),
            'bldgview_allparallel': (3, 'gaussian'),
            'bldgview_ne': (3, 'gaussian'),
            'bldgview_se': (3, 'gaussian'),
            'bldgview_sw': (3, 'gaussian'),
            'bldgview_nw': (3, 'gaussian')
        }

        if args.dataset.lower() in dataset_config:
            self.nc, self.decoder_dist = dataset_config[args.dataset.lower()]
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            raise NotImplementedError
            # net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')
        
        ## load model
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))
        self.C_stored = None
        self.W_k1_stored = None
        self.I_k1_stored = None
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        self.viz_name = args.viz_name

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.viz_on = args.viz_on
        self.viz_writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, 'log'))
            
        self.num_traverse = args.num_traverse
        self.create_gif = args.create_gif
        
        self.gather_epoch = args.gather_epoch
        self.display_epoch = args.display_epoch
        self.save_epoch = args.save_epoch

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        self.data_train, self.data_val, self.data_test =\
            return_data(args)
        
        self.gather = DataGather()
        
    # region Train/Validation
    def train(self):
        out = False
        pbar_epoch = tqdm(total=int(self.max_epoch), desc="Epoch Progress")
        pbar_batch = tqdm(total=len(self.data_train), desc="Batch Progress")
        pbar_epoch.update(self.epoch)
        pbar_batch.update(self.iter)

        if self.epoch == self.max_epoch:
            return

        ## write log to csv file
        csv_file = os.path.join(self.ckpt_dir, "train.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = df[df['epoch'] <= self.epoch]
            df.to_csv(csv_file, index=False)
            csv_log = open(csv_file, mode='a')
            csv_log.flush()
        else:
            csv_log = open(csv_file, mode='w')
            if self.train_with_validation:
                csv_log.write("epoch,recon_loss_train,total_kld_train,beta_vae_loss_train,"
                    "recon_loss_val,total_kld_val,beta_vae_loss_val,mean_kld_train,"
                    "beta\n"
                    )
            else:
                csv_log.write("epoch,recon_loss_train,total_kld_train,beta_vae_loss_train,"
                    "mean_kld_train,beta\n")
            csv_log.flush()

        ## init PID control
        if self.is_PID:
            Kp = 0.01
            Ki = -0.0001
            Kd = 0.0
            period = 5
            if self.pid_fixed:
                C = self.KL_loss
            else:
                if self.C_stored == None:
                    C = self.C_start
                    PID = PIDControl(0.0, 0.0)
                else: 
                    C = self.C_stored
                    PID = PIDControl(self.W_k1_stored, self.I_k1_stored)

        while not out:
            self.epoch += 1
            pbar_batch.reset()
            if self.is_PID and not self.pid_fixed:
                if self.epoch % period == 0:
                    C += (self.step_value*self.KL_loss)
                if C > self.KL_loss:
                    C = self.KL_loss

            self.net_mode(train=True)
            recon_loss_train_record = 0
            total_kld_train_record = 0
            beta_vae_loss_train_record = 0
            
            # region Training
            for x in self.data_train:
                self.iter += 1
                pbar_batch.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss_train = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld_train, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.is_PID:
                    self.beta, _, W_k1_stored, I_k1_stored = PID.pid(C, total_kld_train.item(), Kp, Ki, Kd)
                    beta_vae_loss_train = recon_loss_train + self.beta * total_kld_train
                else:
                    beta_without_pid = self.beta
                    beta_vae_loss_train = recon_loss_train + beta_without_pid * total_kld_train
                
                recon_loss_train_record += recon_loss_train.item()
                total_kld_train_record += total_kld_train.item()
                beta_vae_loss_train_record += beta_vae_loss_train.item()

                self.optim.zero_grad()
                beta_vae_loss_train.backward()
                self.optim.step()
            
            recon_loss_train_record /= len(self.data_train)
            total_kld_train_record /= len(self.data_train)
            beta_vae_loss_train_record /= len(self.data_train)
            # endregion

            # region Validating
            if self.train_with_validation:
                self.net_mode(train=False)
                recon_loss_val_record = 0
                total_kld_val_record = 0
                beta_vae_loss_val_record = 0

                with torch.no_grad():
                    for x_val in self.data_val:
                        x_val = Variable(cuda(x_val, self.use_cuda))
                        x_recon_val, mu_val, logvar_val = self.net(x_val)
                        recon_loss_val = reconstruction_loss(x_val, x_recon_val, self.decoder_dist)
                        total_kld_val_batch, _, _ = kl_divergence(mu_val, logvar_val)
                        total_kld_val = total_kld_val_batch
                        if self.is_PID:
                            beta_vae_loss_val = recon_loss_val + self.beta * total_kld_val
                        else:
                            beta_vae_loss_val = recon_loss_val + beta_without_pid * total_kld_val
                        recon_loss_val_record += recon_loss_val.item()
                        total_kld_val_record += total_kld_val.item()
                        beta_vae_loss_val_record += beta_vae_loss_val.item()
                        
                recon_loss_val_record /= len(self.data_val)
                total_kld_val_record /= len(self.data_val)
                beta_vae_loss_val_record /= len(self.data_val)
            # endregion

            # region Other functions
            if self.epoch % self.gather_epoch == 0:
                self.gather.insert(epoch=self.epoch,
                                   iter=self.iter,
                                   mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                   recon_loss_train_record=recon_loss_train_record,
                                   total_kld_train_record=total_kld_train_record,
                                   total_kld_train_stored=total_kld_train.item(),
                                   bvae_loss_train_record=beta_vae_loss_train_record,
                                   recon_loss_val_record=recon_loss_val_record if self.train_with_validation else None,
                                   total_kld_val_record=total_kld_val_record if self.train_with_validation else None,
                                   bvae_loss_val_record=beta_vae_loss_val_record if self.train_with_validation else None,
                                   dim_wise_kld=dim_wise_kld.data,
                                   mean_kld=mean_kld.data,
                                   beta=self.beta)
            
            if self.epoch % self.gather_epoch == 0:
                if self.train_with_validation:
                    csv_log.write("{},{},{},{},{},{},{},{},{}\n".format(self.epoch,
                                                                        recon_loss_train_record,
                                                                        total_kld_train_record,
                                                                        beta_vae_loss_train_record,
                                                                        recon_loss_val_record,
                                                                        total_kld_val_record,
                                                                        beta_vae_loss_val_record,
                                                                        mean_kld.item(),
                                                                        self.beta
                                                                        ))
                else:
                    csv_log.write("{},{},{},{},{},{}\n".format(self.epoch,
                                                               recon_loss_train_record,
                                                               total_kld_train_record,
                                                               beta_vae_loss_train_record,
                                                               mean_kld.item(), 
                                                               self.beta
                                                               ))
                csv_log.flush()

            if self.viz_on and self.epoch % self.save_epoch == 0:
                self.gather.insert(images=x.data)
                self.gather.insert(images=F.sigmoid(x_recon).data)
                self.viz_reconstruction()
                self.viz_scalars()
                self.gather.flush()

            if (self.viz_on or self.save_output) and self.epoch % self.save_epoch == 0:
                self.viz_traverse()

            if self.epoch % self.save_epoch == 0:
                if self.is_PID:
                    self.C_stored = C
                    self.W_k1_stored = W_k1_stored
                    self.I_k1_stored = I_k1_stored
                else:
                    self.C_stored = None
                    self.W_k1_stored = None
                    self.I_k1_stored = None
                self.save_checkpoint('last')
                self.save_checkpoint(str(self.epoch))                

            pbar_epoch.update(1)
            if self.epoch >= self.max_epoch:                                            
                out = True
                break
            # endregion

        pbar_epoch.write("[Training Finished]")
        pbar_epoch.close()
        pbar_batch.close()
        csv_log.close()
    # endregion

    # region Visualisation (Training)
    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True, nrow=10)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True, nrow=10)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz_writer.add_images(f'{self.viz_name}_reconstruction', images, self.epoch, dataformats='NCHW')
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.epoch))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, fp=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
        self.net_mode(train=True)
        

    def viz_scalars(self):
        self.net_mode(train=False)
        recon_losses_train = torch.Tensor(self.gather.data['recon_loss_train_record']).cpu()
        betas = torch.Tensor(self.gather.data['beta']).cpu()
        mean_klds = torch.stack(self.gather.data['mean_kld']).cpu()
        total_klds_train = torch.Tensor(self.gather.data['total_kld_train_record']).cpu()
        epochs = torch.Tensor(self.gather.data['epoch']).cpu()

        if self.train_with_validation:
            recon_losses_val = torch.Tensor(self.gather.data['recon_loss_val_record']).cpu()
            total_klds_val = torch.Tensor(self.gather.data['total_kld_val_record']).cpu()
        
        for i, epoch in enumerate(epochs):
            self.viz_writer.add_scalar('Loss/Reconstruction_Train', recon_losses_train[i], epoch)
            self.viz_writer.add_scalar('KLD/Total_Train', total_klds_train[i], epoch)
            self.viz_writer.add_scalar('KLD/Mean', mean_klds[i], epoch)
            self.viz_writer.add_scalar('Beta', betas[i], epoch)
                        
            if self.train_with_validation:
                self.viz_writer.add_scalar('Loss/Reconstruction_Val', recon_losses_val[i], epoch)
                self.viz_writer.add_scalar('KLD/Total_Val', total_klds_val[i], epoch)
                                            
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.epoch))
            os.makedirs(output_dir, exist_ok=True)
            fig = plt.figure(figsize=(10, 10), dpi=300)
            recon_losses = torch.Tensor(self.gather.data['recon_loss_train_record']).cpu()
            legend_reconloss = ['Reconstruction Loss']
            plt.plot(epochs, recon_losses)
            plt.legend(legend_reconloss)
            plt.xlabel('epoch')
            plt.title('reconstruction loss')
            fig.savefig(os.path.join(output_dir, 'graph_recon_loss.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(epochs, betas)
            plt.xlabel('epoch')
            plt.title('beta')
            fig.savefig(os.path.join(output_dir, 'graph_beta.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            klds = torch.Tensor(self.gather.data['total_kld_train_record']).cpu()
            legend_kld = ['KL Divergence']
            plt.plot(epochs, klds)
            plt.legend(legend_kld)
            plt.xlabel('epoch')
            plt.title('kl divergence')
            fig.savefig(os.path.join(output_dir, 'graph_kld.jpg'))
            
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        num_image = 10
        
        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        
        n_dsets = len(self.data_train.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_train.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_mu_z = encoder(random_img)[:, :self.z_dim]
        
        ###------------fixed image------------------
        fixed_idx = 0
        fixed_img = self.data_train.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_mu_z = encoder(fixed_img)[:, :self.z_dim]
        # Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        eps = Variable(cuda(torch.FloatTensor(num_image, self.z_dim).uniform_(-1, 1), self.use_cuda),volatile=True)
        fixed_z = fixed_mu_z + eps
        
        ## ------------rand traverse------------------
        ## random hidden state from uniform
        random_z = Variable(cuda(torch.rand(num_image, self.z_dim), self.use_cuda), volatile=True)
        # random_z = Variable(cuda(torch.FloatTensor(1, self.z_dim).uniform_(-1, 1), self.use_cuda),volatile=True)

        ## save image to folder
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.epoch))
            os.makedirs(output_dir, exist_ok=True)

        ## visualise image
        Z_image = {'fixed_z':fixed_z, 'random_z':random_z}

        for key in Z_image.keys():
            z = Z_image[key]
            samples = F.sigmoid(decoder(z)).data
            ## visualise
            title = '{}_latent_traversal(epoch:{})'.format(key, self.epoch)
            self.viz_writer.add_images(f'{self.viz_name}_traverse/{key}', samples, self.epoch, dataformats='NCHW')
            ## save image to folder
            if self.save_output:
                save_image(samples, fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, self.epoch)), \
                            nrow=num_image, pad_value=1)
        ###-------interplote linear space----------

        self.net_mode(train=True)
    # endregion

    # region Test
    def _test_model(self):
        print('******--testing model now--****')
        test_path = os.path.join('results', self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        
        predict_path = os.path.join(test_path, 'testing/predict')
        ground_path = os.path.join(test_path, 'ground/ground_truth')
        traverse_path = os.path.join(test_path, 'traverse')
        image_path = [predict_path, ground_path, traverse_path]
    
        for path in image_path:
            if not os.path.exists(path):
                os.makedirs(path)

        ## evaluate the result
        self.net_mode(train=False)
        ids = 0
        batch = 0

        for x in self.data_test:
            batch += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, _, _ = self.net(x)
            samples = F.sigmoid(x_recon).data
            batch_size = samples.size(0)

            for b in range(batch_size):
                ids += 1
                save_image(samples[b,:,:,:], fp=os.path.join(image_path[0], 'predict_{}.jpg'.format(ids)))
                save_image(x[b,:,:,:], fp=os.path.join(image_path[1], 'ground_{}.jpg'.format(ids)))
        
        ## traversing latent space
        random_int = random.randint(0, len(self.data_test.dataset))
        test_img = self.data_test.dataset.__getitem__(random_int)
        test_img = Variable(cuda(test_img, self.use_cuda), volatile=True).unsqueeze(0)
        test_mu_z = self.net.encoder(test_img)[:, :self.z_dim]
        
        n_traverse = self.num_traverse
        latent_vectors = traverse_latent_space(test_mu_z, self.z_dim, (-3, 3), n_traverse)
    
        traverse_images = []
        for i, latent_vector in enumerate(latent_vectors):
            latent_vector = Variable(latent_vector, volatile=True)
            sample = F.sigmoid(self.net.decoder(latent_vector))
            traverse_images.append(sample)
            save_image(sample, fp=os.path.join(traverse_path,
                                               'traverse_z{}_n{}.jpg'.format(i//n_traverse,i%n_traverse)),
                                               pad_value=1)

        if self.create_gif:
            traverse_gif(traverse_path, traverse_path)

        ## calculate FID and SSIM scores
        cal_fid(predict_path, ground_path, test_path)
        cal_ssim(predict_path, ground_path, test_path)

        print('******--testing model finished--****')             
    # endregion

    # region Misc. (save and load)
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'beta': self.win_beta,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,
                      }
        states = {'epoch':self.epoch,
                  'C_stored':self.C_stored,
                  'Wk1_stored':self.W_k1_stored,
                  'Ik1_stored':self.I_k1_stored,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (epoch {})".format(file_path, self.epoch))   

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.epoch = checkpoint['epoch']
            self.C_stored = checkpoint['C_stored']
            self.W_k1_stored = checkpoint['Wk1_stored']
            self.I_k1_stored = checkpoint['Ik1_stored']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
    # endregion
# endregion
    

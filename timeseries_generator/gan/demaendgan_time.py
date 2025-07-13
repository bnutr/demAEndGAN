import torch
from torch.autograd import grad

import numpy as np
from tqdm import tqdm
import datetime
import os
import math
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import glob

from .util import draw_feature
from .util import draw_attribute
from .util import create_gif
from .util import renormalize_per_sample
from .network import demAEndGANTimeGenerator
from .network import PrimDiscriminator
from .network import AuxlDiscriminator

# This code is based on the DoppelGANger implementation 
# from the original repository: https://github.com/fjxmlzn/DoppelGANger
# This code is the DoppelGANger implementation on PyTorch version 2.6.0


class demAEndGANTime(object):
    # region ___init__
    def __init__(self, args,
                 checkpoint_dir, sample_dir, time_path, generate_dir,
                 data_feature, data_attribute, data_gen_flag,
                 data_feature_outputs, data_attribute_outputs,
                 real_attribute_mask, 
                 data_attribute_real,
                 data_attribute_scalers,
                 data_auxl_signal, data_feature_ground_truth
                 ):
        """Initialise the demAEndGANTime class

        Args:
            args (Namespace): Arguments for the model configuration.
            checkpoint_dir (str): Directory to save checkpoints.
            sample_dir (str): Directory to save samples.
            time_path (str): Path to save training time logs.
            generate_dir (str): Directory to save generated data.
            data_feature (np.ndarray): Feature data for training.
            data_attribute (np.ndarray): Attribute data for training.
            data_gen_flag (np.ndarray): Generation flags for the data.
            data_feature_outputs (list): List of feature output configurations.
            data_attribute_outputs (list): List of attribute output configurations.
            real_attribute_mask (list): Mask for real attributes.
            data_attribute_real (np.ndarray): Real attribute data.
            data_attribute_scalers (list): List of attribute scalers.
            data_auxl_signal (np.ndarray): Auxiliary signal data that informs the real feature signal.
            data_feature_ground_truth (np.ndarray): Ground truth (non-normalised) feature data.
        Raises:
            NotImplementedError: If differential privacy is not implemented.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.generate_dir = generate_dir
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.sample_len = args.sample_len
        self.num_packing = args.num_packing

        self.d_rounds = args.d_rounds
        self.g_rounds = args.g_rounds

        self.d_gp_coe = args.d_gp_coe
        self.d_auxl_gp_coe = args.d_auxl_gp_coe
        self.g_d_auxl_coe = args.g_d_auxl_coe
        self.g_rmse_coe = args.g_rmse_coe

        self.g_lr = args.g_lr
        self.g_beta1 = args.g_beta1
        self.g_beta2 = args.g_beta2
        self.g_feature_feedback = args.g_feature_feedback
        self.g_feature_noise = args.g_feature_noise
        self.g_attribute_latent_dim = args.g_attribute_latent_dim
        self.g_attribute_num_layers = args.g_attribute_num_layers
        self.g_attribute_num_units = args.g_attribute_num_units
        self.g_feature_latent_dim = args.g_feature_latent_dim
        self.g_feature_num_layers = args.g_feature_num_layers
        self.g_feature_num_units = args.g_feature_num_units

        self.d_lr = args.d_lr
        self.d_beta1 = args.d_beta1
        self.d_beta2 = args.d_beta2
        self.d_num_layers = args.d_num_layers
        self.d_num_units = args.d_num_units

        self.d_auxl_lr = args.d_auxl_lr
        self.d_auxl_beta1 = args.d_auxl_beta1
        self.d_auxl_beta2 = args.d_auxl_beta2
        self.d_auxl_num_layers = args.d_auxl_num_layers
        self.d_auxl_num_units = args.d_auxl_num_units

        self.dp_noise_multiplier = args.dp_noise_multiplier
        self.dp_l2_norm_clip = args.dp_l2_norm_clip
        self.dp_delta = args.dp_delta

        self.train_with_attr_noise = args.train_with_attr_noise
        self.train_with_auxl_signal = args.train_with_auxl_signal
        self.train_with_discriminator_auxl = args.train_with_discriminator_auxl
        self.train_with_rmse_loss = args.train_with_rmse_loss
        self.train_fix_feature_net = args.train_fix_feature_net

        self.generate_with_attr_noise = args.generate_with_attr_noise
        self.generate_with_auxl_signal = args.generate_with_auxl_signal
        self.generate_num_sample = args.generate_num_sample
        self.restore = args.restore
        self.verbose_summary = args.verbose_summary
        
        self.epoch_checkpoint_freq = args.epoch_checkpoint_freq
        self.vis_freq = args.vis_freq
        self.vis_num_sample = args.vis_num_sample

        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.data_gen_flag = data_gen_flag
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.real_attribute_mask = real_attribute_mask

        self.data_attribute_real = data_attribute_real
        self.data_attribute_scalers = data_attribute_scalers

        self.data_feature_ground_truth = data_feature_ground_truth

        # TODO: Implement Differential Privacy
        if (self.dp_noise_multiplier is not None 
            and self.dp_l2_norm_clip is not None):
            raise NotImplementedError 
        
        # feature_dim0: number of samples
        # feature_dim1: number of timesteps
        # feature_dim2: number of features
        self.feature_dim0 = self.data_feature.shape[0]
        self.feature_dim1 = self.data_feature.shape[1]
        self.feature_dim2 = self.data_feature.shape[2]

        # attribute_dim0: number of samples
        # attribute_dim1: number of attributes
        self.attribute_dim0 = self.data_attribute.shape[0]
        self.attribute_dim1 = self.data_attribute.shape[1]

        # real_attribute_dim0: number of samples
        # real_attribute_dim1: number of real attributes
        # addi_attribute_dim1: number of additional (min max) attributes
        self.real_attribute_dim0 = self.data_attribute_real.shape[0]
        self.real_attribute_dim1 = self.data_attribute_real.shape[1]
        self.addi_attribute_dim1 = self.attribute_dim1 - self.real_attribute_dim1

        self.gen_flag_dims = []
        self.gen_flag_dims = next(
            ([dim, dim + 1] for dim, output 
             in enumerate(self.data_feature_outputs) 
             if output.is_gen_flag and output.dim == 2),
            None
        )

        self.sample_time = int(self.feature_dim1 / self.sample_len)

        if data_auxl_signal is not None and self.train_with_auxl_signal:
            self.data_auxl_signal = data_auxl_signal.reshape(-1, self.sample_time, 
                                                             self.sample_len*data_auxl_signal.shape[2])
            # auxl_signal_dim0: number of samples
            # auxl_signal_dim1: number of sample_time
            # auxl_signal_dim2: number of sample_len*features
            self.auxl_signal_dim0 = self.data_auxl_signal.shape[0]
            self.auxl_signal_dim1 = self.data_auxl_signal.shape[1]
            self.auxl_signal_dim2 = self.data_auxl_signal.shape[2]
        else:
            self.data_auxl_signal = None
            self.auxl_signal_dim0 = None
            self.auxl_signal_dim1 = None
            self.auxl_signal_dim2 = None     
        
        self.checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(self.model_name)]
        self.checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=False)
        self.checkpoint_file_last = self.checkpoint_files[-1] if self.checkpoint_files else None
        
        self.writer = SummaryWriter(self.checkpoint_dir)
        
        self.network_init()     
    # endregion

    # region Network initialisation
    def network_init(self):
        """Initialize the network components and optimisers."""
        self.generator = demAEndGANTimeGenerator(
                            device=self.device,
                            attribute_outputs=self.data_attribute_outputs,
                            feature_outputs=self.data_feature_outputs,
                            real_attribute_mask=self.real_attribute_mask,
                            sample_len=self.sample_len,
                            feature_dim2=self.feature_dim2,
                            feedback=self.g_feature_feedback,
                            noise=self.g_feature_noise,
                            attribute_latent_dim=self.g_attribute_latent_dim,
                            attribute_num_layers=self.g_attribute_num_layers,
                            attribute_num_units=self.g_attribute_num_units,
                            feature_latent_dim=self.g_feature_latent_dim,
                            feature_num_layers=self.g_feature_num_layers,
                            feature_num_units=self.g_feature_num_units,
                            auxl_signal_dim2=self.auxl_signal_dim2,
                            ).to(self.device)
    
        d_input_dim = (self.feature_dim2 * self.feature_dim1 +
                       self.attribute_dim1)
        self.discriminator = PrimDiscriminator(input_dim=d_input_dim, 
                                               num_layers=self.d_num_layers,
                                               num_units=self.d_num_units).to(self.device)
         
        d_auxl_input_dim = self.attribute_dim1
        if self.train_with_discriminator_auxl:
            self.discriminator_auxl = AuxlDiscriminator(
                input_dim=d_auxl_input_dim,
                num_layers=self.d_auxl_num_layers,
                num_units=self.d_auxl_num_units).to(self.device) 
        
        self.g_loss = None
        self.d_loss = None
        self.d_loss_fake = None
        self.d_loss_real = None
        self.d_loss_gp = None
        if self.discriminator_auxl is not None:
            self.d_auxl_loss = None
            self.d_auxl_loss_fake = None
            self.d_auxl_loss_real = None
            self.d_auxl_loss_gp = None
            
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.g_lr,
                                            betas=(self.g_beta1, self.g_beta2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.d_lr,
                                            betas=(self.d_beta1, self.d_beta2))

        if self.discriminator_auxl is not None:
            self.d_auxl_optimizer = torch.optim.Adam(
                self.discriminator_auxl.parameters(),
                lr=self.d_auxl_lr, betas=(self.d_auxl_beta1, self.d_auxl_beta2))

        long_summary = False
        if self.verbose_summary and not long_summary:
            print(self.generator)
            print("====================================")
            print(self.discriminator)
            print("====================================")
            if self.discriminator_auxl is not None:
                print(self.discriminator_auxl)
                print("====================================")

        if self.verbose_summary and long_summary:        
            if self.data_auxl_signal is not None:
                summary(self.generator, input_size=[(self.batch_size, self.g_attribute_latent_dim),
                                                    (self.batch_size, self.g_attribute_latent_dim),
                                                    (self.batch_size, self.sample_time, self.g_feature_latent_dim),
                                                    (self.batch_size, self.sample_len * self.feature_dim2),
                                                    (self.batch_size, self.sample_time, self.auxl_signal_dim2)]) 
            else:
                summary(self.generator, input_size=[(self.batch_size, self.g_attribute_latent_dim),
                                                    (self.batch_size, self.g_attribute_latent_dim),
                                                    (self.batch_size, self.sample_time, self.g_feature_latent_dim),
                                                    (self.batch_size, self.sample_len * self.feature_dim2)])  
            summary(self.discriminator, input_size=[(self.batch_size, self.feature_dim1, self.feature_dim2),
                                                    (self.batch_size, self.attribute_dim1)])
            if self.discriminator_auxl is not None:
                summary(self.discriminator_auxl, input_size=(self.batch_size,
                                                             d_auxl_input_dim))
    # endregion

    def build_connection(self,
                         real_feature_pl_l, real_attribute_pl_l,
                         auxl_signal_pl_l,
                         g_real_attribute_input_noise_train_pl_l,
                         g_addi_attribute_input_noise_train_pl_l,
                         g_feature_input_noise_train_pl_l,
                         g_feature_input_data_train_pl_l,
                         ):
        """
        Build the connections between the real data placeholders and the generator.

        Args:
            real_feature_pl_l (list of torch.Tensor): List of real feature data tensors.
            real_attribute_pl_l (list of torch.Tensor): List of real attribute data tensors.
            auxl_signal_pl_l (list of torch.Tensor): List of auxiliary signal data tensors.
            g_real_attribute_input_noise_train_pl_l (list of torch.Tensor): List of noise tensors for real attribute input during training.
            g_addi_attribute_input_noise_train_pl_l (list of torch.Tensor): List of additional noise tensors for attribute input during training.
            g_feature_input_noise_train_pl_l (list of torch.Tensor): List of noise tensors for feature input during training.
            g_feature_input_data_train_pl_l (list of torch.Tensor): List of feature input data tensors during training.
        """
        self.local_batch_size = g_feature_input_noise_train_pl_l[0].size(0)
        self.real_attribute_mask_tensor = []
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                sub_mask_tensor = torch.ones(self.local_batch_size,
                                             self.data_attribute_outputs[i].dim).to(self.device)
            else:
                sub_mask_tensor = torch.zeros(self.local_batch_size,
                                              self.data_attribute_outputs[i].dim).to(self.device)
            self.real_attribute_mask_tensor.append(sub_mask_tensor)
        self.real_attribute_mask_tensor = torch.cat(self.real_attribute_mask_tensor, dim=1)

        # region Building generator
        g_output_feature_train_tf_l = []
        g_output_attribute_train_tf_l = []
        g_output_gen_flag_train_tf_l = []
        g_output_length_train_tf_l = []
        g_output_argmax_train_tf_l = []
        real_given_attribute = torch.cat(real_attribute_pl_l, dim=1)
        real_given_attribute = real_given_attribute[:, :self.real_attribute_dim1]
        
        for i in range(self.num_packing):
            (g_output_feature_train_tf, g_output_attribute_train_tf,
             g_output_gen_flag_train_tf, g_output_length_train_tf,
             g_output_argmax_train_tf) = \
                self.generator(
                    g_real_attribute_input_noise_train_pl_l[i] if self.train_with_attr_noise else None,
                    g_addi_attribute_input_noise_train_pl_l[i],
                    g_feature_input_noise_train_pl_l[i],
                    g_feature_input_data_train_pl_l[i],
                    train=True,
                    attribute=None if self.train_with_attr_noise else real_given_attribute,
                    auxl_signal=auxl_signal_pl_l[i],
                    )
            
            if self.train_fix_feature_net:
                g_output_feature_train_tf =\
                    torch.zeros_like(g_output_feature_train_tf).to(self.device)
                g_output_gen_flag_train_tf =\
                    torch.zeros_like(g_output_gen_flag_train_tf).to(self.device)
                g_output_attribute_train_tf *=\
                    self.real_attribute_mask_tensor

            g_output_feature_train_tf_l.append(g_output_feature_train_tf)
            g_output_attribute_train_tf_l.append(g_output_attribute_train_tf)
            g_output_gen_flag_train_tf_l.append(g_output_gen_flag_train_tf)
            g_output_length_train_tf_l.append(g_output_length_train_tf)
            g_output_argmax_train_tf_l.append(g_output_argmax_train_tf)
        
        self.g_output_feature_train_tf =\
            torch.cat(g_output_feature_train_tf_l, dim=1)
        self.g_output_attribute_train_tf =\
            torch.cat(g_output_attribute_train_tf_l, dim=1)
        # endregion
        
        # region Building discriminator (fake)
        self.d_fake_train_tf = self.discriminator(self.g_output_feature_train_tf, 
                                                  self.g_output_attribute_train_tf)
        self.g_output_attribute_train_tf_d_auxl = self.g_output_attribute_train_tf
        if self.discriminator_auxl is not None:
            self.d_auxl_fake_train_tf = self.discriminator_auxl(self.g_output_attribute_train_tf_d_auxl)
        # endregion

        # region Building discriminator (real)
        self.real_feature_pl = torch.cat(real_feature_pl_l, dim=1)
        self.real_attribute_pl = torch.cat(real_attribute_pl_l, dim=1)

        self.d_real_train_tf = self.discriminator(self.real_feature_pl,
                                                  self.real_attribute_pl)
        
        self.real_attribute_pl_d_auxl = self.real_attribute_pl
        if self.discriminator_auxl is not None:
            self.d_auxl_real_train_tf = self.discriminator_auxl(self.real_attribute_pl_d_auxl)
        # endregion    

    # region Losses
    def generator_loss(self):
        """
        Calculate the generator loss.

        Returns:
            torch.Tensor: The generator loss.
        """
        self.g_loss_d = -torch.mean(self.d_fake_train_tf)
        self.g_rmse_addi_attr = torch.sqrt(
            torch.mean((self.g_output_attribute_train_tf[:, -2:] - self.real_attribute_pl[:, -2:]) ** 2)
        )
        self.g_rmse_feat = torch.sqrt(
            torch.mean((self.g_output_feature_train_tf[:, :, 0] - self.real_feature_pl[:, :, 0]) ** 2)
        )
        if self.discriminator_auxl is not None:
            self.g_loss_d_auxl = -torch.mean(self.d_auxl_fake_train_tf)
            self.g_loss = (self.g_loss_d +
                           self.g_d_auxl_coe * self.g_loss_d_auxl)
        else:
            self.g_loss = self.g_loss_d
        
        if self.train_with_rmse_loss:
            self.g_loss = self.g_loss + self.g_rmse_coe * self.g_rmse_addi_attr    
        return self.g_loss
    
    def discriminator_loss(self):
        """
        Calculate the discriminator loss.

        Returns:
            torch.Tensor: The discriminator loss.
        """
        self.d_loss_fake = torch.mean(self.d_fake_train_tf)
        self.d_loss_real = -torch.mean(self.d_real_train_tf)

        alpha_dim2 = torch.rand(self.local_batch_size, 1, device=self.device)
        alpha_dim3 = alpha_dim2.unsqueeze(2)
        differences_input_feature = (self.g_output_feature_train_tf -
                                     self.real_feature_pl)
        interpolates_input_feature = (self.real_feature_pl +
                                      alpha_dim3 * differences_input_feature)
        differences_input_attribute = (self.g_output_attribute_train_tf -
                                       self.real_attribute_pl)
        interpolates_input_attribute = (self.real_attribute_pl +
                                        alpha_dim2 *
                                        differences_input_attribute)

        interpolates_input_feature.requires_grad_(True)
        interpolates_input_attribute.requires_grad_(True)
        d_interpolates = self.discriminator(interpolates_input_feature,
                                            interpolates_input_attribute)

        gradients = grad(
            outputs=d_interpolates,
            inputs=[interpolates_input_feature, interpolates_input_attribute],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )

        slopes1 = torch.sum(gradients[0] ** 2, dim=[1, 2])
        slopes2 = torch.sum(gradients[1] ** 2, dim=[1])
        slopes = torch.sqrt(slopes1 + slopes2 + 1e-8)

        self.d_loss_gp = torch.mean((slopes - 1) ** 2)
        self.d_loss = (self.d_loss_fake + self.d_loss_real +
                       self.d_gp_coe * self.d_loss_gp)
        
        return self.d_loss
    
    def discriminator_auxl_loss(self):
        """
        Calculate the auxiliary discriminator loss.

        Returns:
            torch.Tensor: The auxiliary discriminator loss.
        """
        self.d_auxl_loss_fake = torch.mean(self.d_auxl_fake_train_tf)
        self.d_auxl_loss_real = -torch.mean(self.d_auxl_real_train_tf)

        alpha_dim2 = torch.rand(self.local_batch_size, 1, device=self.device)

        differences_input_attribute = (self.g_output_attribute_train_tf_d_auxl -
                                       self.real_attribute_pl_d_auxl)
        interpolates_input_attribute = (self.real_attribute_pl_d_auxl +
                                        alpha_dim2 *
                                        differences_input_attribute)

        interpolates_input_attribute.requires_grad_(True)

        d_auxl_interpolates = self.discriminator_auxl(
            interpolates_input_attribute)

        gradients = grad(
            outputs=d_auxl_interpolates,
            inputs=[interpolates_input_attribute],
            grad_outputs=torch.ones_like(d_auxl_interpolates),
            create_graph=True,
            retain_graph=True
        )

        slopes1 = torch.sum(gradients[0] ** 2, dim=[1])
        slopes = torch.sqrt(slopes1 + 1e-8)

        self.d_auxl_loss_gp = torch.mean((slopes - 1) ** 2)

        self.d_auxl_loss = (self.d_auxl_loss_fake + self.d_auxl_loss_real +
                            self.d_auxl_gp_coe * self.d_auxl_loss_gp)
        
        return self.d_auxl_loss
    # endregion

    # region SummaryWriter, save, load
    def build_summary(self, global_id):
        """
        Build summary for TensorBoard.

        Args:
            global_id (int): The global step id (batch id) for TensorBoard logging.
        """
        self.writer.add_scalar("loss/g/d", self.g_loss_d.item(), global_step=global_id)
        if self.discriminator_auxl is not None:
            self.writer.add_scalar("loss/g/d_auxl", self.g_loss_d_auxl.item(), global_step=global_id)
        self.writer.add_scalar("loss/g_rmse", self.g_rmse_addi_attr.item(), global_step=global_id)
        self.writer.add_scalar("loss/g_rmse_feat", self.g_rmse_feat.item(), global_step=global_id)
        self.writer.add_scalar("loss/g", self.g_loss.item(), global_step=global_id)

        self.writer.add_scalar("loss/d/fake", self.d_loss_fake.item(), global_step=global_id)
        self.writer.add_scalar("loss/d/real", self.d_loss_real.item(), global_step=global_id)
        self.writer.add_scalar("loss/d/gp", self.d_loss_gp.item(), global_step=global_id)
        self.writer.add_scalar("loss/d", self.d_loss.item(), global_step=global_id)
        self.writer.add_scalar("d/fake", torch.mean(self.d_fake_train_tf).item(), global_step=global_id)
        self.writer.add_scalar("d/real", torch.mean(self.d_real_train_tf).item(), global_step=global_id)

        if self.discriminator_auxl is not None:
            self.writer.add_scalar("loss/d_auxl/fake", self.d_auxl_loss_fake.item(), global_step=global_id)
            self.writer.add_scalar("loss/d_auxl/real", self.d_auxl_loss_real.item(), global_step=global_id)
            self.writer.add_scalar("loss/d_auxl/gp", self.d_auxl_loss_gp.item(), global_step=global_id)
            self.writer.add_scalar("loss/d_auxl", self.d_auxl_loss.item(), global_step=global_id)
            self.writer.add_scalar("d_auxl/fake", torch.mean(self.d_auxl_fake_train_tf).item(), global_step=global_id)
            self.writer.add_scalar("d_auxl/real", torch.mean(self.d_auxl_real_train_tf).item(), global_step=global_id)
        
        self.writer.add_scalars("loss_combined", {
            "g": self.g_loss.item(),
            "d": self.d_loss.item()
        }, global_step=global_id)
        
    def save(self, global_id, checkpoint_dir=None):
        """
        Save the model checkpoint.

        Args:
            global_id (int): The global step id (batch id) for saving the checkpoint.
            checkpoint_dir (str): The directory to save the checkpoint.
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir,
                                       f"{self.model_name}_{global_id}.pth")
        torch.save({
            'global_id': global_id,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'discriminator_auxl_state_dict': self.discriminator_auxl.state_dict() if self.discriminator_auxl else None,
            'd_auxl_optimizer_state_dict': self.d_auxl_optimizer.state_dict() if self.d_auxl_optimizer else None
        }, checkpoint_file)

    def load(self, checkpoint_dir=None, checkpoint_file=None):
        """
        Load the model checkpoint.

        Args:
            checkpoint_dir (str): The directory to load the checkpoint from.
            checkpoint_file (str): The checkpoint file to load.
        """    
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if checkpoint_file is None:
            checkpoint_file = os.path.join(checkpoint_dir, self.checkpoint_file_last)
        else:
            checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            if self.discriminator_auxl:
                self.discriminator_auxl.load_state_dict(checkpoint['discriminator_auxl_state_dict'])
                self.d_auxl_optimizer.load_state_dict(checkpoint['d_auxl_optimizer_state_dict'])
            global_id = checkpoint['global_id']
            return global_id
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")
    # endregion

    # region Discriminating and sampling
    def discriminate_from(self, real_features, real_attributes):
        """
        Discriminate the real data.

        Args:
            real_features (list of torch.Tensor): List of real feature data tensors.
            real_attributes (list of torch.Tensor): List of real attribute data tensors.

        Returns:
            np.ndarray: discriminator results.
        """
        results = []
        round_ = int(math.ceil(float(real_features[0].shape[0]) / self.batch_size))
        for i in range(round_):
            batch_results = []
            for j in range(self.num_packing):
                batch_data_feature = real_features[j][i * self.batch_size:(i + 1) * self.batch_size]
                batch_data_attribute = real_attributes[j][i * self.batch_size:(i + 1) * self.batch_size]

                with torch.no_grad():
                    batch_result = self.discriminator(batch_data_feature.to(self.device), 
                                                      batch_data_attribute.to(self.device))
                batch_results.append(batch_result.cpu().numpy())
            results.append(np.concatenate(batch_results, axis=0))

        results = np.concatenate(results, axis=0)
        return results

    def sample_from(self, real_attribute_input_noise,
                    addi_attribute_input_noise, feature_input_noise,
                    feature_input_data, given_attribute=None,
                    auxl_signal=None,
                    return_gen_flag_feature=False):
        """
        Sample (generate data) from the generator.

        Args:
            real_attribute_input_noise (torch.Tensor): Real attribute input noise tensor.
            addi_attribute_input_noise (torch.Tensor): Additional attribute input noise tensor.
            feature_input_noise (torch.Tensor): Feature input noise tensor.
            feature_input_data (torch.Tensor): Feature input data tensor.
            given_attribute (torch.Tensor): Given attribute tensor.
            auxl_signal (torch.Tensor): Auxiliary signal tensor
            return_gen_flag_feature (bool): Whether to return the generation flag feature.

        Returns:
            np.ndarray: Generated features.
            np.ndarray: Generated attributes.
            np.ndarray: Generated generation flags.
            np.ndarray: Generated lengths (timesteps) of each sample feature.
        """
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(
            math.ceil(float(feature_input_noise.shape[0]) / self.batch_size))
        for i in range(round_):
            if given_attribute is None:
                with torch.no_grad():
                    sub_features, sub_attributes, sub_gen_flags, sub_lengths, _ = self.generator(
                        real_attribute_input_noise[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        addi_attribute_input_noise[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        feature_input_noise[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        feature_input_data[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        train=False,
                        auxl_signal=auxl_signal[i * self.batch_size:(i + 1) * self.batch_size].to(self.device)
                                    if auxl_signal is not None else None
                        )
            else:
                with torch.no_grad():
                    sub_features, sub_attributes, sub_gen_flags, sub_lengths, _ = self.generator(
                        None,
                        addi_attribute_input_noise[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        feature_input_noise[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        feature_input_data[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        train=False,
                        attribute=given_attribute[i * self.batch_size:(i + 1) * self.batch_size].to(self.device),
                        auxl_signal=auxl_signal[i * self.batch_size:(i + 1) * self.batch_size].to(self.device)
                                    if auxl_signal is not None else None
                        )
            features.append(sub_features.cpu().numpy())
            attributes.append(sub_attributes.cpu().numpy())
            gen_flags.append(sub_gen_flags.cpu().numpy())
            lengths.append(sub_lengths.cpu().numpy())

        features = np.concatenate(features, axis=0)
        attributes = np.concatenate(attributes, axis=0)
        gen_flags = np.concatenate(gen_flags, axis=0)
        lengths = np.concatenate(lengths, axis=0)

        if not return_gen_flag_feature:
            features = np.delete(features, self.gen_flag_dims, axis=2)

        assert len(gen_flags.shape) == 3
        assert gen_flags.shape[2] == 1
        gen_flags = gen_flags[:, :, 0]

        return features, attributes, gen_flags, lengths
    # endregion

    # region Noise and Real Data initialisation
    def gen_attribute_input_noise(self, num_sample):
        """
        Generate noise for attribute input.

        Args:
            num_sample (int): Number of samples to generate noise for.

        Returns:
            torch.Tensor: Generated noise tensor for attribute input.
        """
        return torch.randn(num_sample,
                           self.g_attribute_latent_dim).to(self.device)

    def gen_feature_input_noise(self, num_sample, length):
        """
        Generate noise for feature input.

        Args:
            num_sample (int): Number of samples to generate noise for.
            length (int): Length of the noise sequence.

        Returns:
            torch.Tensor: Generated noise tensor for feature input.
        """
        return torch.randn(num_sample, length,
                           self.g_feature_latent_dim).to(self.device)

    def gen_feature_input_data_free(self, num_sample):
        """
        Generate free feature input data.

        Args:
            num_sample (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated feature input data tensor.
        """
        return torch.zeros(num_sample,
                           self.sample_len * self.feature_dim2, 
                           dtype=torch.float32).to(self.device)

    def gen_feature_input_data_teacher(self, num_sample):
        """
        Generate feature input data for teacher forcing.

        Args:
            num_sample (int): Number of samples to generate.

        Returns:
            tuple: A tuple containing the input data, ground truth features, 
               ground truth attributes, and ground truth lengths.
        """
        id_ = np.random.choice(self.feature_dim0,
                               num_sample, replace=False)
        data_feature_ori = torch.tensor(self.data_feature[id_, :, :],
                                        dtype=torch.float32).to(self.device)
        data_feature = data_feature_ori.view(num_sample, self.sample_time,
                                             self.sample_len *
                                             self.feature_dim2)
        input_ = torch.cat(
            [torch.zeros(num_sample, 1,
                         self.sample_len * self.feature_dim2,
                         dtype=torch.float32).to(self.device),
             data_feature[:, :-1, :]],
            dim=1)
        ground_truth_feature = data_feature_ori
        ground_truth_length = torch.sum(
            torch.tensor(self.data_gen_flag[id_, :, :], 
                         dtype=torch.float32).to(self.device), dim=(1, 2))
        ground_truth_attribute = torch.tensor(
            self.data_attribute[id_, :], dtype=torch.float32).to(self.device)
        if self.data_auxl_signal is not None:
            auxl_signal = torch.tensor(self.data_auxl_signal[id_, :, :], 
                                       dtype=torch.float32).to(self.device)
        else:
            auxl_signal = None
        return (input_, ground_truth_feature,
                ground_truth_attribute, ground_truth_length,
                auxl_signal)
    
    def noise_gen_training(self):
        """
        Generate noise inputs for training. 
        Combination of gen_attribute_input_noise, gen_feature_input_noise, gen_feature_input_data_free

        Returns:
            list: A list containing noise tensors for real attribute input, 
              additional attribute input, feature input, and feature input data.
        """
        noise_inputs = []
        g_real_attribute_input_noise_train_pl_l = []
        g_addi_attribute_input_noise_train_pl_l = []
        g_feature_input_noise_train_pl_l = []
        g_feature_input_data_train_pl_l = []

        for i in range(self.num_packing):
            batch_real_attribute_input_noise = \
                self.gen_attribute_input_noise(self.batch_size)
            batch_addi_attribute_input_noise = \
                self.gen_attribute_input_noise(self.batch_size)
            batch_feature_input_noise = \
                self.gen_feature_input_noise(
                    self.batch_size, self.sample_time)
            batch_feature_input_data = \
                self.gen_feature_input_data_free(self.batch_size)
            g_real_attribute_input_noise_train_pl_l.append(batch_real_attribute_input_noise)
            g_addi_attribute_input_noise_train_pl_l.append(batch_addi_attribute_input_noise)
            g_feature_input_noise_train_pl_l.append(batch_feature_input_noise)
            g_feature_input_data_train_pl_l.append(batch_feature_input_data)
        
        noise_inputs.append(g_real_attribute_input_noise_train_pl_l)
        noise_inputs.append(g_addi_attribute_input_noise_train_pl_l)
        noise_inputs.append(g_feature_input_noise_train_pl_l)
        noise_inputs.append(g_feature_input_data_train_pl_l)

        return noise_inputs
    
    def realdata_gen_training(self, data_id, batch_id):
        """
        Batch real data inputs for training.

        Args:
            data_id (np.ndarray): Array of data indices for the batch.
            batch_id (int): The batch index.

        Returns:
            list: A list containing real feature, attribute, and auxl_signal tensors.
        """
        real_inputs = []
        real_feature_pl_l = []
        real_attribute_pl_l = []
        real_auxl_signal_pl_l = []
        for i in range(self.num_packing):
            batch_data_id = data_id[batch_id * self.batch_size:
                                    (batch_id + 1) * self.batch_size,
                                    i]
            batch_data_feature = torch.tensor(self.data_feature[batch_data_id],
                                            dtype=torch.float32).to(self.device)
            batch_data_attribute = torch.tensor(self.data_attribute[batch_data_id],
                                                dtype=torch.float32).to(self.device)
            if self.data_auxl_signal is not None:
                batch_data_auxl_signal = torch.tensor(self.data_auxl_signal[batch_data_id],
                                                    dtype=torch.float32).to(self.device)
                
            else:
                batch_data_auxl_signal = None
            if self.train_fix_feature_net:
                batch_data_feature = torch.zeros_like(batch_data_feature).to(self.device)
                batch_data_attribute *= self.real_attribute_mask_tensor
            real_feature_pl_l.append(batch_data_feature)
            real_attribute_pl_l.append(batch_data_attribute)
            real_auxl_signal_pl_l.append(batch_data_auxl_signal)
        real_inputs.append(real_feature_pl_l)
        real_inputs.append(real_attribute_pl_l)
        real_inputs.append(real_auxl_signal_pl_l)
        return real_inputs
    # endregion

    # region Visualisation
    def visualize(self, epoch_id, batch_id, global_id,
                  vis_free=True, vis_teacher=False, vis_givenatt=True):
        """
        Visualise the generated samples and save them.

        Args:
            epoch_id (int): The current epoch id.
            batch_id (int): The current batch id.
            global_id (int): The global step id (batch id) for TensorBoard logging.
            vis_free (bool): Whether to visualise the free samples.
            vis_teacher (bool): Whether to visualise the teacher samples.
            vis_givenatt (bool): Whether to visualise the samples with given real attributes.
        """
        def sub1(features, attributes, lengths,
                 ground_truth_features, ground_truth_attributes,
                 ground_truth_lengths, type_,
                 do_draw_feature=True, do_draw_attribute=False):
            """
            Save and visualise the generated samples.

            Args:
                features (np.ndarray): Generated features.
                attributes (np.ndarray): Generated attributes.
                lengths (np.ndarray): Generated lengths (timesteps) of each sample feature.
                ground_truth_features (np.ndarray): Ground truth features.
                ground_truth_attributes (np.ndarray): Ground truth attributes.
                ground_truth_lengths (np.ndarray): Ground truth lengths.
                type_ (str): Type of the generated samples (e.g., 'free', 'teacher', 'givenatt').
                do_draw_feature (bool): Whether to draw the feature.
                do_draw_attribute (bool): Whether to draw the attribute.
            """
            file_path = os.path.join(
                self.sample_dir,
                "epoch_id-{},batch_id-{},global_id-{},type-{},samples.npz"
                .format(epoch_id, batch_id, global_id, type_))
            np.savez_compressed(file_path,
                                features=features,
                                attributes=attributes,
                                lengths=lengths,
                                ground_truth_features=ground_truth_features if ground_truth_features is not None else None,
                                ground_truth_attributes=ground_truth_attributes if ground_truth_attributes is not None else None,
                                ground_truth_lengths=ground_truth_lengths if ground_truth_lengths is not None else None)

            if do_draw_feature:
                file_path = os.path.join(
                    self.sample_dir,
                    "epoch_id-{},batch_id-{},global_id-{},type-{},feature"
                    .format(epoch_id, batch_id, global_id, type_))
                if ground_truth_features is None:
                    draw_feature(
                        features,
                        lengths,
                        self.data_feature_outputs,
                        file_path)
                else:
                    draw_feature(
                        np.concatenate([features, ground_truth_features], axis=0),
                        np.concatenate([lengths, ground_truth_lengths], axis=0),
                        self.data_feature_outputs,
                        file_path)

            if do_draw_attribute:
                file_path = os.path.join(
                    self.sample_dir,
                    "epoch_id-{},batch_id-{},global_id-{},type-{},attribute"
                    .format(epoch_id, batch_id, global_id, type_))
                if ground_truth_features is None:
                    draw_attribute(
                        attributes,
                        self.data_attribute_outputs,
                        file_path)
                else:
                    draw_attribute(
                        np.concatenate([attributes, ground_truth_attributes],
                                    axis=0),
                        self.data_attribute_outputs,
                        file_path)

        real_attribute_input_noise = self.gen_attribute_input_noise(
            self.vis_num_sample)
        addi_attribute_input_noise = self.gen_attribute_input_noise(
            self.vis_num_sample)
        feature_input_noise = self.gen_feature_input_noise(
            self.vis_num_sample, self.sample_time)

        feature_input_data_free = self.gen_feature_input_data_free(
            self.vis_num_sample)
        (feature_input_data_teacher, ground_truth_feature,
         ground_truth_attribute, ground_truth_length, auxl_signal) = \
            self.gen_feature_input_data_teacher(self.vis_num_sample)
        
        ground_truth_feature = ground_truth_feature.cpu().numpy()
        ground_truth_attribute = ground_truth_attribute.cpu().numpy()
        ground_truth_length = ground_truth_length.cpu().numpy()

        if vis_free:
            features, attributes, gen_flags, lengths = self.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, feature_input_data_free,
                return_gen_flag_feature=True,
                auxl_signal=auxl_signal)
            sub1(features, attributes, lengths, None, None, None, "free")

        if vis_teacher:       
            features, attributes, gen_flags, lengths = self.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, feature_input_data_teacher,
                return_gen_flag_feature=True,
                auxl_signal=auxl_signal)
            sub1(features, attributes, lengths,
                 ground_truth_feature, ground_truth_attribute, ground_truth_length,
                 "teacher")
        
        if vis_givenatt:
            given_attribute = ground_truth_attribute[:, :self.real_attribute_dim1]
            given_attribute = torch.tensor(given_attribute, dtype=torch.float32).to(self.device)
            features, attributes, gen_flags, lengths = self.sample_from(
                None, addi_attribute_input_noise, 
                feature_input_noise, feature_input_data_free,
                given_attribute=given_attribute,
                return_gen_flag_feature=True,
                auxl_signal=auxl_signal)
            sub1(features, attributes, lengths, 
                 ground_truth_feature, ground_truth_attribute, ground_truth_length, 
                 "givenatt")
    # endregion

    # region Train
    def train(self):
        """Train the model.
        """
        if self.restore and self.checkpoint_files:
            restore_global_id = self.load()
            print("Loaded from global_id {}".format(restore_global_id))
        else:
            restore_global_id = -1

        batch_num = self.feature_dim0 // self.batch_size
        global_id = 0

        for epoch_id in tqdm(range(self.epoch)):
            self.generator.train()
            self.discriminator.train()
            if self.discriminator_auxl is not None:
                self.discriminator_auxl.train()

            data_id = np.random.choice(
                self.feature_dim0,
                size=(self.feature_dim0, self.num_packing))

            if global_id > restore_global_id:
                if ((epoch_id + 1) % self.epoch_checkpoint_freq == 0 or
                        epoch_id == self.epoch - 1):
                    with open(self.time_path, "a") as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write("epoch {} starts: {}\n".format(epoch_id, time))

            for batch_id in range(batch_num):
                real_inputs = self.realdata_gen_training(data_id, batch_id)
                if global_id > restore_global_id:
                    for _ in range(self.d_rounds):
                        self.d_optimizer.zero_grad()
                        noise_inputs = self.noise_gen_training()
                        self.build_connection(*real_inputs, *noise_inputs)
                        d_loss = self.discriminator_loss()
                        d_loss.backward()
                        self.d_optimizer.step()

                        if self.discriminator_auxl is not None:
                            self.d_auxl_optimizer.zero_grad()
                            noise_inputs = self.noise_gen_training()
                            self.build_connection(*real_inputs, *noise_inputs)
                            d_auxl_loss = self.discriminator_auxl_loss()
                            d_auxl_loss.backward()
                            self.d_auxl_optimizer.step()

                    for _ in range(self.g_rounds):
                        self.g_optimizer.zero_grad()
                        noise_inputs = self.noise_gen_training()
                        self.build_connection(*real_inputs, *noise_inputs)
                        g_loss = self.generator_loss()
                        g_loss.backward()
                        self.g_optimizer.step()

                    self.build_summary(global_id)
                    if (batch_id + 1) % self.vis_freq == 0:
                        self.visualize(epoch_id, batch_id, global_id)

                global_id += 1

            if global_id - 1 > restore_global_id:
                if ((epoch_id + 1) % self.epoch_checkpoint_freq == 0 or
                        epoch_id == self.epoch - 1):
                    self.visualize(epoch_id, -1, global_id - 1)
                    self.save(global_id - 1)
                    with open(self.time_path, "a") as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write("epoch {} ends: {}\n".format(epoch_id, time))
    # endregion

    # region Generate
    def generate(self,
                 checkpoint_file=None,
                 generate_gif=False,
                 gif_gen_ids=None,
                 gif_xlim=None,
                 gen_stochastic=False,
                 gen_save_name=None,
                 seed=56):
        """Generate data using the trained model.

        Args:
            checkpoint_file (str): Specify checkpoint file to load.
            generate_gif (bool): Whether to generate GIFs (visualisation only).
            gif_gen_ids (list): List of indices to generate GIFs for.
            gif_xlim (list): List of x-axis limits for the GIFs.
            gen_stochastic (bool): Whether to generate the model in stochastic mode.
            gen_save_name (str): Name of the generated data file, if None default to dataset name.
            seed (int): Random seed for reproducibility.

        Raises:
            FileNotFoundError: If no checkpoint files are found.
        """
        if not gen_stochastic:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        if checkpoint_file is None and not self.checkpoint_files:
            raise FileNotFoundError("No checkpoint files found")

        num_real_attribute = self.real_attribute_dim1
        if generate_gif:
            gen_ids = np.random.choice(
                self.feature_dim0, 2, replace=False) if gif_gen_ids is None else gif_gen_ids
            assert self.feature_dim2 == 1, "Only support 1D feature for GIF generation"
            # TODO: implement multiple feature dimensions for GIF generation

        elif (self.generate_num_sample is None or self.generate_num_sample > self.feature_dim0):
            self.generate_num_sample = self.feature_dim0
            gen_ids = [i for i in range(self.generate_num_sample)]              
        else:
            gen_ids = np.random.choice(self.feature_dim0,
                                       self.generate_num_sample, replace=False)
        
        addi_attribute_input_noise = self.gen_attribute_input_noise(
            self.feature_dim0)
        feature_input_noise = self.gen_feature_input_noise(
            self.feature_dim0, self.sample_time)
        feature_input_data = self.gen_feature_input_data_free(
            self.feature_dim0)
        
        if self.generate_with_attr_noise:
            real_attribute_input_noise = self.gen_attribute_input_noise(
                self.feature_dim0)
            given_attribute = None
        else:
            real_attribute_input_noise = None
            given_attribute = torch.tensor(self.data_attribute_real,
                                           dtype=torch.float32).to(self.device)
        if self.data_auxl_signal is not None and self.generate_with_auxl_signal:
            auxl_signal = torch.tensor(self.data_auxl_signal,
                                       dtype=torch.float32).to(self.device)
        else:
            auxl_signal = None

        if not generate_gif:
            if checkpoint_file is not None:
                restore_global_id = self.load(checkpoint_file=checkpoint_file)
                print("Loaded from global_id {}".format(restore_global_id))
            elif self.checkpoint_files:
                restore_global_id = self.load()
                print("Loaded from global_id {}".format(restore_global_id))           

            features, attributes, gen_flags, lengths = self.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, feature_input_data, given_attribute,
                return_gen_flag_feature=True,
                auxl_signal=auxl_signal)
            
            features = features[:, :, :-2]

            features_normed = features.copy()
            attributes_normed = attributes.copy()
            
            features, attributes = renormalize_per_sample(
                    features, attributes,
                    self.data_attribute_scalers,
                    self.data_feature_outputs,
                    self.data_attribute_outputs,
                    gen_flags,
                    num_real_attribute=num_real_attribute)
            
            features = features[gen_ids]
            attributes = attributes[gen_ids]
            lengths = lengths[gen_ids]

            if gen_save_name is None:
                gen_save_name = self.dataset_name

            file_path = os.path.join(self.generate_dir,
                                     f"data_generated_{gen_save_name}.npz")
            
            np.savez_compressed(file_path,
                                features=features,
                                attributes=attributes,
                                features_normed=features_normed,
                                attributes_normed=attributes_normed,
                                lengths=lengths) 
            
        else:
            assert len(self.checkpoint_files) > 1, "Need at least 2 checkpoint files to generate GIFs"

            features_compiled = []
            feature_ground_truth = self.data_feature_ground_truth[gen_ids]
            feature_ground_truth_compiled = [feature_ground_truth] * len(self.checkpoint_files)

            for i, checkpoint_file in enumerate(self.checkpoint_files):
                self.load(checkpoint_file=checkpoint_file)

                features, attributes, gen_flags, lengths = self.sample_from(
                    real_attribute_input_noise, addi_attribute_input_noise,
                    feature_input_noise, feature_input_data, given_attribute,
                    return_gen_flag_feature=True,
                    auxl_signal=auxl_signal)
                
                features = features[:, :, :-2]

                features, _ = renormalize_per_sample(
                        features, attributes,
                        self.data_attribute_scalers,
                        self.data_feature_outputs,
                        self.data_attribute_outputs,
                        gen_flags,
                        num_real_attribute=num_real_attribute)
                
                features = features[gen_ids]
                features_compiled.append(features)

            create_gif(data=features_compiled,
                       ground_truth=feature_ground_truth_compiled,
                       dataset_name=self.dataset_name,
                       xlim=gif_xlim,
                       path=self.generate_dir,)   
    # endregion
import os
import sys
import argparse
sys.path.append(".")

from gan import output
sys.modules["output"] = output

from gan.demaendgan_time import demAEndGANTime
from gan.load_data import DataLoader
from gan.util import create_path, str2bool

def main(args):
    data_path = os.path.join(args.data_path, args.data_folder)
    checkpoint_dir = create_path(args.run_folder_name, args.checkpoint_folder)
    sample_dir = create_path(args.run_folder_name, args.sample_folder)
    time_path = create_path(args.run_folder_name, args.time_file)
    generate_dir = create_path(args.run_folder_name, args.generate_folder)

    data_loader = DataLoader(args, data_path)

    (data_feature, data_attribute, data_gen_flag, 
     data_feature_outputs, data_attribute_outputs,
     real_attribute_mask, data_attribute_real,
     data_attribute_scalers,
     data_auxl_signal, data_feature_ground_truth
     ) = \
        data_loader.load_data()
    
    gan = demAEndGANTime(args,
                         checkpoint_dir, sample_dir, time_path, generate_dir,
                         data_feature, data_attribute, data_gen_flag,
                         data_feature_outputs, data_attribute_outputs,
                         real_attribute_mask,
                         data_attribute_real,
                         data_attribute_scalers,
                         data_auxl_signal, data_feature_ground_truth
                         )
    if args.train_mode:
        gan.train()
    else:
        gan.generate(generate_gif=args.generate_gif_training,
                     gif_gen_ids=args.generate_gif_sample_ids,
                     gif_xlim=args.generate_gif_xlim,
                     gen_stochastic=args.generate_stochastic,
                     gen_save_name=args.generate_save_name,
                     seed=args.generate_random_seed,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", default=True, type=str2bool, help="train mode or generate mode")

    parser.add_argument("--data_path", default='_data', type=str, help="data directory name")
    parser.add_argument("--data_folder", default='sine_cosine', type=str, help="data folder name")
    parser.add_argument("--run_folder_name", default='test', type=str, help="folder name containing training/generating outputs")
    parser.add_argument("--checkpoint_folder", default='checkpoint', type=str, help="checkpoint directory name")
    parser.add_argument("--sample_folder", default='sample', type=str, help="sample directory name")
    parser.add_argument("--time_file", default='time.txt', type=str, help="time file path")
    parser.add_argument("--generate_folder", default='generate', type=str, help="generate directory name")
    parser.add_argument("--dataset_name", default='train', type=str, help="dataset suffix name")
    parser.add_argument("--model_name", default='model', type=str, help="model name")
                        
    parser.add_argument("--epoch", default=400, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=100, type=int, help="batch size (min=2)")
    parser.add_argument("--sample_len", default=5, type=int, help="length of timesteps generated each RNN rollout")
    parser.add_argument("--num_packing", default=1, type=int, help="packing degree in PacGAN")

    parser.add_argument("--d_rounds", default=1, type=int, help="number of discriminator training rounds")
    parser.add_argument("--g_rounds", default=1, type=int, help="number of generator training rounds")
    
    parser.add_argument("--d_gp_coe", default=10.0, type=float, help="coefficient of gradient penalty in Wasserstein GAN for dsicriminator")
    parser.add_argument("--d_auxl_gp_coe", default=10.0, type=float, help="coefficient of gradient penalty in Wasserstein GAN for auxiliary discriminator")
    parser.add_argument("--g_d_auxl_coe", default=1.0, type=float, help="coefficient of auxiliary discriminator loss in generator loss")
    parser.add_argument("--g_rmse_coe", default=1.0, type=float, help="coefficient of RMSE additional (max-min) loss in generator loss")

    parser.add_argument("--g_lr", default=0.001, type=float, help="learning rate of generator")
    parser.add_argument("--g_beta1", default=0.5, type=float, help="beta1 of Adam optimiser for generator")
    parser.add_argument("--g_beta2", default=0.999, type=float, help="beta2 of Adam optimiser for generator")
    parser.add_argument("--g_feature_feedback", default=False, type=str2bool, help="train generator RNN with feedback from previous rollout")
    parser.add_argument("--g_feature_noise", default=True, type=str2bool, help="train generator RNN with random noise")
    parser.add_argument("--g_attribute_latent_dim", default=5, type=int, help="dimension of attribute latent variable")
    parser.add_argument("--g_attribute_num_layers", default=3, type=int, help="number of layers in attribute generator")
    parser.add_argument("--g_attribute_num_units", default=100, type=int, help="number of units in attribute generator")
    parser.add_argument("--g_feature_latent_dim", default=5, type=int, help="dimension of feature latent variable")
    parser.add_argument("--g_feature_num_layers", default=1, type=int, help="number of layers in feature generator")
    parser.add_argument("--g_feature_num_units", default=100, type=int, help="number of units in feature generator")

    parser.add_argument("--d_lr", default=0.001, type=float, help="learning rate of discriminator")
    parser.add_argument("--d_beta1", default=0.5, type=float, help="beta1 of Adam optimiser for discriminator")
    parser.add_argument("--d_beta2", default=0.999, type=float, help="beta2 of Adam optimiser for discriminator")
    parser.add_argument("--d_num_layers", default=5, type=int, help="number of layers in discriminator")
    parser.add_argument("--d_num_units", default=200, type=int, help="number of units in discriminator")

    parser.add_argument("--d_auxl_lr", default=0.001, type=float, help="learning rate of auxiliary discriminator")
    parser.add_argument("--d_auxl_beta1", default=0.5, type=float, help="beta1 of Adam optimiser for auxiliary discriminator")
    parser.add_argument("--d_auxl_beta2", default=0.999, type=float, help="beta2 of Adam optimiser for auxiliary discriminator")
    parser.add_argument("--d_auxl_num_layers", default=5, type=int, help="number of layers in auxiliary discriminator")
    parser.add_argument("--d_auxl_num_units", default=200, type=int, help="number of units in auxiliary discriminator")

    parser.add_argument("--dp_noise_multiplier", default=None, type=float, help="noise multiplier for differential privacy")
    parser.add_argument("--dp_l2_norm_clip", default=None, type=float, help="l2 norm clip for differential privacy")
    parser.add_argument("--dp_delta", default=1e-5, type=float, help="delta for differential privacy")

    parser.add_argument("--addi_attr_power_transform", default=True, type=str2bool, help="power transform additional (min-max) attribute")

    parser.add_argument("--train_with_attr_noise", default=False, type=str2bool, help="train with noise as real attribute")
    parser.add_argument("--train_with_auxl_signal", default=True, type=str2bool, help="train with auxiliary signal")
    parser.add_argument("--train_with_discriminator_auxl", default=True, type=str2bool, help="train with auxiliary discriminator")
    parser.add_argument("--train_with_rmse_loss", default=True, type=str2bool, help="train with additional (max-min) attribute RMSE loss")
    parser.add_argument("--train_fix_feature_net", default=False, type=str2bool, help="fix feature network during training")
    
    parser.add_argument("--generate_with_attr_noise", default=False, type=str2bool, help="generate with noise as real attribute")
    parser.add_argument("--generate_with_auxl_signal", default=True, type=str2bool, help="generate with auxiliary signal")
    parser.add_argument("--generate_num_sample", default=1e+10, type=int, help="number of samples to generate")
    parser.add_argument("--generate_random_seed", default=56, type=int, help="random seed for generating samples")
    parser.add_argument("--generate_gif_training", default=False, type=str2bool, help="generate gif of results of training")
    parser.add_argument("--generate_gif_sample_ids", default=None, nargs='+', type=int, help="sample ids to generate gif")
    parser.add_argument("--generate_gif_xlim", default=None, type=int, help="x-axis limit of gif")
    parser.add_argument("--generate_stochastic", default=False, type=str2bool, help="generate samples stochastically")
    parser.add_argument("--generate_save_name", default=None, type=str, help="name of saved generated files")

    parser.add_argument("--restore", default=True, type=str2bool, help="restore from checkpoint")
    parser.add_argument("--verbose_summary", default=False, type=str2bool, help="print summary of dataset and model")

    parser.add_argument("--epoch_checkpoint_freq", default=1, type=int, help="checkpoint frequency in epochs")
    parser.add_argument("--vis_freq", default=200, type=int, help="visualisation frequency in batches")
    parser.add_argument("--vis_num_sample", default=2, type=int, help="number of samples to visualise (min=2)")

    args = parser.parse_args()

    if args.train_mode:
        assert args.dataset_name == 'train', "if training, dataset_name must be 'train'"
    assert args.batch_size >= 2, "batch_size must be at least 2"
    assert args.vis_num_sample >= 2, "vis_num_sample must be at least 2"
    if not args.train_mode:
        assert args.generate_num_sample >= 2, "generate_num_sample must be at least 2"
    main(args)

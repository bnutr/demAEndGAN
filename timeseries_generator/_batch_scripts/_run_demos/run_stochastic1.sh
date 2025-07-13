#! /bin/sh

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random1 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random2 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random3 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random4 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random5 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random6 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random7 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random8 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random9 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random10 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random11 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random12 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random13 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random14 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random15 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random16 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random17 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random18 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random19 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random20 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random21 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random22 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random23 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random24 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random25 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random26 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random27 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random28 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random29 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random30 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random31 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random32 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random33 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random34 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random35 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random36 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random37 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random38 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random39 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random40 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random41 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random42 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random43 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random44 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random45 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random46 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random47 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random48 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random49 ;

python gan/main.py --train_mode False \
    --data_folder d_small_kg3_nooutlier_auxl_1m --run_folder_name run_d_small_kg3_nooutlier_auxl_1m_rmse \
    --dataset_name test --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --addi_attr_power_transform False \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 \
    --generate_stochastic True --generate_save_name random50 ;



#! /bin/sh

python gan/main.py --train_mode True \
    --data_folder d_small_kg2_nooutlier_auxl_2m --run_folder_name run_d_small_kg2_nooutlier_auxl_2m_rmse \
    --dataset_name train --model_name model \
    --epoch 400 --batch_size 100 --sample_len 24 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --train_with_auxl_signal True \
    --addi_attr_power_transform False \
    --g_rmse_coe 1 --train_with_rmse_loss True \
    --epoch_checkpoint_freq 1 --vis_freq 300 --vis_num_sample 2 ;

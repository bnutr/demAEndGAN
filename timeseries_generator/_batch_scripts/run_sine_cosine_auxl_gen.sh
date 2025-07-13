#! /bin/sh

python gan/main.py --train_mode False \
    --data_folder sine_cosine_auxl --run_folder_name sine_cosine_auxl_test \
    --dataset_name train --model_name model \
    --epoch 150 --batch_size 100 --sample_len 5 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --train_with_auxl_signal True \
    --epoch_checkpoint_freq 1 --vis_freq 200 --vis_num_sample 2 ;

python gan/main.py --train_mode False \
    --data_folder sine_cosine_auxl --run_folder_name sine_cosine_auxl_test \
    --dataset_name test --model_name model \
    --epoch 150 --batch_size 100 --sample_len 5 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --train_with_auxl_signal True \
    --epoch_checkpoint_freq 1 --vis_freq 200 --vis_num_sample 2 ;

python gan/main.py --train_mode False \
    --data_folder sine_cosine_auxl --run_folder_name sine_cosine_auxl_test \
    --dataset_name test --model_name model \
    --epoch 150 --batch_size 100 --sample_len 5 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --train_with_auxl_signal True \
    --generate_gif_training True --generate_num_sample 2 \
    --generate_gif_sample_ids 584 516 \
    --epoch_checkpoint_freq 1 --vis_freq 200 --vis_num_sample 2 ;
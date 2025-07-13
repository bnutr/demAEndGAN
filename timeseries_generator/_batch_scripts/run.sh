#! /bin/sh

python gan/main.py --train_mode True \
    --data_folder sine_cosine --run_folder_name test \
    --dataset_name train --model_name model \
    --epoch 400 --batch_size 100 --sample_len 10 \
    --d_rounds 1 --g_rounds 1 \
    --g_feature_feedback False --g_feature_noise True \
    --g_attribute_latent_dim 5 --g_attribute_num_layers 3 --g_attribute_num_units 100 \
    --g_feature_latent_dim 5 --g_feature_num_layers 1 --g_feature_num_units 100 \
    --d_num_layers 5 --d_num_units 200 \
    --d_auxl_num_layers 5 --d_auxl_num_units 200 \
    --epoch_checkpoint_freq 1 --vis_freq 200 --vis_num_sample 5 ;

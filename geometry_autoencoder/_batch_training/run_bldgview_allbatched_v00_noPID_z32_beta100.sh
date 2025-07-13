#! /bin/sh

python _main_training.py --train True --seed 1 --cuda True \
    --dset_dir _data --dataset bldgview_top --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 128 --z_dim 32 --max_epoch 500 \
    --ratio_train 0.8 --ratio_val 0.1 \
    --gather_epoch 1 --display_epoch 10 --save_epoch 50 \
    --beta 100 --viz_on True --viz_name bldgview_top_v00_noPID_z32_beta100 \
    --is_PID False --KL_loss 120 --image_size 128 ;

python _main_training.py --train True --seed 1 --cuda True \
    --dset_dir _data --dataset bldgview_nw --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 128 --z_dim 32 --max_epoch 500 \
    --ratio_train 0.8 --ratio_val 0.1 \
    --gather_epoch 1 --display_epoch 10 --save_epoch 50 \
    --beta 100 --viz_on True --viz_name bldgview_nw_v00_noPID_z32_beta100 \
    --is_PID False --KL_loss 120 --image_size 128 ;

python _main_training.py --train True --seed 1 --cuda True \
    --dset_dir _data --dataset bldgview_se --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 128 --z_dim 32 --max_epoch 500 \
    --ratio_train 0.8 --ratio_val 0.1 \
    --gather_epoch 1 --display_epoch 10 --save_epoch 50 \
    --beta 100 --viz_on True --viz_name bldgview_se_v00_noPID_z32_beta100 \
    --is_PID False --KL_loss 120 --image_size 128 ;

# python _main_training.py --train True --seed 1 --cuda True \
#     --dset_dir _data --dataset bldgview_allparallel --lr 1e-4 --beta1 0.9 --beta2 0.999 \
#     --objective H --model H --batch_size 128 --z_dim 32 --max_epoch 500 \
#     --ratio_train 0.8 --ratio_val 0.1 \
#     --gather_epoch 1 --display_epoch 10 --save_epoch 50 \
#     --beta 100 --viz_on True --viz_name bldgview_allparallel_v00_noPID_z32_beta100 \
#     --is_PID False --KL_loss 120 --image_size 128 ;
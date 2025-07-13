#! /bin/sh

python _main_encoding.py --z_dim 64 --model_name controlvae_se_z64_KL50 \
    --dataset bldgview_se --batch_size 100 \
    --z_useful 10 29 \
    --dset_dir _data --save_filename_suffix 'useful' \
    --num_workers 4 --image_size 128 ;

python _main_encoding.py --z_dim 64 --model_name controlvae_se_z64_KL50 \
    --dataset bldgview_se --batch_size 100 \
    --dset_dir _data --save_filename_suffix 'all' \
    --num_workers 4 --image_size 128 ;
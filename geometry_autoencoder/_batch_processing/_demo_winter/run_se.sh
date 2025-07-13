#! /bin/sh

python _main_encoding.py --z_dim 64 --model_name controlvae_se_z64_KL50 \
    --dataset bldgview_se --batch_size 100 \
    --z_useful 10 29 \
    --dset_dir _data2 --num_workers 4 --image_size 128 ;
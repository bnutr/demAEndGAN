#! /bin/sh

python _main_reconstructing.py --z_dim 32 --model_name controlvae_top_z32_KL50 \
    --dataset bldgview_top --batch_size 100 --dset_dir _data2 ;

python _main_reconstructing.py --z_dim 64 --model_name controlvae_nw_z64_KL50 \
    --dataset bldgview_nw --batch_size 100 --dset_dir _data2 ;

python _main_reconstructing.py --z_dim 64 --model_name controlvae_se_z64_KL50 \
    --dataset bldgview_se --batch_size 100 --dset_dir _data2 ;
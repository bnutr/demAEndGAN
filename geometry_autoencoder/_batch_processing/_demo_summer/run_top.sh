#! /bin/sh

python _main_encoding.py --z_dim 32 --model_name controlvae_top_z32_KL50 \
    --dataset bldgview_top --batch_size 20 \
    --z_useful 0 1 3 5 7 8 11 12 13 14 15 16 17 18 19 23 28 29 30 \
    --dset_dir _data3 --num_workers 4 --image_size 128 ;
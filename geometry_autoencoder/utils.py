"""utils.py"""

import os
import argparse
import subprocess
from PIL import Image


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def traverse_gif(input_folder, output_folder,
                 duration=10, loop=0):
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    prefixes = {}
    for image_file in image_files:
        prefix = image_file.split('_')[0] + '_' + image_file.split('_')[1]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(image_file)

    # Create GIFs for each prefix
    for prefix, files in prefixes.items():
        images = [Image.open(os.path.join(input_folder, file)) for file in sorted(files)]
        gif_path = os.path.join(output_folder, f"{prefix}.gif")
        if gif_path not in os.listdir(output_folder):
            images[0].save(gif_path, save_all=True, 
                           append_images=images[1:], duration=duration, loop=loop)
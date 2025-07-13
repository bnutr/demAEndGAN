"""metric.py"""

import os
import csv
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import pandas as pd

def cal_fid (folder1, folder2, output_path,
             batch_size=50, device='cuda', dims=2048):
    fid_value = fid_score.calculate_fid_given_paths([folder1, folder2], 
                                                    batch_size, device,
                                                    dims)
    fid_file = os.path.join(output_path, "fid.txt")
    with open(fid_file, "w") as f:
        f.write(str(fid_value))
    print('FID score: {}'.format(fid_value))

def cal_ssim (folder1, folder2, output_path):
    ssim_scores = []
    for file1, file2 in zip(sorted(os.listdir(folder1)), sorted(os.listdir(folder2))):
        img1 = Image.open(os.path.join(folder1, file1)).convert('L')
        img2 = Image.open(os.path.join(folder2, file2)).convert('L')
        if img1.size != img2.size:
            raise ValueError(f"Image size mismatch: {file1} and {file2}")
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        score, _ = ssim(img1_np, img2_np, full=True)
        ssim_scores.append(score)

    ssim_csv_file = os.path.join(output_path, "ssim_scores.csv")
    with open(ssim_csv_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Index", "SSIM Score"])
        for i, score in enumerate(ssim_scores):
            writer.writerow([f"{i+1}", score])
    print(f'SSIM scores saved to {ssim_csv_file}')

    mean_ssim = np.mean(ssim_scores)
    mssim_file = os.path.join(output_path, "mssim.txt")
    with open(mssim_file, "w") as f:
        f.write(str(mean_ssim))
    print('MSSIM: {}'.format(mean_ssim))  
import os
import shutil
import pandas as pd

checkpoint_dir = "./checkpoints"
results_dir = "./results"
target_dir = "./_storage"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

subdirectories_checkpoint = [name for name in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, name)) and name != "#ss"]

for folder in subdirectories_checkpoint:
    last_file = os.path.join(checkpoint_dir, folder, "last")
    target_file = os.path.join(target_dir, folder)
    if os.path.isfile(last_file):
        shutil.copy2(last_file, target_file)

    train_csv_file = os.path.join(checkpoint_dir, folder, "train.csv")
    target_csv_file = os.path.join(target_dir, folder + ".csv")
    if os.path.isfile(train_csv_file):
        shutil.copy2(train_csv_file, target_csv_file)

subdirectories_results = [name for name in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, name)) and name != "#ss"]

ssim_combined = pd.DataFrame()

for folder in subdirectories_results:
    fid_file = os.path.join(results_dir, folder, "fid.txt")
    mssim_file = os.path.join(results_dir, folder, "mssim.txt")
    if os.path.isfile(fid_file and mssim_file):
        with open(fid_file, 'r') as f:
            fid_score = round(float(f.read().strip()), 3)
        with open(mssim_file, 'r') as f:
            mssim_score = round(float(f.read().strip()), 3)
        
        with open(os.path.join(target_dir, "scores.csv"), 'a') as csv_file:
            if csv_file.tell() == 0:
                csv_file.write("Folder,FID,MS-SIM\n")
            csv_file.write(f"{folder},{fid_score},{mssim_score}\n")
    
    ssim_file = os.path.join(results_dir, folder, "ssim_scores.csv")
    if os.path.isfile(ssim_file):
        df = pd.read_csv(ssim_file)
        df['SSIM Score'] = df['SSIM Score'].round(3)
        df.rename(columns={'SSIM Score': folder}, inplace=True)
        ssim_combined = pd.merge(ssim_combined, df[['Index', folder]], on='Index', how='outer') if not ssim_combined.empty else df[['Index', folder]]
    
if not ssim_combined.empty:
    ssim_combined.to_csv(os.path.join(target_dir, "ssim_combined.csv"), index=False)

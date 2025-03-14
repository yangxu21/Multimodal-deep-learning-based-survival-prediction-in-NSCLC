import pandas as pd
import os
import torch
from tqdm import tqdm
import h5py
import numpy as np
# from internal imports
from utils.utils import norm_HnE

destination_path = 'h5_patches_normalized'
source_path = 'h5_patches_original'
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

files = os.listdir(source_path)
files = [f for f in files if f.endswith('.h5') and not f.startswith('.')]
files = [os.path.join(source_path, f) for f in files]

for file in tqdm(files):
    name = file.split('/')[-1].split('.')[0]
    all_imgs = []
    with h5py.File(file, 'r') as f:
        error_id = []
        imgs = f['imgs'][:]
        for j in range(imgs.shape[0]):
            try:
                imgs[j] = norm_HnE(imgs[j])
            except:
                error_id.append(j)
        imgs = np.delete(imgs, error_id, axis=0)
        all_imgs.append(imgs)
    all_imgs = np.concatenate(all_imgs, axis=0)
    with h5py.File(os.path.join(destination_path, name + '.h5'), 'w') as f:
        f.create_dataset('imgs', data=all_imgs)






import torch 
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import h5py

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available, use GPU')
else:
    device = torch.device('cpu')
    print('use CPU')


res50 = torch.load('res50_trained_model.pt', map_location=device)
res50.to(device)
res50.eval()


transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.76271061, 0.62484627, 0.7082412],
                            std=[0.16601817, 0.19249444, 0.15693703])
])


class bag(Dataset):
    def __init__(self, path, transform=transform_img):
        super(bag, self).__init__()
        with h5py.File(path, "r") as f:
            self.imgs = f['imgs'][()]
        self.transform = transform
    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img

def extract_features(dataset, model, name, destination_path, batch_size=32):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    features = []
    for img in tqdm(loader):
        with torch.inference_mode():	
            img = img.to(device)
            feature = model(img)
            feature = feature.to('cpu')
            features.append(feature)
    features = torch.cat(features, dim=0)
    torch.save(features, os.path.join(destination_path, name + '.pt'))
    print('finish ' + name)

source_path = 'h5_patches_normalized'
destination_path = 'extracted_features_2048'
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

files = os.listdir(source_path)
files = [f for f in files if f.endswith('.h5') and not f.startswith('.')]
files = [os.path.join(source_path, f) for f in files]

for file in tqdm(files):
    name = file.split('/')[-1].split('.')[0]
    dataset = bag(file)
    extract_features(dataset, res50, name, destination_path, batch_size=128)


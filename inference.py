import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available, use GPU')
else:
    device = torch.device('cpu')
    print('use CPU')

attention_model = torch.load('trained_attention_model_pathology.pt', map_location='cpu')
attention_model.to(device)
attention_model.eval()

class bag(Dataset):
    def __init__(self, df):
        super(bag, self).__init__()
        self.df = df
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        data = torch.load(self.df['pt_path'][idx], map_location='cpu')
        name = self.df['name'][idx]
        return data, name

source_path = 'extracted_features_2048'
files = os.listdir(source_path)
files = [f for f in files if f.endswith('.pt') and not f.startswith('.')]
files = [os.path.join(source_path, f) for f in files]
df = pd.DataFrame(files, columns=['pt_path'])
df['name'] = df['pt_path'].apply(lambda x: x.split('/')[-1].split('.')[0])


def inference(df, model, destination_path, device=device):
    all_preds = []
    names = []
    dataset = bag(df)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
    for data, name in tqdm(loader):
        with torch.inference_mode():
            data = data.squeeze(0)
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().detach()
            all_preds.extend(pred)
            names.extend(name)
    # perform sigmoid on the predictions
    all_preds = torch.stack(all_preds)
    all_preds = torch.sigmoid(all_preds)
    # make a dataframe and save it
    df = pd.DataFrame(all_preds.numpy(), columns=['pred'])
    df['name'] = names
    df.to_csv(destination_path + '/predictions.csv', index=False)


destination_path='reference_results'
inference(df, attention_model, destination_path)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import MultiStepLR

# internal imports
from model.model_attention import MIL_attn

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available, use GPU')
else:
    device = torch.device('cpu')
    print('use CPU')

class pathology_bag(Dataset):
    def __init__(self, df, patch_size):
        super(pathology_bag, self).__init__()
        self.df = df
        self.patch_size = patch_size

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = torch.load(self.df['pathology_CNN_pt_path'][idx], map_location='cpu')   
        data = data[torch.randperm(data.size(0))]                                
        data = data[:self.patch_size]
        label = self.df['cancer_type_cat'][idx]
        return data, label

def training_process(model, dataloader, optimizer, device, weight, gc=32):
    model.train()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    losses = []

    for i, (data, label) in enumerate(tqdm(dataloader)):
        data = data.squeeze(0).to(device)
        label = label[0].unsqueeze(0).float().to(device)
        pred = model(data)
        loss = criterion(pred.squeeze(0), label)
        losses.append(loss.item())
        loss = loss / gc
        loss.backward()

        if (i+1) % gc == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    print("loss_first: ", losses[0])
    print('loss_last: ', losses[-1])
    return losses

def model_training(model, df_path, weight, destination_path, patch_size=4096, num_epochs=10, lr=1e-4, weight_decay=1e-3, device=device, scheduler=False, gc=32):
    '''
    model: model to be trained
    df_path: dataframe of all the samples with columns ['pathology_CNN_pt_path', 'cancer_type_cat']
          - 'pathology_CNN_pt_path': path to the preprocessed .pt files
          - 'cancer_type_cat': label of the samples (0 for LUAD, 1 for LUSC)
    weight: ratio of positive and negative samples
    destination_path: path to save the model
    patch_size: number of patches in each bag
    gc: affect the gradient accumulation
    scheduler: whether to use scheduler to adjust learning rate
    '''
    
    df = pd.read_csv(df_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    if scheduler:
        scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs//2), int(num_epochs//1.2)], gamma=0.5)

    all_losses = []

    for epoch in tqdm(range(num_epochs)):
        print('epoch: ', epoch)
        df = df.sample(frac=1, replace=False).reset_index(drop=True)
        patch_size = int(patch_size * 1.5)
        dataset = pathology_bag(df, patch_size)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)
        losses = training_process(model=model, dataloader=loader, optimizer=optimizer, device=device, weight=weight, gc=gc)
        all_losses.extend(losses)

        if scheduler:
            scheduler.step()

    np.save(os.path.join(destination_path, 'attention_model_all_losses.npy'), all_losses)  
    torch.save(model.state_dict(), os.path.join(destination_path, 'attention_model_weight.pth'))

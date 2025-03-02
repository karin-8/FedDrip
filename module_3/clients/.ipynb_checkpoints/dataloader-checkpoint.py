import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from numpy import asarray
from torch.utils.data import Dataset

class ClientDataset(Dataset):
    def __init__(self, df, labels_col, root_dir, sep_folder=False, transform=False):
        self.df = df
        self.labels = df[labels_col].values.tolist()
        self.root_dir = root_dir
        self.sep_folder = sep_folder
        
        if transform:
            self.convert_tensor = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225] , inplace=True)
                               ])
        else:
            self.convert_tensor = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225], inplace=True)])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.df.iloc[idx]['Image Index']
        if self.sep_folder==True:
            img_root = self.root_dir + f'site_{self.df.iloc[idx]["Site"]}/syn/'
            image_path = os.path.join(img_root, filename)
        else:
            image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path)
        npimg = np.asarray(image)
        if len(npimg.shape) == 2:
            npimg = npimg[:, :, np.newaxis]
            npimg = np.concatenate([npimg, npimg, npimg], axis=2)
        if len(npimg.shape)>2:
            npimg = npimg[:,:,0]
            npimg = npimg[:, :, np.newaxis]
            npimg = np.concatenate([npimg, npimg, npimg], axis=2)
        # image = np.concatenate([npimg, npimg, npimg], axis=2)
        image = Image.fromarray(npimg, 'RGB')
        tensor = self.convert_tensor(image)
        # stack = torch.stack((tensor,)*3, axis=1)[0]
        labels = torch.tensor(self.labels[idx])
        return {
            'image': tensor,
            'labels': labels
        }
        
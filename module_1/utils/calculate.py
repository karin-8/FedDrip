"""Script for calculating Inception statistics"""

import pickle
import numpy as np
import torch
import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ChestXDataset

#----------------------------------------------------------------------------

def cal_stats_imlist(imlist, num_samples=1000, device=torch.device('cuda:0')):
    
    print('Loading Inception-v3 model...')
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    # detector_net = pickle.load(open('detector_net.pkl', 'rb')).to(device)  # In my case this call is buried deeper in torch-agnostic code\
    detector_net = torch.load("./module_1/models/inception/detector_net.pt", map_location=device)
    # detector_net = torch.load(open('detector_net.pkl', 'rb'), map_location=device)

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    image_ds = ChestXDataset(imlist)
    num_samples = min(num_samples, len(image_ds))
    image_dl = DataLoader(image_ds, batch_size=32, shuffle=False)
    
    count = 0
    with tqdm(total=num_samples) as pbar:
        for idx, batch in enumerate(image_dl):
            images = batch.to(device)
            # images = np.array(Image.open(file))
            # images = batch.permute(0, 3, 1, 2).to(device)
            # images = torch.tensor(images).to(device)
            features = detector_net(images, **detector_kwargs).to(torch.float64)
            if count + images.shape[0] > num_samples:
                remaining_num_samples = num_samples - count
            else:
                remaining_num_samples = images.shape[0]
            mu += features[:remaining_num_samples].sum(0)
            sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
            count = count + remaining_num_samples
            pbar.update(remaining_num_samples)
            if count >= num_samples:
                break

    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def cal_stats_loader(loader, num_samples=1000, device=torch.device('cuda:0')):
    
    print('Loading Inception-v3 model...')
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    # detector_net = pickle.load(open('detector_net.pkl', 'rb')).to(device)  # In my case this call is buried deeper in torch-agnostic code\
    detector_net = torch.load("./module_1/detector_net.pt", map_location=device)
    # detector_net = torch.load(open('detector_net.pkl', 'rb'), map_location=device)

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    num_samples = min(num_samples, len(loader.dataset))
    
    count = 0
    with tqdm(total=num_samples) as pbar:
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            # images = np.array(Image.open(file))
            # images = batch.permute(0, 3, 1, 2).to(device)
            # images = torch.tensor(images).to(device)
            features = detector_net(images, **detector_kwargs).to(torch.float64)
            if count + images.shape[0] > num_samples:
                remaining_num_samples = num_samples - count
            else:
                remaining_num_samples = images.shape[0]
            mu += features[:remaining_num_samples].sum(0)
            sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
            count = count + remaining_num_samples
            pbar.update(remaining_num_samples)
            if count >= num_samples:
                break

    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def stats_to_fid(mu_x, sigma_x, mu_y, sigma_y, device=torch.device('cuda:0')):
    mu_x = torch.tensor(mu_x).to(device)
    sigma_x = torch.tensor(sigma_x).to(device)
    mu_y = torch.tensor(mu_y).to(device)
    sigma_y = torch.tensor(sigma_y).to(device)
    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)
    return a + b - 2 * c

#----------------------------------------------------------------------------

def cal_fid(imlist, ref_list, num_samples=1000, device=torch.device('cuda:0')):
            mu_x, sigma_x = cal_stats_imlist(imlist, num_samples, device)
            mu_y, sigma_y = cal_stats_imlist(ref_list, num_samples, device)
            fid = stats_to_fid(mu_x, sigma_x, mu_y, sigma_y, device)
            print(f"FID = {fid}")
            return fid
            
#----------------------------------------------------------------------------
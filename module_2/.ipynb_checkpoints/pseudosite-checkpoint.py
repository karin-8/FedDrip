import os

import calculate
from calculate import *

import numpy as np
import pandas as pd

import torch
from torchvision import transforms

import json
import random
from tqdm import tqdm

from generate import generate_synthetic
from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

from get_report import get_report

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


with open("./configs/config_module_2/config.json", "r") as conf:
    d = json.load(conf)
    args = Config(d)
    

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    config = OmegaConf.load(args.aekl_config_file_path)
    aekl = AutoencoderKL(**config["aekl"]["params"])
    aekl.load_state_dict(torch.load(args.aekl_path))
    aekl.to(device)
    aekl.eval()
    
    config = OmegaConf.load(args.diffusion_config_file_path)
    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
    
    pm = Pipeline_Models(aekl, None, tokenizer, None, scheduler, config)
    pm.diffusion = load_diffusion(args, config)
    pm.diffusion.to(device)
    pm.diffusion.eval()
    pm.text_encoder = load_text_encoder(args)
    
    if not os.path.exists(args.synthetic_dir):
        os.makedirs(args.synthetic_dir)
    
    samp_dir = args.synthetic_dir
    log_dir = args.log_dir
    
    if args.prompt_style in ["term", "single-term", "single_term"]:
        report_dir = "./module_2/prompts/single-term.json"
    elif args.prompt_style in ["keyword", "kw"]:
        report_dir = "./module_2/prompts/keyword.json"
    elif args.prompt_style in ["full", "full-sentence", "full_sentence"]:
        report_dir = "./module_2/prompts/full-sentence.json"
    else:
        raise ValueError("Invalid prompt style")
    train_df = pd.read_csv(args.pseudo_site_train_csv)
    val_df = pd.read_csv(args.pseudo_site_val_csv)
    df = pd.concat([train_df, val_df])
    ds = ChestXDataset(df)
    image_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch_count = 0
    for batch in tqdm(image_loader):
        img_indices = batch[0]
        targets = torch.transpose(torch.stack(batch[1]),0,1)
        prompts = []
        for t in targets:
            prompts.append(get_report(report_dir, val_df, t, args))
        args.prompts = prompts
        img_list = generate_synthetic(args,pm)
        for idx, im in enumerate(img_list):
            im.save(samp_dir + img_indices[idx])
        args.seed = random.randint(0,1e6)
        batch_count += 1
    exit(1)

class ChestXDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __call__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["Image Index"], list(row[self.df.columns[-15:]])
        
class Pipeline_Models:
    def __init__(self, vae, unet, tokenizer, text_encoder, scheduler, config):
        self.aekl = vae
        self.diffusion = unet
        self.tokenizer  = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.config = config

def load_diffusion(args, config):
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path)["model_state_dict"])
    return diffusion

def load_text_encoder(args):
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)
    return text_encoder

if __name__=="__main__":
    # try:
    #     main(args)
    # except Exception as e:
    #     print(e)
    #     exit(0)
    main(args)
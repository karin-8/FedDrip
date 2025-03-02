import os
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import argparse
import json
import pandas as pd
from tqdm import tqdm
import random

from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf

from aekl_wrapper import AEKLWrapper
from functions.util import get_dataloader

from calculate import cal_fid
from generate_multiple import generate_synthetic

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, default=0)
parser.add_argument("--mode", type=str, default='loss')
parser.add_argument("--sample", type=float, default=1.0)

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

class PipelineModels:
    def __init__(self, vae, unet, tokenizer, text_encoder, scheduler, config):
        self.aekl = vae
        self.diffusion = unet
        self.tokenizer  = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.config = config

with open("./configs/config_module_1/df_config.json", "r") as conf:
    d = json.load(conf)
    df_conf = Config(d)
    
with open("./configs/config_module_1/fl_config.json", "r") as conf:
    d = json.load(conf)
    fl_conf = Config(d)

with open("./configs/config_module_1/eval_config.json", "r") as conf:
    d = json.load(conf)
    eval_conf = Config(d)

config_aekl = OmegaConf.load(df_conf.config_file_aekl)
aekl = AutoencoderKL(**config_aekl["aekl"]["params"])
aekl.load_state_dict(torch.load(df_conf.aekl_uri), strict=False)

config_ldm = OmegaConf.load(df_conf.config_file_ldm)
config_ldm['ldm']['base_lr'] = df_conf.base_lr
diffusion = DiffusionModelUNet(**config_ldm["ldm"].get("params", dict()))

scheduler = DDIMScheduler(
    num_train_timesteps=config_ldm["ldm"]["scheduler"]["num_train_timesteps"],
    beta_start=config_ldm["ldm"]["scheduler"]["beta_start"],
    beta_end=config_ldm["ldm"]["scheduler"]["beta_end"],
    prediction_type=config_ldm["ldm"]["scheduler"]["prediction_type"],
    clip_sample=False,
)
scheduler.set_timesteps(df_conf.num_inference_steps)

text_encoder = CLIPTextModel.from_pretrained(df_conf.text_encoder_folder, subfolder="text_encoder")

device = torch.device(df_conf.device)

aekl = aekl.to(device)
diffusion = diffusion.to(device)

def main(args):
    if args.round != fl_conf.fl_rounds:
        diffusion.load_state_dict(torch.load(os.path.join(fl_conf.work_dir, f"global_model_round_{args.round}.pt"))["model_state_dict"])
    else:
        diffusion.load_state_dict(torch.load(os.path.join(fl_conf.work_dir, f"final_model.pt"))["model_state_dict"])


    real_dir = eval_conf.ref_data_path
    
    if args.mode == 'loss':
        val_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, "val.csv")).sample(frac=args.sample, random_state=eval_conf.seed)
        train_df = pd.DataFrame(columns=val_df.columns) # dummy train_df
        train_loader, val_loader = get_dataloader(
            cache_dir=None,
            prompt_path=df_conf.prompt_path,
            batch_size=df_conf.batch_size,
            val_batch_size=eval_conf.batch_size,
            img_size=df_conf.img_size,
            train_df=train_df,
            val_df=val_df,
            eval_mode=args.mode,
            model_type="diffusion"
        )
        loss = eval_ldm_loss(diffusion, aekl, scheduler, text_encoder, val_loader, device, args.round, scale_factor=eval_conf.scale_factor)
    elif args.mode == 'fid':
        val_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, "val.csv")).sample(int(args.sample), random_state=eval_conf.seed)
        train_df = pd.DataFrame(columns=val_df.columns) # dummy train_df
        ref_list = val_df["Image Index"].apply(lambda x: Image.open(os.path.join(real_dir, x))).tolist()
        fid, samp_img = eval_ldm_fid(diffusion, aekl, scheduler, text_encoder, ref_list, val_df, device, args.round)
        samp_img.save(os.path.join(fl_conf.work_dir, f"round_{args.round}_sample.png"))
        
    if "losses.json" not in os.listdir(fl_conf.work_dir):
        with open(os.path.join(fl_conf.work_dir, "losses.json"), "w") as f:
            d = {}
            if args.mode == "loss":
                d["losses"] = [loss]
            elif args.mode == "fid":
                d["fids"] = [fid]
            f.write(json.dumps(d))
    else:
        with open(os.path.join(fl_conf.work_dir, "losses.json"), "r") as f:
            d = json.load(f)
            if args.mode == "loss":
                if "fids" in d.keys():
                    d = {"losses":[loss]}
                else:
                    if len(d["losses"]) >= args.round//fl_conf.eval_every:
                        d["losses"] = d["losses"][:args.round//fl_conf.eval_every - 1]
                    d["losses"].append(loss)
            elif args.mode == "fid":
                if "losses" in d.keys():
                    d = {"fids":[fid]}
                else:
                    if len(d["fids"]) >= args.round//fl_conf.eval_every:
                        d["fids"] = d["fids"][:args.round//fl_conf.eval_every - 1]
                    d["fids"].append(fid)
        with open(os.path.join(fl_conf.work_dir, "losses.json"), "w") as f:
            f.write(json.dumps(d))

@torch.no_grad()
def eval_ldm_loss(
    model: nn.Module,
    aekl: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    scale_factor: float = 1.0,
) -> float:
    model.eval()
    aekl = AEKLWrapper(model=aekl)
    aekl.eval()
    raw_aekl = aekl.module if hasattr(aekl, "module") else aekl
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()
    
    text_encoder = text_encoder.to(device)

    for x in loader:
        images = x["image"].to(device)
        reports = x["report"].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(enabled=True):
            e = aekl(images) * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred = model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    return total_losses["loss"]

@torch.no_grad()
def eval_ldm_fid(
    model: nn.Module,
    aekl: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    ref_list: list,
    val_df: pd.DataFrame,
    device: torch.device,
    step: int,
) -> float:
    model.to(device)
    model.eval()
    aekl.to(device)
    aekl.eval()
    raw_aekl = aekl.module if hasattr(aekl, "module") else aekl
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()

    imlist = list()
    
    tokenizer = CLIPTokenizer.from_pretrained(df_conf.tokenizer_folder, subfolder="tokenizer")
    config = OmegaConf.load(df_conf.config_file_ldm)
    
    pm = PipelineModels(aekl, model, tokenizer, text_encoder, scheduler, config)
    
    print("evaluating diffusion model")
    batch_size = eval_conf.batch_size
    n_batch = -(-len(val_df)//batch_size)
    for b in tqdm(range(n_batch)):
        eval_conf.prompts = []
        rows = val_df.iloc[b*batch_size:(b+1)*batch_size]
        for i, row in rows.iterrows():
            eval_conf.prompts.append(get_report(eval_conf.eval_prompt_path, val_df, row))
        outputs = generate_synthetic(eval_conf,pm)
        imlist += outputs
    fid = cal_fid(imlist, ref_list, len(val_df), device).item()
    return fid, imlist[random.randint(0, len(imlist)-1)]

def get_report(prompt_path, df, row):
    with open(prompt_path, "r") as f:
        report_dict = json.load(f)
    labels = df.columns[-15:].tolist()
    onehot = row[labels].tolist()
    disease_list = [labels[idx] for idx in range(len(onehot)) if onehot[idx] != 0]
    report = list()
    for disease in disease_list:
        report.append(report_dict[disease.replace("_", " ")])
    selected = random.sample(report, k=min(len(report),5))
    return ". ".join(report)

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
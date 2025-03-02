import os

import argparse
from pathlib import Path
import itertools
import pandas as pd
import json
from PIL import Image
from datetime import datetime
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from functions.training_functions import train_ldm
from functions.util import get_dataloader
from aekl_wrapper import AEKLWrapper

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
with open("./configs/config_module_1/df_config.json", "r") as conf:
    d = json.load(conf)
    df_conf = Config(d)

with open("./configs/config_module_1/fl_config.json", "r") as conf:
    d = json.load(conf)
    fl_conf = Config(d)

with open("./configs/config_module_1/eval_config.json", "r") as conf:
    d = json.load(conf)
    eval_conf = Config(d)

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="test experiment")
parser.add_argument("--client", type=int, default=None)
parser.add_argument("--round", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--n_steps", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--log", type=str, default="")

def main(df_conf, fl_conf, args):
    work_dir = fl_conf.work_dir

    now = str(datetime.now()).replace(" ", "_")
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    logname = args.log + now + ".log"
    
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.DEBUG)
    
    project_name = f"site {args.client}"
    
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info(f"Training {project_name}")
    if df_conf.train_sample_frac:
        train_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, f"train_{args.client}.csv")).sample(frac=df_conf.train_sample_frac, random_state=fl_conf.seed)
    else:
        train_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, f"train_{args.client}.csv"))

    if df_conf.local_eval_mode == "loss":
        val_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, f"val_{args.client}.csv")).sample(frac=df_conf.local_eval_sample_frac_loss, random_state=fl_conf.seed)
        val_imlist = []
    elif df_conf.local_eval_mode == "fid":
        val_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, f"val_{args.client}.csv")).sample(df_conf.local_eval_sample_fid, random_state=fl_conf.seed)
        val_imlist = val_df["Image Index"].apply(lambda x: Image.open(os.path.join(eval_conf.ref_data_path, x))).tolist()
    set_determinism(seed=df_conf.seed)
    
    logger.info(f"Run directory: {str(work_dir)}")
    writer_train = SummaryWriter(log_dir=Path(work_dir) / "tb" / str(args.client) / "train")
    writer_val = SummaryWriter(log_dir=Path(work_dir) / "tb" / str(args.client) / "val")
              
    
    logger.info("Getting data...")

    train_loader, val_loader = get_dataloader(
        cache_dir=None,
        prompt_path=df_conf.prompt_path,
        batch_size=df_conf.batch_size,
        val_batch_size=eval_conf.batch_size,
        img_size=df_conf.img_size,
        num_workers=df_conf.num_workers,
        train_df=train_df,
        val_df=val_df,
        eval_mode=df_conf.local_eval_mode,
        model_type="diffusion",
    )
    
    

    # Load Autoencoder to produce the latent representations
    logger.info(f"Loading AEKL from {df_conf.aekl_uri}")

    config_aekl = OmegaConf.load(df_conf.config_file_aekl)
    aekl = AutoencoderKL(**config_aekl["aekl"]["params"])
    aekl.load_state_dict(torch.load(df_conf.aekl_uri), strict=True)
    aekl = AEKLWrapper(model=aekl)
    aekl.eval()

    # Create the diffusion model
    logger.info("Creating model...")
    config_ldm = OmegaConf.load(df_conf.config_file_ldm)
    config_ldm['ldm']['base_lr'] = df_conf.base_lr
    diffusion = DiffusionModelUNet(**config_ldm["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config_ldm["ldm"].get("scheduler", dict()))
    val_scheduler = DDIMScheduler(
        num_train_timesteps=config_ldm["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config_ldm["ldm"]["scheduler"]["beta_start"],
        beta_end=config_ldm["ldm"]["scheduler"]["beta_end"],
        prediction_type=config_ldm["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    val_scheduler.set_timesteps(eval_conf.num_inference_steps)
    
    text_encoder = CLIPTextModel.from_pretrained(df_conf.text_encoder_folder, subfolder="text_encoder")

    logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device(df_conf.device)
    
    aekl = aekl.to(device)
    diffusion = diffusion.to(device)
    # text_encoder = text_encoder.to(device)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=config_ldm["ldm"]["base_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    best_fid = float("inf")
    
    start_epoch = 0
    
    logger.info(f"Loading global model from round {args.round}")
    checkpoint = torch.load(Path(work_dir) / f"global_model_round_{args.round}.pt")
    diffusion.load_state_dict(checkpoint["model_state_dict"])

    args.n_steps = min(args.n_steps, args.n_epochs * len(train_df))
    args.n_epochs = -(-args.n_steps//len(train_df))
    if args.patience == 0:
        args.patience = None
    
    # Train model
    logger.info(f"Starting Training")
    best_model, val_loss = train_ldm(
        my_round=args.round,
        site=args.client,
        model=diffusion,
        aekl=aekl,
        scheduler=scheduler,
        val_scheduler=val_scheduler,
        text_encoder=text_encoder,
        eval_config=eval_conf,
        start_epoch=start_epoch,
        eval_mode=df_conf.local_eval_mode,
        best_loss=best_loss,
        best_fid=best_fid,
        train_loader=train_loader,
        val_loader=val_loader,
        val_df=val_df,
        val_imlist=val_imlist,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        sample=df_conf.train_sample_frac,
        patience=args.patience,
        eval_freq=df_conf.local_eval_freq,
        eval_sample=df_conf.local_eval_sample_fid,
        writer_train=writer_train,
        writer_val=writer_val,
        logger=logger,
        device=device,
        run_dir=work_dir,
        scale_factor=df_conf.scale_factor,
    )

    logger.info("Training Finished")
    save_info = {
        "updates":best_model,
        "samples":len(train_df)
    }
    save_name = f"round_{args.round+1}_site_{args.client}.ckpt"
    torch.save(save_info, Path(work_dir) / save_name) 


if __name__ == "__main__":
    args = parser.parse_args()
    main(df_conf, fl_conf, args)

    
    
    
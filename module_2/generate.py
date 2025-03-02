import argparse
from pathlib import Path

import time

import numpy as np
import pandas as pd
import torch
from monai.config import print_config
from monai.utils import set_determinism
from PIL import Image
from tqdm import tqdm

import logging
import sys

# gen image according to df



def generate_synthetic(args, pipe_models):
    
    device = torch.device(args.device)
    seed = args.seed
    prompts_ids = []

    for prompt in args.prompts:
        prompt = ["", prompt.replace("_", " ")]
        text_inputs = pipe_models.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe_models.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompts_ids.append(text_input_ids)
        
    prompts_ids = torch.stack(prompts_ids).reshape((-1, pipe_models.tokenizer.model_max_length))

    prompt_embeds = pipe_models.text_encoder(prompts_ids)
    prompt_embeds = prompt_embeds[0].to(device)
    set_determinism(seed=seed)
    noise = torch.randn((len(args.prompts), pipe_models.config["ldm"]["params"]["in_channels"], args.x_size, args.y_size)).to(device)

    with torch.no_grad():
        for t in pipe_models.scheduler.timesteps:
            noise_input = torch.cat([noise] * 2)
            model_output = pipe_models.diffusion(
                noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
            )
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            noise, _ = pipe_models.scheduler.step(noise_pred, t, noise)

    with torch.no_grad():
        sample = pipe_models.aekl.decode_stage_2_outputs(noise / args.scale_factor)
    imgs = []
    for s in sample.squeeze(1):
        s = np.clip(s.cpu().numpy(), 0, 1)
        s = (s * 255).astype(np.uint8)
        im = Image.fromarray(s)
        imgs.append(im)
    return imgs
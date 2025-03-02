import os
import numpy
import argparse
import torch
from pathlib import Path
from collections import OrderedDict
import copy
import json

parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, default=0)
parser.add_argument("--num_clients", type=int, default=6)
parser.add_argument("--device", type=str, default="cuda")

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


with open("./configs/config_module_1/fl_config.json", "r") as conf:
    d = json.load(conf)
    fl_conf = Config(d)
    
with open("./configs/config_module_1/df_config.json", "r") as conf:
    d = json.load(conf)
    df_conf = Config(d)

def main(args):
    model = torch.load(Path("./") / fl_conf.work_dir / f"global_model_round_{args.round}.pt")["model_state_dict"]
    updates = []
    for c in range(1, args.num_clients+1):
        saved_name = f"round_{args.round+1}_site_{c}.ckpt"
        print("loading " + str(Path("./") / fl_conf.work_dir / saved_name))
        checkpoint = torch.load(Path("./") / fl_conf.work_dir / saved_name)
        num_samples = checkpoint["samples"]
        updates.append((num_samples, checkpoint["updates"]))
        os.remove(Path("./") / fl_conf.work_dir / saved_name)
    averaged_soln = update_model(model, updates, torch.device(args.device))
    save_model(averaged_soln, args.round, os.path.join(fl_conf.work_dir, "averaged_soln.pt"))

def update_model(model, updates, device):

    total_weight = 0.
    base = OrderedDict()
    for (client_samples, client_model) in updates:
        total_weight += client_samples
        for key, value in client_model.items():
            if key in base:
                base[key] += (client_samples * value.type(torch.FloatTensor))
            else:
                base[key] = (client_samples * value.type(torch.FloatTensor))
    averaged_soln = copy.deepcopy(model)
    for key, value in base.items():
        if total_weight != 0:
            averaged_soln[key] = value.to(device) / total_weight
    torch.cuda.empty_cache()
    return averaged_soln
    
def save_model(model, my_round, ckpt_path):

    save_info = {'model_state_dict': model,
                 'round': my_round}
    torch.save(save_info, ckpt_path)
    return ckpt_path

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
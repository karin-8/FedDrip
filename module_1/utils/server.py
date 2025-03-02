import os
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
import subprocess as subp
import copy
import time

class Server:
    
    def __init__(self, client_model, args):

        self.project = args.project
        self.device = torch.device(args.device)
        self.model = copy.deepcopy(client_model)
        self.num_clients = args.num_clients
        self.parallel_clients = args.parallel_clients
        self.work_dir = args.work_dir
        self.selected_clients = []
        self.updates = []
        self.eval_mode = args.global_eval_mode
        self.eval_sample_fid = args.global_eval_sample_fid
        self.eval_sample_frac_loss = args.global_eval_sample_frac_loss

    def select_clients(self, my_round, possible_clients, num_clients):

        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients], self.selected_clients

    def train_model(self, my_round, writer, where_log, num_epochs=None, num_steps=None, batch_size=10, patience=5):

        clients = self.selected_clients
        p = self.parallel_clients
        n = self.num_clients
        for i in range(n//p):
            procs = {}
            for j in range(p): ### Parallel training
                proc = subp.Popen([
                    "python", "./module_1/utils/training_fn.py",
                    "--project", str(self.project),
                    "--client", str(i*p+j+1),
                    "--round", str(my_round),
                    "--n_epochs", str(num_epochs),
                    "--n_steps", str(num_steps),
                    "--batch_size", str(batch_size),
                    "--patience", str(patience),
                    "--log", str(where_log[i*p+j+1]),
                ])
                procs[p] = proc
            for j in range(p):
                procs[p].wait() ### wait for all to complete
        return

    def update_model(self, my_round):
        
        subp.run(["python", "module_1/utils/fedavg.py", "--round", str(my_round), "--num_clients", str(self.num_clients)])
        self.model = torch.load(os.path.join(self.work_dir, "averaged_soln.pt"), map_location="cpu")["model_state_dict"]
        os.remove(os.path.join(self.work_dir, "averaged_soln.pt"))

    def save_model(self, my_round, ckpt_path):

        save_info = {'model_state_dict': self.model,
                     'round': my_round}
        torch.save(save_info, ckpt_path)
        return ckpt_path

    def eval_model(self, my_round, mode):
        if mode == "loss":
            subp.run(["python", "module_1/utils/eval_fn.py", "--round", str(my_round), "--mode", self.eval_mode, "--sample", str(self.eval_sample_frac_loss)])
        elif mode == "fid":
            subp.run(["python", "module_1/utils/eval_fn.py", "--round", str(my_round), "--mode", self.eval_mode, "--sample", str(self.eval_sample_fid)])

    def clear_model(self):
        self.model = None
        

class Client:

    def __init__(self, idx, train_df, test_df):

        self.num_train_samples = len(train_df)
        self.num_test_samples = len(test_df)
        self.idx = idx
        


                
            
import copy
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
import subprocess
import time

class FedAVGServer:
    
    def __init__(self, client_model, args):
        self.project = args.project_name + "_" + args.experiment
        self.work_dir = args.work_dir
        self.client_model = copy.deepcopy(client_model)
        self.device = self.client_model.device
        self.seed = args.seed
        self.model = copy.deepcopy(client_model.state_dict())
        self.num_clients = args.num_clients
        self.pseudo_site_ids = args.pseudo_site_ids
        self.pseudo_site_every = args.pseudo_site_every
        self.pseudo_site_start = args.pseudo_site_start
        self.parallel_clients = args.parallel_clients
        self.client_weights = list(args.client_weights)
        self.selected_clients = []
        self.updates = []
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.args = args
        
    def select_clients(self, my_round, possible_clients, num_clients):

        num_clients = min(num_clients, len(possible_clients))
        done = False
        if len(self.pseudo_site_ids) != 0:
            self.selected_clients = np.random.choice(possible_clients[:-len(self.pseudo_site_ids)], num_clients, replace=False)
            if (my_round+1) % self.pseudo_site_every == 0 and my_round+1 >= self.pseudo_site_start:
                pseudo_site = [c for c in possible_clients if c.id in self.pseudo_site_ids]
                self.selected_clients = np.append(self.selected_clients, np.random.choice(pseudo_site, len(pseudo_site), replace=False))
        else:
            self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients], self.selected_clients

    def train_model(self, rounds, writer, where_logs, num_epochs=1, batch_size=10, patience= 5, minibatch=None, clients=None, analysis=False):

        if clients is None:
            clients = self.selected_clients
            
        aucs = {"name":"f1"}
        aucs["values"] = {}
        val_losses = {"name":"val_loss"}
        val_losses["values"] = {}
        clients = [c.id for c in clients]
        train_bools = [c in clients for c in range(1,self.num_clients+1)]
        train_args = ["true" if t else "false" for t in train_bools]
        pseudo_args = [c in self.pseudo_site_ids for c in range(1,self.num_clients+1)]
        pseudo_args = [1 if c else 0 for c in pseudo_args]
        procs = {}
        # selected_clients = [c.id for c in self.selected_clients]
        not_selected = [c for c in range(1,self.num_clients+1) if c not in clients]
        client_list = sorted(clients) + not_selected
        pc = self.parallel_clients
        bs = math.ceil((self.num_clients-len(self.pseudo_site_ids))//pc)
        for i in range(bs):
            for j in range(pc):
                if pc*i + j == self.num_clients - len(self.pseudo_site_ids):
                    break
                c = client_list[pc* i + j]
                p = subprocess.Popen(["python", "./module_3/clients/training_fn_fedavg.py", "--project", str(self.project), "--client", str(c), "--pseudo", str(pseudo_args[c-1]), "--round", str(rounds), 
                                  "--train", str(train_args[c-1]), "--n_epochs", str(num_epochs), "--batch_size", str(batch_size), "--lr", str(self.lr),
                                  "--patience", str(patience), "--log", str(where_logs[c-1]), "--pbar", str(c)])
                procs[j] = p
            for j in range(pc):
                if pc*i + j == self.num_clients - len(self.pseudo_site_ids):
                    break
                procs[j].wait()
        if self.pseudo_site_ids:
            c = self.num_clients
            p = subprocess.Popen(["python", "./module_3/clients/training_fn_teachers.py", "--project", str(self.project), "--client", str(c), "--pseudo", str(pseudo_args[c-1]), "--round", str(rounds), 
                                      "--train", str(train_args[c-1]), "--n_epochs", str(num_epochs), "--batch_size", str(batch_size), "--lr", str(self.lr),
                                      "--patience", str(patience), "--log", str(where_logs[c-1]), "--pbar", str(c)])
            p.wait()
        for c in clients:
            saved_name = f"round_{rounds+1}_site_{c}.ckpt"
            checkpoint = torch.load(os.path.join("module_3", self.work_dir, saved_name))
            if rounds+1 != self.args.num_rounds:
                os.remove(os.path.join("module_3", self.work_dir, saved_name))
            aucs["values"][c] = checkpoint["f1"]
            val_losses["values"][c] = checkpoint["val_losses"]
            weight = checkpoint["samples"] * self.client_weights[c-1]
            self.updates.append((weight, checkpoint["update"]))
            
        return [aucs, val_losses]

    def update_model(self):

        averaged_soln = self._average_updates()

        self.client_model.load_state_dict(averaged_soln)
        self.model = copy.deepcopy(self.client_model.state_dict())
        self.updates = []
        return

    def _average_updates(self):

        total_weight = 0.
        base = OrderedDict()
        for (client_weight, client_model) in self.updates:
            total_weight += client_weight
            for key, value in client_model.items():
                if key in base:
                    base[key] += (client_weight * value.type(torch.FloatTensor))
                else:
                    base[key] = (client_weight * value.type(torch.FloatTensor))

        averaged_soln = copy.deepcopy(self.model)
        for key, value in base.items():
            if total_weight != 0:
                averaged_soln[key] = value.to(self.device) / total_weight
        return averaged_soln

    def save_model(self, rounds, ckpt_path):

        save_info = {'model_state_dict': self.model,
                     'round': round}
        torch.save(save_info, ckpt_path)
        return ckpt_path

    def set_num_clients(self, n):
        """Sets the number of total clients"""
        self.num_clients = n

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples

    def num_parameters(self, params):
        """Number of model parameters requiring training"""
        return sum(p.numel() for p in params if p.requires_grad)

    def get_model_params_norm(self):
        """Returns:
            total_params_norm: L2-norm of the model parameters"""
        total_norm = 0
        for p in self.client_model.parameters():
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_params_norm = total_norm ** 0.5
        return total_params_norm

    def get_model_grad(self):
        """Returns:
            self.total_grad: total gradient of the model (zero in case of FedAvg, where the gradient is never stored)"""
        return self.total_grad

    def get_model_grad_by_param(self):
        """Returns:
            params_grad: dictionary containing the L2-norm of the gradient for each trainable parameter of the network
                        (zero in case of FedAvg where the gradient is never stored)"""
        params_grad = {}
        for name, p in self.client_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    params_grad[name] = param_norm
                except Exception:
                    # this param had no grad
                    pass
        return params_grad
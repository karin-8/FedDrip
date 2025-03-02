import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import numpy as np
import torch
import json
import importlib
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.metrics import f1_score
import sys
from datetime import datetime, timezone, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="fedavg")
parser.add_argument("--client", type=int, default=None)
parser.add_argument("--pseudo", type=int, default=0)
parser.add_argument("--round", type=int, default=0)
parser.add_argument("--train", type=str, default=False)
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--pbar", type=int, default=0)
parser.add_argument("--log", type=str, default=None)

class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
with open("./configs/config_module_3/fedavg_config.json", "r") as conf:
    d = json.load(conf)
    args = Args(d)

model_path = args.model
dataset_path = 'dataloader'
mod = importlib.import_module(model_path)
ClientModel = getattr(mod, 'ClientModel')
dataset = importlib.import_module(dataset_path)
ClientDataset = getattr(dataset, 'ClientDataset')

def main(train_args):

    logfile = train_args.log
    random.seed(train_args.round)
    np.random.seed(train_args.round)
    torch.manual_seed(train_args.round)
    torch.cuda.manual_seed(train_args.round)
    torch.cuda.manual_seed_all(train_args.round)

    if train_args.train == 'true':
        with open(logfile, "a") as f:
            f.write(now()+ '\n')
            f.write("#"*15 + f"  Round {train_args.round+1}: Training started | site {train_args.client} " + "#"*15 + '\n')
            model_params = (args.lr, args.num_classes)
            model = ClientModel(*model_params, args.device)
            model.load_state_dict(torch.load(os.path.join("module_3", args.work_dir, f"global_model_{train_args.round}.ckpt"))["state_dict"])
            model = model.to(model.device)
        

            if train_args.pseudo == 0:
                train_data = pd.read_csv(os.path.join(args.real_csv_path, f'train_{train_args.client}.csv'))
                label_dist = (train_data[args.labels_col].sum()/train_data[args.labels_col].sum().sum()).tolist()
                test_data = pd.read_csv(os.path.join(args.real_csv_path, f'val_{train_args.client}.csv'))
                c_traindata = ClientDataset(train_data, args.labels_col, args.data_dir, sep_folder=False, transform=True)
                c_testdata = ClientDataset(test_data, args.labels_col, args.data_dir, sep_folder=False, transform=False)
                trainloader = torch.utils.data.DataLoader(c_traindata, batch_size=train_args.batch_size, shuffle=True, num_workers=args.num_workers)
                testloader = torch.utils.data.DataLoader(c_testdata, batch_size=train_args.batch_size, shuffle=False, num_workers=args.num_workers)
            else:
                train_data = pd.read_csv(os.path.join(args.pseudo_csv_path, f'train_{train_args.client}.csv'))
                label_dist = (train_data[args.labels_col].sum()/train_data[args.labels_col].sum().sum()).tolist()
                test_data = pd.read_csv(os.path.join(args.pseudo_csv_path, f'val_{train_args.client}.csv'))
                c_traindata = ClientDataset(train_data, args.labels_col, args.syn_data_dir, sep_folder=False, transform=True)
                c_testdata = ClientDataset(test_data, args.labels_col, args.syn_data_dir, sep_folder=False, transform=False)
                trainloader = torch.utils.data.DataLoader(c_traindata, batch_size=train_args.batch_size*args.parallel_clients, shuffle=True, num_workers=args.num_workers)
                testloader = torch.utils.data.DataLoader(c_testdata, batch_size=train_args.batch_size*args.parallel_clients, shuffle=False, num_workers=args.num_workers)
                
            num_train_samples = len(train_data)
            num_test_samples = len(test_data)
            f.write(f"found {num_train_samples} train samples, {num_test_samples} test samples.\n")
            f.flush()
        

            
            update, f1, val_losses = train(model, trainloader, testloader, train_args, f)
            save_info = {"update":update,
                        "samples": num_train_samples,
                        "f1":f1,
                        "val_losses":val_losses,
                        "label_dist":label_dist}
            save_name = f"round_{train_args.round+1}_site_{train_args.client}.ckpt"
            f.write(f"{now()} Round {train_args.round + 1}, finished training.\n")
            f.write("#"*60 + "\n")
            torch.save(save_info, os.path.join("module_3", args.work_dir, save_name))
    else:
        with open(logfile, "a") as f:
            f.write(now() + '\n')
            f.write("#"*15 + f"  Round {train_args.round+1}: Not training | site {train_args.client} " + "#"*15 + '\n')
            f.write("Waiting for other sites.\n")
            f.flush()
    exit()

def now():
    return str(datetime.now(tz=timezone(timedelta(hours=7))))

def train(model, trainloader, testloader, train_args, f):
    if train_args.pseudo > -1:
        criterion = nn.BCELoss().to(args.device)
    else:
        num_features = model.num_features
        model.densenet.classifier = nn.Sequential(
                                            nn.Linear(num_features, len(args.labels_col)),
                                            nn.Softmax(dim=1)
                                    )
        criterion = nn.CrossEntropyLoss().to(args.device)
        model.to(args.device)
    if train_args.pseudo == 0:
        lr = train_args.lr
        n_epochs = train_args.n_epochs
    else:
        lr = args.ps_lr_factor * train_args.lr
        """
        if args.ps_warmup_rounds != 0 and train_args.round < args.ps_warmup_rounds:
            lr = lr*(train_args.round/args.ps_warmup_rounds)
            f.write(f"Learning rate is {lr}")
        """
        n_epochs = args.pseudo_site_epochs
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    losses = np.empty(n_epochs)
    val_losses = np.empty(n_epochs)
    best_f1 = 0
    early_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        f.write(f"Training epoch {epoch}/{train_args.n_epochs}\n")
        f.flush()
        losses[epoch] = run_epoch(epoch, model, trainloader, optimizer, criterion, train_args)
        val_losses[epoch], f1 = evaluate(epoch, model, testloader, criterion, train_args)
        f.write(f"Epoch {epoch}: val_loss = {val_losses[epoch]}, f1 = {f1}\n")
        f.flush()
        # update = model.state_dict()
        if train_args.patience != 0:
            if f1 - best_f1 > 0.001:
                f.write(f"f1 improved from {best_f1} to {f1}.\n")
                best_f1 = f1
                early_counter = 0
                update = model.state_dict()
            else:
                early_counter += 1
                f.write(f"Early_counter = {early_counter}/{train_args.patience}\n")
            
            if early_counter == train_args.patience:
                f.write(f"f1 not improve for {train_args.patience} epochs. Training stopped.\n")
                break
        else:
            update = model.state_dict()
    f.flush()
    return update, best_f1, val_losses

def run_epoch(epoch, model, trainloader, optimizer, criterion, train_args):
    
    running_loss = 0
    i = 0
    
    # for j, data in enumerate(tqdm(trainloader, position=train_args.pbar, leave=True)):
    for j, data in enumerate(trainloader):
        input_data_tensor, target_data_tensor = data["image"].to(args.device), data["labels"].to(args.device)
        optimizer.zero_grad()
        outputs = model(input_data_tensor)
        loss = criterion(outputs.float(), target_data_tensor.float())
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        i += 1
    if i == 0:
        printout("Not running epoch", train_args.client)
        return 0
    return running_loss / i

def evaluate(epoch, model, testloader, criterion, train_args):
    with torch.no_grad():
        running_loss = 0
        i = 0
        y_prob = []
        y_true = []
        for j, data in enumerate(testloader):
            input_data_tensor, target_data_tensor = data["image"].to(args.device), data["labels"].to(args.device)
            outputs = model(input_data_tensor)
            loss = criterion(outputs.float(), target_data_tensor.float())
            running_loss += loss.item()
            i += 1
            y_true += [t for t in target_data_tensor.tolist()]
            y_prob += [p for p in outputs.tolist()]
        if i == 0:
            printout("Not validating", train_args.client)
            return 0
    thresholds = torch.tensor([0.5]*len(args.labels_col))
    return running_loss / i, f1_score(torch.tensor(y_true)==1, torch.tensor(y_prob)>=thresholds, average='macro', zero_division=0)
    # return running_loss / i, 0.5


if __name__ == "__main__":
    train_args = parser.parse_args()
    main(train_args)
    
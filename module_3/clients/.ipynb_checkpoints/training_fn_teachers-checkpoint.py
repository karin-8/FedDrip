import os
import argparse
import numpy as np
import torch
import json
import importlib
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim

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
parser.add_argument("--algo", type=str, default="fedavg")

class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

def main(train_args):
    with open(f"./configs/config_module_3/{train_args.algo}_config.json", "r") as conf:
        d = json.load(conf)
        args = Args(d) 

    model_path = args.model
    dataset_path = 'dataloader'
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    dataset = importlib.import_module(dataset_path)
    ClientDataset = getattr(dataset, 'ClientDataset')

    logfile = train_args.log
    if train_args.train == 'true':
        with open(logfile, "a") as f:
            f.write(now()+ '\n')
            f.write("#"*15 + f"  Round {train_args.round+1}: Training started | site {train_args.client} " + "#"*15 + '\n')
            model_params = (args.lr, args.num_classes)
            model = ClientModel(*model_params, args.device)
            teachers = []
                
            # cloud_model = ClientModel(*model_params, args.device)
            model.load_state_dict(torch.load(os.path.join("module_3", args.work_dir, f"global_model_{train_args.round}.ckpt"))["state_dict"])
            # cloud_model.load_state_dict(torch.load(os.path.join("module_3", args.work_dir, f"global_model_{train_args.round}.ckpt"))["state_dict"])
            
            # cloud_model_param = get_mdl_params([cloud_model])[0]
            
            # cloud_model_param_tensor = torch.tensor(cloud_model_param, dtype=torch.float32, device=args.device)
                        
            model = model.to(model.device)
            # cloud_model = cloud_model.to(model.device)
        
            if train_args.pseudo == 0:
                train_data = pd.read_csv(os.path.join(args.real_csv_path, f'train_{train_args.client}.csv'))
                test_data = pd.read_csv(os.path.join(args.real_csv_path, f'val_{train_args.client}.csv'))
                c_traindata = ClientDataset(train_data, args.labels_col, args.data_dir, sep_folder=False, transform=True)
                c_testdata = ClientDataset(test_data, args.labels_col, args.data_dir, sep_folder=False, transform=False)
                trainloader = torch.utils.data.DataLoader(c_traindata, batch_size=train_args.batch_size, shuffle=True, num_workers=args.num_workers)
                testloader = torch.utils.data.DataLoader(c_testdata, batch_size=train_args.batch_size, shuffle=False, num_workers=args.num_workers)
            else:
                train_data = pd.read_csv(os.path.join(args.pseudo_csv_path, f'train_{train_args.client}.csv'))
                test_data = pd.read_csv(os.path.join(args.pseudo_csv_path, f'val_{train_args.client}.csv'))
                c_traindata = ClientDataset(train_data, args.labels_col, args.syn_data_dir, sep_folder=False, transform=True)
                c_testdata = ClientDataset(test_data, args.labels_col, args.syn_data_dir, sep_folder=False, transform=False)
                trainloader = torch.utils.data.DataLoader(c_traindata, batch_size=train_args.batch_size*args.parallel_clients, shuffle=True, num_workers=args.num_workers)
                testloader = torch.utils.data.DataLoader(c_testdata, batch_size=train_args.batch_size*args.parallel_clients, shuffle=False, num_workers=args.num_workers)

                num_real_clients = args.num_clients - len(args.pseudo_site_ids)
                
                label_weights = torch.ones((num_real_clients, args.num_classes))

                for i in range(num_real_clients):
                    site = i + 1
                    model_path = os.path.join("module_3", args.work_dir, f"round_{train_args.round+1}_site_{site}.ckpt")
                    if os.path.exists(model_path):
                        teacher_model = ClientModel(*model_params, args.device)
                        teacher_dict = torch.load(model_path)
                        teacher_model.load_state_dict(teacher_dict["update"])
                        label_weights[i] = torch.tensor(teacher_dict["label_dist"])
                        teacher_model = teacher_model.to(model.device)
                        teachers.append(teacher_model)
                
            num_train_samples = len(train_data)
            num_test_samples = len(test_data)
            f.write(f"found {num_train_samples} train samples, {num_test_samples} test samples.\n")
            f.flush()
        
            trainloader = torch.utils.data.DataLoader(c_traindata, batch_size=train_args.batch_size, shuffle=True, num_workers=args.num_workers)
            testloader = torch.utils.data.DataLoader(c_testdata, batch_size=train_args.batch_size, shuffle=False, num_workers=args.num_workers)

            max_norm = args.max_norm
            
            update, f1, val_losses = train(model, (teachers, label_weights), trainloader, testloader, train_args, max_norm, f, args)
            save_info = {"update":update,
                        "samples": num_train_samples,
                        "f1":f1,
                        "val_losses":val_losses}
            save_name = f"round_{train_args.round+1}_site_{train_args.client}.ckpt"
            f.write(f"{now()} Round {train_args.round + 1}, finished training.\n")
            f.write("#"*60 + "\n")
            torch.save(save_info, os.path.join("module_3", args.work_dir, save_name))
            torch.save({"state_dict":update}, os.path.join("module_3", args.work_dir, f"local_model_site_{train_args.client}.ckpt"))
    else:
        with open(logfile, "a") as f:
            f.write(now() + '\n')
            f.write("#"*15 + f"  Round {train_args.round+1}: Not training | site {train_args.client} " + "#"*15 + '\n')
            f.write("Waiting for other sites.\n")
            f.flush()
    exit()

def now():
    return str(datetime.now(tz=timezone(timedelta(hours=7))))

def train(model, teachers, trainloader, testloader, train_args, max_norm, f, args):
    
    criterion = nn.BCELoss().to(args.device)
    if train_args.pseudo == 0:
        lr = train_args.lr
        n_epochs = train_args.n_epochs
    else:
        lr = args.ps_lr_factor * train_args.lr
        n_epochs = args.pseudo_site_epochs
        criterion_distillation = nn.BCELoss().to(args.device)
        distil_alpha = args.distil_alpha  # Weight for distillation loss
        distil_temperature = 3.0  # Temperature parameter for distillation
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    losses = np.empty(n_epochs)
    val_losses = np.empty(n_epochs)
    best_f1 = 0
    early_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        if train_args.pseudo == 0:
            losses[epoch] = run_epoch(epoch, model, teachers, trainloader, optimizer, criterion, train_args, max_norm, f, args)
            # losses[epoch] = 0.5
        else:
            losses[epoch] = run_epoch_distil(epoch, model, teachers, trainloader, optimizer, criterion, 
                          criterion_distillation, distil_alpha, distil_temperature, train_args, max_norm, f, args)
            
        f.write(f"Training epoch {epoch}/{train_args.n_epochs}\n")
        f.flush()
        val_losses[epoch], f1 = evaluate(epoch, model, testloader, criterion, train_args, f, args)
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


def run_epoch(epoch, model, teachers, trainloader, optimizer, criterion, train_args, max_norm, f, args, pseudo_site=False):
    
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


def run_epoch_distil(epoch, model, teachers_tuple, trainloader, optimizer, criterion, criterion_distillation, distil_alpha, distill_temperature,
                     train_args, max_norm, f, args, pseudo_site=False):
    
    running_loss = 0
    i = 0
    
    # for j, data in enumerate(tqdm(trainloader, position=train_args.pbar, leave=True)):
    for j, data in enumerate(trainloader):
        input_data_tensor, target_data_tensor = data["image"].to(args.device), data["labels"].to(args.device)
        optimizer.zero_grad()
        student_outputs = model(input_data_tensor)
        teachers, label_weights = teachers_tuple[0], teachers_tuple[1]
        teachers_outputs = []
        label_weights = label_weights.to(args.device)
        with torch.no_grad():  # No gradient for teacher model
            for t, teacher in enumerate(teachers):
                tcr_outputs = teacher(input_data_tensor)
                teachers_outputs.append(tcr_outputs * label_weights[t])

        teachers_outputs = sum(teachers_outputs)/len(teachers_outputs)
        loss_student = criterion(student_outputs.float(), target_data_tensor.float())
        loss_distillation = criterion_distillation(student_outputs, teachers_outputs)
        loss = distil_alpha * loss_student + (1 - distil_alpha) * loss_distillation
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        i += 1
    if i == 0:
        printout("Not running epoch", train_args.client)
        return 0
    return running_loss / i

def evaluate(epoch, model, testloader, criterion, train_args, f, args):
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
            f.write("Not validating",)
            return 0
    thresholds = torch.tensor([0.5]*len(args.labels_col))
    return running_loss / i, f1_score(torch.tensor(y_true)==1, torch.tensor(y_prob)>=thresholds, average='macro', zero_division=0)
    # return running_loss / i, 0.5

def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for param in exp_mdl.parameters():
            n_par += len(param.reshape(-1))
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for param in mdl.parameters():
            temp = param.reshape(-1).detach().cpu().numpy()
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

if __name__ == "__main__":
    train_args = parser.parse_args()
    main(train_args)
    
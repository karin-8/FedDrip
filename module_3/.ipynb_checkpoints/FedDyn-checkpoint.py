import importlib
import json
import numpy as np
import os

import pandas as pd
from datetime import datetime, timezone, timedelta
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.args import parse_args, check_args
from utils.cutout import Cutout
from utils.main_utils import *
from utils.model_utils import read_data

from sklearn.metrics import roc_auc_score

import logging
import sys

class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
with open("./configs/config_module_3/feddyn_config.json", "r") as conf:
    d = json.load(conf)
    args = Args(d)

def main(args):
    project_name = args.project_name
    experiment = args.experiment
    
    
    
    now = str(datetime.now(tz=timezone(timedelta(hours=7)))).replace(" ","_")
    if not os.path.exists("./module_3/logs/%s" % project_name):
        os.makedirs("./module_3/logs/%s" % project_name)
    logname = f"./module_3/logs/{project_name}/{now}.log"
    
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    logging.info(f"Running {project_name}")
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    site_logs = []
    if not os.path.exists("./module_3/logs/%s/clients/" % project_name):
        os.makedirs("./module_3/logs/%s/clients/" % project_name)
    for site in range(1, args.num_clients+1):
        if not os.path.exists("./module_3/logs/%s/clients/site_"  % project_name + str(site)):
            os.makedirs("./module_3/logs/%s/clients/site_"  % project_name + str(site))
        site_logs.append(f"./module_3/logs/{project_name}/clients/site_{site}/{now}.log")
    if not os.path.exists("./module_3/output"):
        os.makedirs("./module_3/output")
    
    
    writer = SummaryWriter(log_dir=f"./module_3/runs/{project_name}_{experiment}/")
        
    # Set the random seed if provided (affects client sampling and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    logger.info(f"Using device: {torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu'}")
    
    model_path = './module_3/%s/%s.py' % ('clients', args.model)
    dataset_path = './module_3/%s/%s.py' % ('clients', 'dataloader')
    server_path = './module_3/servers/%s.py' % (args.algorithm + '_server')
    client_path = './module_3/clients/%s.py' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    check_init_paths([model_path, dataset_path, server_path, client_path])
    
    model_path = '%s.%s' % ('clients', args.model)
    dataset_path = '%s.%s' % ('clients', 'dataloader')
    server_path = 'servers.%s' % (args.algorithm + '_server')
    client_path = 'clients.%s' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    
    # Load model and dataset
    logger.info(f'############################## {model_path} ##############################')
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    dataset = importlib.import_module(dataset_path)
    ClientDataset = getattr(dataset, 'ClientDataset')
    
    # Load client and server
    logger.info(f"Running experiment with server {server_path} and client {client_path}")
    Client, Server = get_client_and_server(server_path, client_path)
    logger.info(f"Verify client and server: {Client} {Server}")
    
    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    # tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    
    model_params = (args.lr, args.num_classes)
    
    # Create client model, and share params with servers model
    client_model = ClientModel(*model_params, device)
    # client_model = client_model.to(device)
    torch.save({"state_dict":client_model.state_dict()}, os.path.join("module_3", args.work_dir, "global_model_0.ckpt"))
    for i in range(1, args.num_clients+1):
        torch.save({"state_dict":client_model.state_dict()}, os.path.join("module_3", args.work_dir, f"local_model_site_{i}.ckpt"))
    
    # Get synthetic dataset if any
    # syn_ds = get_synthetic_dataset(ClientDataset, args)
    val_df = pd.read_csv(args.global_val_csv_path)
    val_loader = torch.utils.data.DataLoader(ClientDataset(val_df, args.labels_col, args.data_dir), batch_size=args.batch_size, shuffle=False)
    
    #### Create server ####
    server_params = define_server_params(args, client_model, args.algorithm)
    server = Server(**server_params)
    
    #### Create and set up clients ####
    train_clients, test_clients = setup_clients(args, client_model, Client, ClientDataset, run=None, device=device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    test_client_ids, test_client_num_samples = server.get_clients_info(test_clients)
    if set(train_client_ids) == set(test_client_ids):
        logger.info(f"Clients in Total: {len(train_clients)}")
    else:
        logger.info(f"Clients in Total: {len(train_clients)} training clients and {len(test_clients)} test clients")
    
    server.set_num_clients(len(train_clients))
    trained_clients = set()
    
    if not args.start_round:
        start_round = 0 
    else:
        start_round = args.start_round - 1
        for i in range(start_round):
            a,b = server.select_clients(i, online(train_clients), num_clients=clients_per_round)
    logger.info(f"Start round: {start_round + 1}")
    
    
    # Initial status
    logger.info('--- Random Initialization ---')
    
    start_time = datetime.now()
    current_time = start_time.strftime("%m%d%y_%H:%M:%S")

    # Start training
    for i in range(start_round, num_rounds):
        logger.info(f'--- Round {i+1} of {num_rounds}: Training {clients_per_round} Clients ---')
    
        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        trained_clients.update(c_ids)
        logger.info(f"Selected clients: {sorted(c_ids)}")
    
        ##### Simulate servers model training on selected clients' data #####
        test_metrics = server.train_model(rounds=i, writer=writer, where_logs=site_logs, num_epochs=args.num_epochs, batch_size=args.batch_size, patience=args.patience,
                                         minibatch=args.minibatch)
    
        ##### Update server model (FedAvg) #####
        logger.info("--- Updating central model ---")
        server.update_model()
    
        torch.save({"state_dict":server.client_model.state_dict()}, os.path.join("module_3", args.work_dir, f"global_model_{i+1}.ckpt"))
        ##### Test model #####
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            criterion = nn.BCELoss().to(args.device)
            loss, auc = evaluate(i+1, server.client_model, val_loader, criterion, args)
            if i+1 >= args.lr_decay_start:
                server.lr = server.lr*args.lr_decay_per_round
            writer.add_scalar("Global loss", loss, i+1)
            writer.add_scalar("Global AUC", auc, i+1)
            if not os.path.isfile(os.path.join("module_3", args.work_dir, "losses.json")):
                with open(os.path.join("module_3", args.work_dir, "losses.json"), "w") as f:
                    f.write(json.dumps({"loss":[loss], "auc":[auc]}))
            else:
                with open(os.path.join("module_3", args.work_dir, "losses.json"), "r") as f:
                    d = json.load(f)
                if i == start_round:
                    d["loss"] = d["loss"][:start_round]
                    d["auc"] = d["auc"][:start_round]
                d["loss"].append(loss)
                d["auc"].append(auc)
                with open(os.path.join("module_3", args.work_dir, "losses.json"), "w") as f:
                    f.write(json.dumps(d))
    
    # Save results
    writer.close()
    exit(1)
    
def evaluate(epoch, model, testloader, criterion, args):
    with torch.no_grad():
        model = model.to(args.device)
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
            print("Not validating")
            return 0
        del model
        torch.cuda.empty_cache()
    return running_loss / i, roc_auc_score(y_true, y_prob)

if __name__=="__main__":
    main(args)
    # try:
        # main(args)
    # except Exception as e:
        # print(e)
        # exit(0)


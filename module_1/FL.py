import os
import pandas as pd
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import sys
import torch

from utils.server import Server, Client

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

def main(fl_conf, df_conf):

    now = str(datetime.now()).replace(" ", "_")
    logpath = "./module_1/logs/server/"
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logname = logpath + now + ".log"
    
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    project_name = fl_conf.project
    
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    writer = SummaryWriter(log_dir="./module_1/tfevents")

    logger.info(f"Running {project_name}")

    ### Loading Global Model ###
    logger.info("Loading global model...")
    global_model = torch.load(df_conf.ldm_checkpoint_path)

    ### Create working directory if not one ###
    if not os.path.exists(fl_conf.work_dir):
        os.makedirs(fl_conf.work_dir)

    ### Create server ###
    logger.info("Creating server...")
    server = Server(global_model, fl_conf)
    server.save_model(0, os.path.join(fl_conf.work_dir, "global_model_round_0.pt")) # save first model

    ### Set up clients ###
    logger.info("Setting up clients...")
    clients = []
    where_log = {}
    for i in range(fl_conf.num_clients):
        c = i+1
        train_df = pd.read_csv(os.path.join(fl_conf.split_csv_path, f"train_{c}.csv"))
        val_df =  pd.read_csv(os.path.join(fl_conf.split_csv_path, f"val_{c}.csv"))
        clients.append(Client(c, train_df, val_df))
        where_log[c] = "./module_1/logs/clients/" + str(c) + "/"

    if not fl_conf.start_round:
        start_round = 0 
    else:
        start_round = fl_conf.start_round - 1
        for i in range(start_round):
            a,b = server.select_clients(i, clients, fl_conf.num_clients)
    logger.info(f"Start round: {start_round + 1}")

    ### Begin FL ###
    logger.info("Start FL")
    for r in range(start_round, fl_conf.fl_rounds):
        logger.info(f"Round {r+1}")
        selected_clients = server.select_clients(r, clients, fl_conf.num_clients) # select clients
        server.train_model(r, writer, where_log, fl_conf.num_epochs, fl_conf.num_steps, df_conf.batch_size, df_conf.patience) # train
        server.update_model(r) # FedAVG
        if r%fl_conf.eval_every != 0:
            os.remove(os.path.join(fl_conf.work_dir, f"global_model_round_{r}.pt"))
        if r+1 != fl_conf.fl_rounds:
            server.save_model(r+1, os.path.join(fl_conf.work_dir, f"global_model_round_{r+1}.pt")) # save model
        else:
            server.save_model(r+1, os.path.join(fl_conf.work_dir, "final_model.pt"))
        if (r+1)%fl_conf.eval_every == 0:
            server.eval_model(r+1, fl_conf.global_eval_mode) # evaluate
            with open(os.path.join(fl_conf.work_dir, "losses.json"), "r") as f:
                d = json.load(f)
            if fl_conf.global_eval_mode == 'loss' and  d["losses"][-1] == min( d["losses"]):
                server.save_model(r+1, os.path.join(fl_conf.work_dir, "best_model.pt"))
            elif fl_conf.global_eval_mode == 'fid' and d["fids"][-1] == min(d["fids"]):
                server.save_model(r+1, os.path.join(fl_conf.work_dir, "best_model.pt"))

        server.clear_model()
        logger.info(f"Finished Round {r+1}")
    exit(1)

if __name__=="__main__":
    try:
        main(fl_conf, df_conf)
    except Exception as e:
        print(e)
        exit(0)

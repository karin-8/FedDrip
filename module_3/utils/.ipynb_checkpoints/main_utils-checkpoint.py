import copy
import matplotlib.pyplot as plt
import os
import torch

import importlib
import numpy as np
import pandas as pd
from datetime import datetime
import random
import inspect
from utils.model_utils import read_data


def create_paths(args, current_time, alpha=None, resume=False):
    """ Create paths for checkpoints, plots, analysis results and experiment results. """
    ckpt_path = os.path.join('ckpts', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # Create file for storing results
    res_path = os.path.join('results', args.dataset, args.model)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    run_info = 'K' + str(args.clients_per_round) + '_N' + str(args.num_rounds) + '_clr' + str(args.lr) + '_' +\
                args.algorithm

    run_info += '_' + current_time

    ckpt_name = None
    if alpha is not None:
        file = os.path.join(res_path, 'results_' + str(alpha) + run_info +  '.txt')
        if not resume:
            ckpt_name = os.path.join(ckpt_path, '{}.ckpt'.format(str(alpha) + run_info))
    else:
        file = os.path.join(res_path, 'results_' + run_info + '.txt')
        if not resume:
            ckpt_name = os.path.join(ckpt_path, '{}.ckpt'.format(run_info))

    return ckpt_path, res_path, file, ckpt_name

def get_run_checkpoint(run, dataset, restart_round=None):
    api = wandb.Api()
    run_path = run.entity + '/' + run.project + '/' + run.id
    run_api = api.run(run_path)
    ckptpath = os.path.join('ckpts', dataset)
    ckpt, final_path = None, None
    for file in run_api.files():
        if file.name.startswith(ckptpath) and file.name.endswith('.ckpt'):
            if (restart_round is None and 'round' not in file.name) or (restart_round is not None and 'round:' + str(restart_round) + '_' in file.name):
                ckpt = run.restore(file.name, run_path=run_path)
                final_path = file.name.split('/')[-1]
                # break
    print("Restored checkpoint:", final_path)
    return ckpt, final_path

def resume_run(client_model, args, run):
    # print("--- Loading model", CHECKPOINT, "from checkpoint ---")
    print("Resuming run", run.id)
    ckpt, ckpt_path_resumed = get_run_checkpoint(run, args.dataset, args.restart_round)
    if ckpt is None:
        print("Checkpoint not found")
        exit(-2)
    checkpoint = torch.load(ckpt.name)
    client_model.load_state_dict(checkpoint['model_state_dict'])
    return client_model, checkpoint, ckpt_path_resumed

def get_alpha(dataset):
    data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    train_file = os.listdir(data_dir)
    if not train_file:
        print("Expected training file. Not found.")
        exit(-1)
    alpha = train_file[0].split('train_')[1][:-5]
    return alpha

def check_init_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            print("The path", path, "does not exist. Please specify a valid one.")
            exit(-1)

def define_server_params(args, client_model, server_name):
    if server_name == 'fedavg':
        server_params = {'args': args, 'client_model': client_model}
    elif server_name == 'feddyn':
        server_params = {'args': args, 'client_model': client_model}
    elif server_name == 'scaffold':
        server_params = {'args': args, 'client_model': client_model}
    elif server_name == 'fedprox':
        server_params = {'args': args, 'client_model': client_model}
    else:
        raise NotImplementedError
    return server_params

def define_client_params(client_name, args):
    client_params = {'seed': args.seed, 'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
                     'num_workers': args.num_workers}

    return client_params

def schedule_cycling_lr(round, c, lr1, lr2):
    t = 1 / c * (round % c + 1)
    lr = (1 - t) * lr1 + t * lr2
    return lr

def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn

def plot_metrics(accuracy, loss, n_rounds, figname, figpath, title, prefix='val_'):
    name = os.path.join(figpath, figname)

    plt.plot(n_rounds, loss, '-b', label=prefix + 'loss')
    plt.plot(n_rounds, accuracy, '-r', label=prefix + 'accuracy')

    plt.xlabel("# rounds")
    plt.ylabel("Average accuracy")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(name + '.png')  # should before show method
    plt.close()

def get_plots_name(args, current_time, alpha=None):
    if alpha is None:
        img_name_val = 'val_N' + str(args.num_rounds) + '_K' + str(args.clients_per_round) + '_lr' + str(
            args.lr) + current_time
        img_name_test = 'test_N' + str(args.num_rounds) + '_K' + str(args.clients_per_round) + '_lr' + str(
            args.lr) + current_time
    else:
        img_name_val = str(alpha) + '_val_N' + str(args.num_rounds) + '_K' + str(
            args.clients_per_round) + '_lr' + str(args.lr) + current_time
        img_name_test = str(alpha) + '_test_N' + str(args.num_rounds) + '_K' + str(
            args.clients_per_round) + '_lr' + str(args.lr) + current_time
    return img_name_val, img_name_test

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, train_data, test_data, model, args, ClientDataset, Client, run=None, device=None):
    clients = []
    client_params = define_client_params(args.client_algorithm, args)
    client_params['model'] = model
    client_params['run'] = run
    client_params['device'] = device
    for u in users:
        if u not in args.pseudo_site_ids:
            c_traindata = ClientDataset(train_data[u], args.labels_col, args.data_dir, sep_folder=False, transform=True)
            c_testdata = ClientDataset(test_data[u], args.labels_col, args.data_dir, sep_folder=False, transform=False)
        else:
            c_traindata = ClientDataset(train_data[u], args.labels_col, args.syn_data_dir, sep_folder=False, transform=True)
            c_testdata = ClientDataset(test_data[u], args.labels_col, args.syn_data_dir, sep_folder=False, transform=False)
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        client_params['eval_data'] = c_testdata
        clients.append(Client(**client_params))
    return clients


def setup_clients(args, model, Client, ClientDataset, run=None, device=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """

    train_users = [site+1 for site in range(args.num_clients)]
    test_users = [site+1 for site in range(args.num_clients)]
    train_data = {}
    test_data = {}
    for site in range(args.num_clients):
        train_data[site+1] = pd.read_csv(os.path.join(args.real_csv_path , f'train_{site+1}.csv'))
        test_data[site+1] = pd.read_csv(os.path.join(args.real_csv_path , f'val_{site+1}.csv'))

    train_clients = create_clients(train_users, train_data, test_data, model, args, ClientDataset, Client, run, device)
    test_clients = create_clients(test_users, train_data, test_data, model, args, ClientDataset, Client, run, device)

    return train_clients, test_clients

def get_client_and_server(server_path, client_path):
    mod = importlib.import_module(server_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    server_name = server_path.split('.')[1].split('_server')[0]
    server_name = list(map(lambda x: x[0], filter(lambda x: 'Server' in x[0] and server_name in x[0].lower(), cls)))[0]
    Server = getattr(mod, server_name)
    mod = importlib.import_module(client_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    client_name = max(list(map(lambda x: x[0], filter(lambda x: 'Client' in x[0], cls))), key=len)
    Client = getattr(mod, client_name)
    return Client, Server

def print_stats(num_round, server, train_clients, train_num_samples, test_clients, test_num_samples, test_stat_metrics, args, logger, fp):

    # test_stat_metrics = server.test_model(test_clients, args.batch_size, set_to_use='test' )
    test_metrics = print_metrics(test_stat_metrics, test_num_samples, logger, fp, prefix='{}_'.format('test'))

    logger.info(f"Validation auc: {test_metrics[0]}, Validation loss: {test_metrics[1]}, round: {num_round}") 
    # return val_metrics, test_metrics
    return test_metrics


def print_metrics(metrics, weights, logger, fp, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    metrics_values = {}
    for m in metrics:
        metrics_name = m["name"]
        metrics_values[metrics_name] = []
        for c in m["values"].keys():
            if not isinstance(c, float):
                val = np.average(m['values'][c])
            else:
                val = m['values'][c]
            logger.info(f"Client {c}, {metrics_name} = {val}")
            fp.write(f"Client {c}, {metrics_name} = {val}")
            metrics_values[metrics_name].append(val)
    return [np.average(metrics_values[k]) for k in metrics_values.keys()]
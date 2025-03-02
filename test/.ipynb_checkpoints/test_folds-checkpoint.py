import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import os

from collections import OrderedDict

import pandas as pd
from dataloader import ClientDataset

from PIL import Image
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
import json

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--folds", nargs='+', type=int, help="List of numbers")
parser.add_argument("--algo", type=str)
parser.add_argument("--round", type=int, default=0)

def find_max(metric):
    pointer = 0
    best = -1e6
    for i in metric:
        if i > best:
            best = i
            out = (best, pointer)
        pointer += 1
    return out


labels_col = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', \
   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', \
   'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet(x)
        return x
        
def test(fold,algo,size,pseudo,overwrite,round_sel):
    
    run_dir = f"./module_3/{fold}/{size[0]}{algo}{pseudo}"
    
    with open(f"{run_dir}/losses.json") as f:
        d = json.load(f)
        aucs = d["auc"][:13]

    
    if overwrite:
        for filename in os.listdir(run_dir):
            # Check if the file starts with 'sauc' or 'sclass'
            if filename.startswith("sauc") or filename.startswith("sclass"):
                file_path = os.path.join(run_dir, filename)
                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
    if round_sel:
        best_round = round_sel
    else:
        best_round = find_max(aucs)[1] + 1
    # best_round = 4

    

    model = 'densenet'
    model_dir = f"{run_dir}/global_model_{best_round}.ckpt"
    
    global_model = DenseNet121(len(labels_col))
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    test_df = pd.read_csv(f'./splits/{fold}/{size}/test.csv')
    val_df = pd.read_csv(f'./splits/{fold}/{size}/val.csv')
    
    data_dir = "/karin/project/fdl/data"
    
    test_ds = ClientDataset(test_df, labels_col, data_dir, sep_folder=False, transform=False)
    test_dl_global= data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    val_ds = ClientDataset(val_df, labels_col, data_dir, sep_folder=False, transform=False)
    val_dl_global= data.DataLoader(val_ds, batch_size=32, shuffle=False)
    
    state_dict = torch.load(model_dir)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    global_model.load_state_dict(new_state_dict)
    """
    print("Validating...")
    global_model.to(device)
    y_pred_val = []
    y_true_val = []
    y_prob_val = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dl_global)):
            img = batch['image'].to(device)
            targets = batch['labels'].to(device)
            prob = global_model(img)
            y_true_val += [targ for targ in targets.tolist()]
            y_prob_val += [proba for proba in prob.tolist()]
    
    thres = pd.DataFrame(columns=['threshold', 'f1', 'precision', 'recall'])
    
    y_true_t = torch.tensor(y_true_val)
    y_prob_t = torch.tensor(y_prob_val)
    
    for idx, l in enumerate(labels_col):
        xs = torch.linspace(0.01,0.99,100)
        f1s = [f1_score(y_true_t[:,idx]==1, y_prob_t[:,idx]>=i, average='macro', zero_division=0) for i in xs]
        precs = [precision_score(y_true_t[:,idx]==1, y_prob_t[:,idx]>=i, average='macro', zero_division=0) for i in xs]
        recs = [recall_score(y_true_t[:,idx]==1, y_prob_t[:,idx]>=i, average='macro', zero_division=0) for i in xs]
        xmax = xs[torch.argmax(torch.tensor(f1s))]
        thres.loc[l, 'threshold'] = xmax
        thres.loc[l, 'f1'] = max(f1s)
        thres.loc[l, 'precision'] = precs[torch.argmax(torch.tensor(f1s))]
        thres.loc[l, 'recall'] = recs[torch.argmax(torch.tensor(f1s))]
    thresholds = torch.tensor(thres["threshold"].array)
    """
    thresholds = torch.ones(len(labels_col))/2
    
    print("Testing...")
    global_model.to(device)
    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dl_global)): 
            img = batch['image'].to(device)
            targets = batch['labels'].to(device)
            prob = global_model(img)
            y_prob += [proba for proba in prob.tolist()]
    
    y_true = test_df[labels_col].to_numpy()
    report = classification_report(torch.tensor(y_true)==1, torch.tensor(y_prob)>=thresholds, digits=4, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{run_dir}/sclass_rep_r{best_round}.csv')
    report_print = classification_report(torch.tensor(y_true)==1, torch.tensor(y_prob)>=thresholds, digits=4, target_names=labels_col, zero_division=0)
    print("classification report:")
    print(report_print)
    
    report_with_auc = roc_auc_score(
        y_true=y_true,  
        y_score=y_prob,
        average=None)
    
    list_report = report_with_auc.tolist()
    list_report += [sum(list_report)/len(list_report)]
    max_f1s = [report[str(i)]['f1-score'] for i in range(len(labels_col))] + [report['macro avg']['f1-score']]
    data_auc = {'class':labels_col + ['macro avg'],'f1-score':max_f1s , 'auc':list_report}
    auc = pd.DataFrame(data=data_auc)
    auc.to_csv(f'{run_dir}/sauc_r{best_round}.csv')
    print("overall report:")
    print(auc)

args = parser.parse_args()

# folds = [f"fold_{i}" for i in range(1,6)]
folds = [f"fold_{i}" for i in args.folds]
# folds = ["fold_1"]
# sizes = ["half", "full"]
sizes = ["half"]
pseudo = [1]
algo = args.algo
ow = args.overwrite
        
for fold in folds:
    for a in algo:
        for s in sizes:
            for p in pseudo:
                test(fold,a,s,p,ow,args.round)


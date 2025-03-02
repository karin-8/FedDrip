import pandas as pd
import random
import json

def get_report(prompt_path, df, row):
    with open(prompt_path, "r") as f:
        report_dict = json.load(f)
    labels = df.columns[-15:].tolist()
    onehot = row[-15:].tolist()
    disease_list = [labels[idx] for idx in range(len(onehot)) if onehot[idx] != 0]
    report = list()
    for disease in disease_list:
        report += report_dict[disease.replace("_", " ")]
    selected = random.sample(report, k=min(len(report),5))
    return ". ".join(report) + "."
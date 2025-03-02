import pandas as pd
import random
import json

def get_report(prompt_path, df, row, args):
    with open(prompt_path, "r") as f:
        report_dict = json.load(f)
    labels = df.columns[-15:].tolist()
    onehot = row[-15:].tolist()
    disease_list = [labels[idx] for idx in range(len(onehot)) if onehot[idx] != 0]
    if args.prompt_style in ["term", "single-term", "single_term"]:
        disease_list = [labels[idx] for idx in range(len(onehot)) if onehot[idx] != 0]
        return ", ".join(disease_list) + "."
    elif args.prompt_style in ["keyword", "kw"]:
        report = list()
        for disease in disease_list:
            report.append(report_dict[disease.replace("_", " ")])
        return ", ".join(report) + "."
    elif args.prompt_style in ["full", "full-sentence", "full_sentence"]:
        report = list()
        for disease in disease_list:
            report += report_dict[disease.replace("_", " ")]
        selected = random.sample(report, k=min(len(report),5))
        return ". ".join(report) + "."
    else:
        raise ValueError("Invalid prompt style")
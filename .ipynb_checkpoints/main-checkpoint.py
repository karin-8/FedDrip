import os
import numpy as np
import os 
import torch
import subprocess as subp
import json
import sys

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
with open("./configs/main_config.json", "r") as conf:
    d = json.load(conf)
    conf = Config(d)

def main(config):
    # Begin Module 1
    if 1 in conf.modules:
        print("Starting module 1")
        proc = subp.run(["python", "module_1/FL.py"])
        output = proc.stdout
        return_code = proc.returncode
        if return_code == 0:
            print("Error in module 1:")
            print(output)
            exit(0)
    if 2 in conf.modules:
        # Begin Module 2
        p1 = subp.run(["python", "module_2/pseudosite.py"])
        return_code1 = p1.returncode
        p2 = subp.run(["python", "module_2/calculate.py"])
        return_code2 = p2.returncode
        if not return_code1 or not return_code2:
            print("Error in ldm_test:")
            exit(0)
        else:
            print("Module 2 complete")

    if 3 in conf.modules:
        print("Starting module 3")
        if conf.fl_algo.lower() == "fedavg":
            proc = subp.run(["cp", "configs/config_module_3/fedavg_config.json", "module_3/output"])
            proc = subp.run(["python", "module_3/FedAvg.py"])
        if config.fl_algo.lower() == "fedprox":
            proc = subp.run(["cp", "configs/config_module_3/fedprox_config.json", "module_3/output"])
            proc = subp.run(["python", "module_3/FedProx.py"])
        if config.fl_algo.lower() == "feddyn":
            proc = subp.run(["cp", "configs/config_module_3/feddyn_config.json", "module_3/output"])
            proc = subp.run(["python", "module_3/FedDyn.py"])

    proc = subp.run(["python", "test/test.py"])
    # proc = subp.run(["python", "test/test_search.py"])
    print("Process complete.")

    

if __name__=="__main__":
    main(conf)
    
    
    
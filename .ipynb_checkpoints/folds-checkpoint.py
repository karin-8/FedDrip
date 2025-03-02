import json
import subprocess
import time
import itertools
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--folds", nargs='+', type=int, help="List of numbers")
# Load JSON configuration

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Save JSON configuration
def save_config(file_path, config):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

# Generate parameter combinations
def generate_combinations(param_ranges):
    keys = param_ranges.keys()
    values = (param_ranges[key] for key in keys)
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


# Create an empty output directory
def create_empty_output_directory(base_path='./module_3/output'):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)  # Remove existing directory if it exists
    os.makedirs(base_path, exist_ok=True)  # Create a new empty directory
    print(f"Created new empty output directory: {base_path}")

# Run experiment with given parameters
def run_experiment(config_file, parameters, overwrite):
    config = load_config(config_file)
    config.update(parameters)
    save_config(config_file, config)
    
    # Check if already run
    base_path='./module_3/output'
    param_str = '_'.join(f"{key}={value}" for key, value in parameters.items())
    new_dir_name = f"./module_3/fold_{parameters['fold']}/{parameters['size'][0]}{parameters['algo']}{parameters['num_clients']%2}"
    # if os.path.exists(new_dir_name):
    #     print(f"{param_str} already run.")
    #     return
    # shutil.move(new_dir_name, base_path)
    if os.path.exists(new_dir_name) and os.path.isdir(new_dir_name):
    # Check if the directory is not empty
        if "losses.json" in os.listdir(new_dir_name):
            print(f"Already run")
            if not overwrite:
                return

    # Create empty output directory for the experiment
    create_empty_output_directory()
    
    # Run the experiment and capture output in real-time
    start_time = time.time()
    print("Running experiment...")
    process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print real-time output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.stdout.close()
    process.wait()

    # process = subprocess.run(['python', 'test/test_search.py'])
    # shutil.move(base_path, new_dir_name)
    rename_output_directory(parameters, "./module_3/output")
    
    # Check for any errors
    # stderr_output = process.stderr.read()
    # if stderr_output:
        # print(f"Errors:\n{stderr_output}")

# Rename output directory based on parameters
def rename_output_directory(params, base_path='./module_3/output'):
    # param_str = '_'.join(f"{key}={value}".replace("/","-") for key, value in params.items())
    new_dir_name = f"./module_3/fold_{params['fold']}/{params['size'][0]}{params['algo']}{params['num_clients']%2}"
    if os.path.exists(base_path):
        shutil.copytree(base_path, new_dir_name, dirs_exist_ok=True)
    print(f"Output directory renamed to: {new_dir_name}")

# Create an empty output directory
def create_empty_output_directory(base_path='./module_3/output'):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)  # Remove existing directory if it exists
    os.makedirs(base_path, exist_ok=True)  # Create a new empty directory
    print(f"Created new empty output directory: {base_path}")

# Main function for grid search
def grid_search(config_file, param_ranges, overwrite):
    combinations = generate_combinations(param_ranges)
    results = []
    
    for params in combinations:
        params["syn_data_dir"] = f"module_2/pseudo-site_f{params['fold']}"
        params["real_csv_path"] = f"splits/fold_{params['fold']}/{params['size']}"
        params["global_val_csv_path"] = f"splits/fold_{params['fold']}/{params['size']}/val.csv"
        params["pseudo_site_ids"] = [7] if params['num_clients'] == 7 else []
        print(f"Running experiment with parameters: {params}")
        output = run_experiment(config_file, params, overwrite)
        results.append((params, output))
        print(f"Output: {output}")
    
    return results

other_clients = [1]*6

# Define parameter ranges for grid search
"""
param_ranges = {
    'size':['very_scarce'],
    'num_clients':[6,7],
    'fold':[1],
    'algo':['a'],
    'client_weights':[other_clients + [w] for w in [1,0.5,0.1]],
    'pseudo_site_every':[1,3,5],
    'pseudo_site_start':[3],
    'ps_wamup_rounds':[0,5],
    'pseudo_site_epochs':[3,6],
    
}
"""

# Run grid search
if __name__ == "__main__":
    args = parser.parse_args()
    param_ranges = {
        'fold':args.folds,
        'num_clients':[7],
        'algo':['a'],
        'size':['half'],
    }
    config_file = './configs/config_module_3/fedavg_config.json'
    results = grid_search(config_file, param_ranges, args.overwrite)
    # Optionally, save or analyze results
    # e.g., save to a file or print the best result

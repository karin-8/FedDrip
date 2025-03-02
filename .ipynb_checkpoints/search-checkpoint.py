import json
import subprocess
import time
import itertools
import os
import shutil

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
def run_experiment(config_file, parameters):
    config = load_config(config_file)
    config.update(parameters)
    save_config(config_file, config)
    
    # Check if already run
    base_path='./module_3/output'
    param_str = '_'.join(f"{key}={value}" if type(value) != list else f"weight={value[-1]}" for key, value in parameters.items())
    new_dir_name = f"{base_path}_{param_str}"
    if os.path.exists(new_dir_name):
        print(f"{param_str} already run.")
        return
    # shutil.move(new_dir_name, base_path)

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

    process = subprocess.run(['python', 'test/test_search.py'])
    # shutil.move(base_path, new_dir_name)
    rename_output_directory(parameters, "./module_3/output")
    
    # Check for any errors
    # stderr_output = process.stderr.read()
    # if stderr_output:
        # print(f"Errors:\n{stderr_output}")

# Rename output directory based on parameters
def rename_output_directory(params, base_path='./module_3/output'):
    param_str = '_'.join(f"{key}={value}" if type(value) != list else f"weight={value[-1]}" for key, value in params.items())
    new_dir_name = f"{base_path}_{param_str}"
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
def grid_search(config_file, param_ranges):
    combinations = generate_combinations(param_ranges)
    results = []
    
    for params in combinations:
        print(f"Running experiment with parameters: {params}")
        output = run_experiment(config_file, params)
        results.append((params, output))
        print(f"Output: {output}")
    
    return results

other_clients = [1]*6

# Define parameter ranges for grid search
param_ranges = {
    'client_weights': [other_clients + [i] for i in [0.1, 0.5]],
    'ps_warmup_rounds': [1, 2, 3],
    'pseudo_site_epochs': [1, 2],
    'ps_lr_factor': [0.5, 1],
    'pseudo_site_start':[3]
}

# Run grid search
if __name__ == "__main__":
    config_file = './configs/config_module_3/fedavg_config.json'
    results = grid_search(config_file, param_ranges)
    # Optionally, save or analyze results
    # e.g., save to a file or print the best result

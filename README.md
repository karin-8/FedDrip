# FedDrip: Federated Learning with Diffusion-Generated Synthetic Data [paper](https://ieeexplore.ieee.org/document/10824802).

## Overview

**FedDrip** is a novel federated learning (FL) framework designed to enhance model performance in non-IID environments by incorporating **diffusion-generated synthetic data** and **knowledge distillation**. This approach improves model generalization and robustness while preserving data privacy.

Federated learning is essential for privacy-sensitive domains like healthcare, where data cannot be shared across institutions. However, non-IID data distributions and data scarcity pose significant challenges. FedDrip mitigates these issues by introducing a **pseudo-site** that leverages diffusion models to generate synthetic data, acting as a virtual client in the FL process.

## Features

- **Federated Learning Integration**: Enhances traditional FL frameworks (FedAvg, FedDyn, FedProx) with a pseudo-site mechanism.
- **Synthetic Data Generation**: Uses diffusion models to create high-quality synthetic data to address data heterogeneity and scarcity.
- **Knowledge Distillation**: Transfers knowledge from real clients to the pseudo-site to improve global model performance.
- **Privacy Preservation**: Ensures no raw data is shared while still improving model generalization.
- **Scalability**: Works seamlessly across multiple FL frameworks and data settings.

## Architecture

FedDrip consists of three core modules:
1. **Diffusion Model Training**: A federated diffusion model is trained across all clients to learn the global data distribution without violating privacy constraints.
2. **Pseudo-Site Data Generation**: The global diffusion model generates synthetic images that are used for training at the pseudo-site.
3. **Federated Learning with Pseudo-Site**: The pseudo-site participates in FL training alongside real clients, improving overall model performance through knowledge distillation.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch 2.1.2+
- CUDA 12.1+ (for GPU acceleration)
- Required dependencies (install using the command below)

### Setup
Clone the repository:
```bash
git clone https://github.com/karin-8/FedDrip.git
cd FedDrip
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure Settings
Modify the configuration files in the `configs/` directory to set hyperparameters, data paths, and experiment settings.

### 2. Prepare Data
Ensure the dataset is available in the specified location and formatted according to the configuration settings.

### 3. Run Federated Learning Training
Start the FL process with the following command:
```bash
python main.py
```

Monitor training logs in the `logs/` directory and model checkpoints in `checkpoints/`.

## Repository Structure

```
FedDrip/
│── configs/                 # Configuration files
│── module_1/                # Diffusion Model Training
│── module_2/                # Pseudo-Site Data Generation
│── module_3/                # Pseudo-Site Federated Learning
│── test/                    # Testing scripts
│── main.py                  # Main execution script
│── folds.py                 # Data processing utility
│── search.py                # Model search utility
│── zip.py                   # Utility scripts
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
```

## Experimental Results
Experiments conducted on the **NIH ChestX-ray14** dataset show that FedDrip improves FL model performance:
- **FedAvg**: +2.15% AUC improvement
- **FedDyn**: +0.95% AUC improvement
- **FedProx**: +1.96% AUC improvement

Performance improvements are most significant under extreme data scarcity conditions (400 images per site).

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use FedDrip in your research, please cite our paper:

```
@article{FedDrip2024,
  author    = {Karin Huangsuwan, Timothy Liu, Simon See, Aik Beng Ng, Peerapon Vateekul},
  title     = {FedDrip: Federated Learning with Diffusion-Generated Synthetic Data},
  journal   = {arXiv},
  year      = {2024}
}
```

## Acknowledgments
This research was funded by the **Department of Computer Engineering, Faculty of Engineering, Chulalongkorn University**, under the **Chula Computer Engineering Graduate Scholarship**.

For more details, refer to our [paper](https://ieeexplore.ieee.org/document/10824802).

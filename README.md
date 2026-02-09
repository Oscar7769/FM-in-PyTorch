# PyTorch Factorization Machine

## Version Requirements

The following specific versions are required for compatibility:

| Package | Version | Source | Note |
| :--- | :--- | :--- | :--- |
| **Python** | `3.9.23` | `conda-forge` | Base interpreter |
| **torch** | `2.6.0+cu124` | `pip` | Deep Learning (CUDA 12.4) |
| **numpy** | `2.0.2` | `pip` | Scientific Computing |
| **matplotlib** | `3.9.4` | `pip` | Visualization |
| **jupyterlab** | `4.4.4` | `pip` | Interactive Environment |
| **dimod** | `0.12.20` | `pip` | Sampler API (Quantum Annealing) |
---

## Installation Steps

Follow these steps strictly in order to handle dependency resolution between Conda and Pip.

### 1. Create Conda Environment
Create a clean environment with the specific Python version.


# Create environment
```bash
conda create -n photonics python=3.9.23 -y
```
# Activate environment
```bash
conda activate mp
```
# Install PyTorch (CUDA 12.4)
```bash
pip install torch==2.6.0+cu124 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
```
# Install Remaining Packages
```bash
pip install numpy==2.0.2 matplotlib==3.9.4 jupyterlab==4.4.4 dimod==0.12.20
```

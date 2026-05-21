#!/bin/bash
# HPC Bootstrap Script for TS-GNN
# Run this on the HPC login node once: bash hpc/bootstrap.sh

set -e

echo ">>> Loading miniconda3..."
module load miniconda3

# Check if environment exists
if conda info --envs | grep -q "tsgnn"; then
    echo ">>> Environment 'tsgnn' already exists. Updating..."
else
    echo ">>> Creating conda environment 'tsgnn' with Python 3.10..."
    conda create -n tsgnn python=3.10 -y
fi

echo ">>> Activating environment..."
source activate tsgnn

echo ">>> Installing PyTorch with CUDA 12.1 support..."
# Adjust version if needed, but 2.2.0 is in requirements.txt
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo ">>> Installing project in editable mode..."
pip install -e .

echo ">>> Setup complete. You can now submit jobs using: sbatch hpc/submit.sbatch"

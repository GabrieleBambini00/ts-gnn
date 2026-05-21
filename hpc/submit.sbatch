#!/bin/bash
#SBATCH --job-name=ts-gnn
#SBATCH --output=logs/tsgnn_%j.out
#SBATCH --error=logs/tsgnn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Specific directives for your account
#SBATCH --account=3393519
#SBATCH --partition=stud
#SBATCH --qos=stud

# NOTE: If you have access to the H100s, you can change the above to:
# #SBATCH --partition=gpunew
# #SBATCH --qos=normal

# Load modules
module purge
module load cuda/12.1
module load miniconda3

# Activate environment
source activate tsgnn

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the pipeline
# --skip-download assumes you already have the data or will handle it separately
python scripts/run_pipeline.py --config configs/default.yaml --skip-download

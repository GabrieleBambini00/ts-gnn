#!/bin/bash
################################################################################
# SLURM Job Submission Script for Sheaf GNN HPC Training
#
# Usage:
#   sbatch scripts/submit_hpc_job.sh
#
# Or with custom parameters:
#   sbatch --job-name=sheaf-tp53 \
#          --output=logs/sheaf-tp53.log \
#          scripts/submit_hpc_job.sh --config configs/tp53_binary.yaml
#
# Environment:
#   - Loads python/cuda/pytorch via module system
#   - Clones/updates repo in $SCRATCH or $HOME
#   - Creates isolated venv
#   - Trains on A100 GPU
#   - Syncs results back to local machine
################################################################################

#SBATCH --job-name=sheaf-gnn-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=per_node
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@bocconi.it

set -eu

# ============================================================================
# CONFIGURATION
# ============================================================================

# Git repository URL (update with your GitHub repo)
REPO_URL="${REPO_URL:-https://github.com/your-username/ts-gnn.git}"
BRANCH="${BRANCH:-sheaf-verified-integration}"

# Work directory on HPC (prefer $SCRATCH for I/O, fall back to $HOME)
if [ -d "/scratch/${USER}" ]; then
  WORK_DIR="/scratch/${USER}/ts-gnn"
else
  WORK_DIR="${HOME}/ts-gnn"
fi

# Config file (relative to repo root)
CONFIG="${1:-configs/brca_real.yaml}"

# Output directory on HPC
OUTPUT_DIR="${WORK_DIR}/checkpoints/$(date +%Y%m%d_%H%M%S)"

# Local sync target (on your machine, set via environment)
LOCAL_SYNC_TARGET="${LOCAL_SYNC_TARGET:-.}"

# ============================================================================
# LOGGING & SETUP
# ============================================================================

mkdir -p "${WORK_DIR}/logs"
LOG_FILE="${WORK_DIR}/logs/hpc-training-$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================" | tee "${LOG_FILE}"
echo "SHEAF GNN HPC TRAINING JOB" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "Job ID: ${SLURM_JOB_ID}" | tee -a "${LOG_FILE}"
echo "Node: $(hostname)" | tee -a "${LOG_FILE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "${LOG_FILE}"
echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP
# ============================================================================

echo "[$(date)] Phase 1: Loading HPC modules..." | tee -a "${LOG_FILE}"

# Load required modules (adjust for your HPC environment)
module load python/3.11
module load cuda/12.1
module load pytorch/2.1.0

echo "[$(date)] Loaded modules:" | tee -a "${LOG_FILE}"
module list 2>&1 | tee -a "${LOG_FILE}"

# ============================================================================
# PHASE 2: REPOSITORY SETUP
# ============================================================================

echo "[$(date)] Phase 2: Repository setup..." | tee -a "${LOG_FILE}"

if [ ! -d "${WORK_DIR}" ]; then
  echo "[$(date)] Cloning repository..." | tee -a "${LOG_FILE}"
  mkdir -p "$(dirname ${WORK_DIR})"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${WORK_DIR}" | tee -a "${LOG_FILE}"
else
  echo "[$(date)] Updating repository..." | tee -a "${LOG_FILE}"
  cd "${WORK_DIR}"
  git fetch origin "${BRANCH}" | tee -a "${LOG_FILE}"
  git checkout "${BRANCH}" | tee -a "${LOG_FILE}"
  git pull origin "${BRANCH}" | tee -a "${LOG_FILE}"
fi

cd "${WORK_DIR}"
echo "[$(date)] Repository ready at: ${WORK_DIR}" | tee -a "${LOG_FILE}"

# ============================================================================
# PHASE 3: VIRTUAL ENVIRONMENT & DEPENDENCIES
# ============================================================================

echo "[$(date)] Phase 3: Virtual environment setup..." | tee -a "${LOG_FILE}"

VENV_DIR="${WORK_DIR}/venv-hpc"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[$(date)] Creating virtual environment..." | tee -a "${LOG_FILE}"
  python3.11 -m venv "${VENV_DIR}" | tee -a "${LOG_FILE}"
fi

echo "[$(date)] Activating virtual environment..." | tee -a "${LOG_FILE}"
source "${VENV_DIR}/bin/activate"

echo "[$(date)] Installing dependencies..." | tee -a "${LOG_FILE}"
pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"
pip install -e . 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date)] Installation complete. Python: $(which python)" | tee -a "${LOG_FILE}"

# ============================================================================
# PHASE 4: DATA VERIFICATION
# ============================================================================

echo "[$(date)] Phase 4: Data verification..." | tee -a "${LOG_FILE}"

CONFIG_PATH="${WORK_DIR}/${CONFIG}"
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[$(date)] ERROR: Config file not found: ${CONFIG_PATH}" | tee -a "${LOG_FILE}"
  exit 1
fi

# Extract data_dir from config
DATA_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}'))['data_dir'])" 2>/dev/null || echo "data/")
echo "[$(date)] Data directory: ${DATA_DIR}" | tee -a "${LOG_FILE}"

if [ ! -d "${WORK_DIR}/${DATA_DIR}" ]; then
  echo "[$(date)] WARNING: Data directory may not exist. Training will attempt to download." | tee -a "${LOG_FILE}"
fi

# ============================================================================
# PHASE 5: TRAINING
# ============================================================================

echo "[$(date)] Phase 5: Starting training..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Set environment for reproducibility
export OMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUTPUT_DIR}"

# Run training
python scripts/hpc_train_sheaf.py \
  --config "${CONFIG}" \
  --output "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"

TRAIN_EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "[$(date)] Training completed with exit code: ${TRAIN_EXIT_CODE}" | tee -a "${LOG_FILE}"

# ============================================================================
# PHASE 6: RESULTS SYNCHRONIZATION
# ============================================================================

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
  echo "[$(date)] Phase 6: Syncing results back to local machine..." | tee -a "${LOG_FILE}"

  # If on HPC with rsync capability
  if command -v rsync &> /dev/null; then
    # Note: This assumes SSH setup is configured (see scripts/sync_down.sh)
    # For now, just copy within HPC
    if [ -d "${OUTPUT_DIR}" ]; then
      echo "[$(date)] Results directory created: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
      echo "[$(date)] Contents:" | tee -a "${LOG_FILE}"
      ls -lh "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    fi
  fi
fi

# ============================================================================
# COMPLETION
# ============================================================================

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "JOB SUMMARY" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "Job ID: ${SLURM_JOB_ID}" | tee -a "${LOG_FILE}"
echo "Status: $([ ${TRAIN_EXIT_CODE} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

exit ${TRAIN_EXIT_CODE}

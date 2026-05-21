#!/bin/bash
################################################################################
# ONE-COMMAND HPC JOB SUBMISSION
#
# Usage: bash RUN_HPC_NOW.sh
#
# This script:
# 1. Clones/updates repo on HPC
# 2. Submits SLURM job for full BRCA training
# 3. Shows job ID and monitoring commands
################################################################################

set -e

HPC_USER="3393519"
HPC_HOST="slogin.hpc.unibocconi.it"
HPC_HOME="/home/${HPC_USER}"
REPO_URL="https://github.com/GabrieleBambini00/ts-gnn.git"
CONFIG="configs/brca_real.yaml"

echo "================================================================================"
echo "SHEAF GNN HPC TRAINING SUBMISSION"
echo "================================================================================"
echo "HPC User: $HPC_USER"
echo "HPC Host: $HPC_HOST"
echo "Config: $CONFIG"
echo "================================================================================"
echo ""

# Step 1: SSH and setup
echo "[1/4] Connecting to HPC and cloning repository..."
ssh -T "${HPC_USER}@${HPC_HOST}" << 'EOF'
  set -e
  cd ~
  if [ ! -d "ts-gnn" ]; then
    echo "  Cloning repository..."
    git clone https://github.com/GabrieleBambini00/ts-gnn.git
  else
    echo "  Updating repository..."
    cd ts-gnn
    git fetch origin sheaf-verified-integration
    git checkout sheaf-verified-integration
    git pull origin sheaf-verified-integration
    cd ~
  fi
  echo "  ✓ Repository ready"
EOF

# Step 2: Submit job
echo ""
echo "[2/4] Submitting SLURM job..."
JOB_ID=$(ssh -T "${HPC_USER}@${HPC_HOST}" << 'EOF'
  cd ~/ts-gnn
  sbatch scripts/submit_hpc_job.sh configs/brca_real.yaml | grep -oP 'Submitted batch job \K\d+'
EOF
)

echo "  ✓ Job submitted: JOB_ID=$JOB_ID"
echo ""

# Step 3: Show queue status
echo "[3/4] Job queue status..."
ssh -T "${HPC_USER}@${HPC_HOST}" << EOF
  echo "  Checking queue..."
  squeue -u ${HPC_USER} | head -5
  echo "  ✓ Job queued"
EOF

echo ""
echo "================================================================================"
echo "✅ JOB SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Job ID: $JOB_ID"
echo ""
echo "MONITORING COMMANDS (run these on your local machine):"
echo ""
echo "1. Check queue status:"
echo "   ssh ${HPC_USER}@${HPC_HOST} squeue -u ${HPC_USER}"
echo ""
echo "2. Watch logs in real-time:"
echo "   ssh ${HPC_USER}@${HPC_HOST} 'tail -f ~/ts-gnn/logs/slurm-${JOB_ID}.log'"
echo ""
echo "3. Download results when done:"
echo "   scp -r ${HPC_USER}@${HPC_HOST}:~/ts-gnn/checkpoints ./hpc_results/"
echo ""
echo "4. Cancel job (if needed):"
echo "   ssh ${HPC_USER}@${HPC_HOST} scancel ${JOB_ID}"
echo ""
echo "Expected runtime: 2-4 hours on A100 GPU"
echo "================================================================================"

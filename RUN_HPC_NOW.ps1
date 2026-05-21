# ONE-COMMAND HPC JOB SUBMISSION (PowerShell)
#
# Usage: powershell -ExecutionPolicy Bypass -File RUN_HPC_NOW.ps1
#
# This script:
# 1. Clones/updates repo on HPC
# 2. Submits SLURM job for full BRCA training
# 3. Shows job ID and monitoring commands

$ErrorActionPreference = "Stop"

$HPC_USER = "3393519"
$HPC_HOST = "slogin.hpc.unibocconi.it"
$HPC_HOME = "/home/$HPC_USER"
$REPO_URL = "https://github.com/GabrieleBambini00/ts-gnn.git"
$CONFIG = "configs/brca_real.yaml"

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "SHEAF GNN HPC TRAINING SUBMISSION" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "HPC User: $HPC_USER"
Write-Host "HPC Host: $HPC_HOST"
Write-Host "Config: $CONFIG"
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Step 1: SSH and setup
Write-Host "[1/4] Connecting to HPC and cloning repository..." -ForegroundColor Cyan
$setupScript = @'
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
'@

ssh -T "$HPC_USER@$HPC_HOST" $setupScript
Write-Host ""

# Step 2: Submit job
Write-Host "[2/4] Submitting SLURM job..." -ForegroundColor Cyan
$submitScript = @'
cd ~/ts-gnn
sbatch scripts/submit_hpc_job.sh configs/brca_real.yaml | grep -oP 'Submitted batch job \K\d+'
'@

$JOB_ID = (ssh -T "$HPC_USER@$HPC_HOST" $submitScript) | Select-Object -Last 1
Write-Host "  ✓ Job submitted: JOB_ID=$JOB_ID" -ForegroundColor Green
Write-Host ""

# Step 3: Show queue status
Write-Host "[3/4] Job queue status..." -ForegroundColor Cyan
$queueScript = "squeue -u $HPC_USER | head -5"
ssh -T "$HPC_USER@$HPC_HOST" $queueScript
Write-Host "  ✓ Job queued" -ForegroundColor Green

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✅ JOB SUBMITTED SUCCESSFULLY" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Job ID: $JOB_ID" -ForegroundColor Yellow
Write-Host ""
Write-Host "MONITORING COMMANDS (copy and run on your machine):" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Check queue status:"
Write-Host "   ssh $HPC_USER@$HPC_HOST squeue -u $HPC_USER" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Watch logs in real-time:"
Write-Host "   ssh $HPC_USER@$HPC_HOST 'tail -f ~/ts-gnn/logs/slurm-$JOB_ID.log'" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Download results when done:"
Write-Host "   scp -r ${HPC_USER}@${HPC_HOST}:~/ts-gnn/checkpoints ./hpc_results/" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Cancel job (if needed):"
Write-Host "   ssh $HPC_USER@$HPC_HOST scancel $JOB_ID" -ForegroundColor Cyan
Write-Host ""
Write-Host "Expected runtime: 2-4 hours on A100 GPU" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Green

# TS-GNN HPC -> Local Sync (PowerShell version)
# Usage: .\hpc\sync_down.ps1

$RemoteHost = "bocconi"
$RemoteDir = "~/ts-gnn"

Write-Host ">>> Downloading results from $RemoteHost..." -ForegroundColor Cyan

# Create local directories if they don't exist
if (!(Test-Path "checkpoints")) { New-Item -ItemType Directory "checkpoints" }
if (!(Test-Path "outputs")) { New-Item -ItemType Directory "outputs" }
if (!(Test-Path "logs")) { New-Item -ItemType Directory "logs" }

# SCP results - unfortunately SCP -r doesn't merge well, but for simplicity:
scp -r "${RemoteHost}:${RemoteDir}/checkpoints/*" ./checkpoints/
scp -r "${RemoteHost}:${RemoteDir}/outputs/*" ./outputs/
scp -r "${RemoteHost}:${RemoteDir}/logs/*" ./logs/

Write-Host ">>> Results downloaded." -ForegroundColor Green

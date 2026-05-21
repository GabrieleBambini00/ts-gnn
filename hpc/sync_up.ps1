# TS-GNN Local -> HPC Sync (PowerShell version)
# Usage: .\hpc\sync_up.ps1

$RemoteHost = "bocconi"
$RemoteDir = "~/ts-gnn"
$ZipFile = "tsgnn_sync.zip"

Write-Host ">>> Creating archive via git (respecting .gitignore)..." -ForegroundColor Cyan

# Remove old zip if exists
if (Test-Path $ZipFile) { Remove-Item $ZipFile }

# Use git archive to get a clean snapshot of the repo
git archive --format=zip --output $ZipFile HEAD

Write-Host ">>> Uploading to $RemoteHost..." -ForegroundColor Cyan
scp $ZipFile "${RemoteHost}:${RemoteDir}/"

Write-Host ">>> Unpacking on HPC..." -ForegroundColor Cyan
ssh $RemoteHost "cd $RemoteDir && unzip -o $ZipFile && rm $ZipFile"

Write-Host ">>> Sync complete." -ForegroundColor Green

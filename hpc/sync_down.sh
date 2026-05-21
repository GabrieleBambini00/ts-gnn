#!/bin/bash
# Sync HPC -> Local
# Run this from the project root in Git Bash: ./hpc/sync_down.sh

REMOTE_HOST="bocconi"
REMOTE_DIR="~/ts-gnn"

echo ">>> Pulling results from $REMOTE_HOST..."

# Create local directories if they don't exist
mkdir -p checkpoints outputs logs

rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/checkpoints/" ./checkpoints/ \
    "$REMOTE_HOST:$REMOTE_DIR/outputs/" ./outputs/ \
    "$REMOTE_HOST:$REMOTE_DIR/logs/" ./logs/

echo ">>> Results downloaded."

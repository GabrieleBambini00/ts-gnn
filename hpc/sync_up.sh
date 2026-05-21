#!/bin/bash
# Sync Local -> HPC
# Run this from the project root in Git Bash: ./hpc/sync_up.sh

REMOTE_HOST="bocconi"
REMOTE_DIR="~/ts-gnn"

echo ">>> Syncing code to $REMOTE_HOST..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '.vscode' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'data/' \
    --exclude 'checkpoints/' \
    --exclude 'outputs/' \
    --exclude 'logs/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

echo ">>> Sync complete."

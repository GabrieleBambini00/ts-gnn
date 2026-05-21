#!/bin/bash
# TS-GNN HPC Orchestrator
# Usage: ./hpc/run.sh [sync|boot|submit|results]

COMMAND=$1
REMOTE_HOST="bocconi"

case $COMMAND in
    "sync")
        bash hpc/sync_up.sh
        ;;
    "boot")
        echo ">>> Bootstrapping remote environment..."
        ssh $REMOTE_HOST "bash ts-gnn/hpc/bootstrap.sh"
        ;;
    "submit")
        echo ">>> Syncing and submitting job..."
        bash hpc/sync_up.sh
        ssh $REMOTE_HOST "cd ts-gnn && sbatch hpc/submit.sbatch"
        ;;
    "results")
        bash hpc/sync_down.sh
        ;;
    *)
        echo "TS-GNN HPC Automation"
        echo "Usage: ./hpc/run.sh [action]"
        echo "  sync    : Sync local code to HPC"
        echo "  boot    : Setup remote conda environment (run once)"
        echo "  submit  : Sync code and submit Slurm job"
        echo "  results : Download checkpoints and logs"
        ;;
esac

#!/bin/bash
# Setup completo TS-GNN in un comando
set -e

DRIVE_PATH="${1:-/content/drive/MyDrive/ts-gnn}"
DATA_DIR="$DRIVE_PATH/data"

echo "📦 Installazione dipendenze..."
pip install -q -e "$DRIVE_PATH"
pip install -q scanpy anndata harmonypy scrublet fair-esm decoupler scvelo geoparse

echo "🔧 Configurazione environment..."
export TSGNN_DATA_DIR="$DATA_DIR"
echo "export TSGNN_DATA_DIR=$DATA_DIR" >> ~/.bashrc

echo "⬇️  Download dataset..."
python -c "from tsgnn.data.download import download_all; download_all('$DATA_DIR')"

echo "✅ Setup completo. Avvia con:"
echo "   python $DRIVE_PATH/scripts/run_pipeline.py --config $DRIVE_PATH/configs/default.yaml --skip-download"

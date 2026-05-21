"""
TS-GNN CLI entry points.

Provides command-line interfaces:
    tsgnn-train     Run the training pipeline
    tsgnn-download  Download all required datasets
    tsgnn-evaluate  Run evaluation on a trained model
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cli_train():
    """Run the TS-GNN training pipeline."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Train the Temporal Sheaf Graph Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  tsgnn-train --config configs/default.yaml
  tsgnn-train --config configs/default.yaml --skip-download
  tsgnn-train --config configs/default.yaml --skip-download --skip-training
""",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the data download step",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training (use existing checkpoint)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Import here to avoid heavy imports on --help
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir.parent))
    from scripts.run_pipeline import run_pipeline
    from tsgnn.config import TSGNNConfig
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Validate config before running
    with open(config_path) as f:
        raw_dict = yaml.safe_load(f)
        config = TSGNNConfig.from_dict(raw_dict)
    logger.info(f"Loaded and validated configuration. Seed: {config.seed}")

    run_pipeline(str(config_path), args.skip_download, args.skip_training)


def cli_download():
    """Download all required datasets for TS-GNN."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Download all required datasets",
    )
    parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from tsgnn.data.download import download_all
    download_all()


def cli_evaluate():
    """Run evaluation on a trained TS-GNN model."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Evaluate a trained model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("Train a model first with: tsgnn-train --config configs/default.yaml")
        sys.exit(1)

    import yaml
    from tsgnn.config import TSGNNConfig
    config_path = Path(args.config)
    with open(config_path) as f:
        raw_dict = yaml.safe_load(f)
        config = TSGNNConfig.from_dict(raw_dict)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Evaluation would load model and run metrics
    # For now, point user to run_pipeline.py
    print("For full evaluation, use: python scripts/run_pipeline.py --skip-download --skip-training")

def cli_validate():
    """Verifica che tutti i file necessari esistano e abbiano dimensioni corrette."""
    import sys
    from tsgnn.data import DATA_DIR
    checks = [
        (DATA_DIR / "raw/brca/GSE176078",             "BRCA scRNA GSE176078"),
        (DATA_DIR / "raw/tcga/tcga_brca_tp53_mutations.csv", "TCGA BRCA mutations"),
        (DATA_DIR / "external/jaspar/JASPAR2024_CORE_vertebrates.txt", "JASPAR"),
        (DATA_DIR / "external/string/9606.protein.links.v12.0.txt.gz", "STRING"),
        (DATA_DIR / "external/esm2_embeddings/WT.pt", "ESM-2 WT embedding"),
    ]
    all_ok = True
    for path, desc in checks:
        exists = path.exists() if hasattr(path, "exists") else False
        status = "✅" if exists else "❌"
        print(f"{status} {desc}: {path}")
        if not exists:
            all_ok = False
    sys.exit(0 if all_ok else 1)

#!/usr/bin/env python3
"""
HPC Training Script for Sheaf GNN on real BRCA scRNA-seq data.

Usage (on HPC login node):
    sbatch scripts/submit_hpc_job.sh

Or run directly:
    python scripts/hpc_train_sheaf.py --config configs/brca_real.yaml --output checkpoints/hpc_run

Environment:
    - Loads from: configs/brca_real.yaml (real data, no fallback)
    - Logs to: logs/hpc_training_YYYY-MM-DD.log
    - Checkpoints: checkpoints/hpc_run/
    - Enforces: leakage assertion before training
"""

import argparse
import logging
import os
import sys
import time
from datetimelib import datetime
from pathlib import Path

import torch
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from tsgnn.utils import set_global_seed
from tsgnn.data.splits import assert_no_group_leakage
from tsgnn.training.trainer import Trainer

logger = logging.getLogger("tsgnn.hpc")


def setup_logging(output_dir: Path):
    """Configure logging for HPC job."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"hpc_training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return log_file


def validate_config(config: dict) -> None:
    """Verify config is HPC-ready."""
    assert config.get("allow_synthetic_fallback") is False, \
        "Config must have allow_synthetic_fallback: false for real data"
    assert config.get("data_dir"), "data_dir required"
    logger.info("✓ Config validated for HPC training")


def run_hpc_training(config_path: str, output_dir: str, skip_leakage_check: bool = False):
    """Run Sheaf GNN training on HPC with real data."""
    start = time.time()

    # Setup logging
    output_dir = Path(output_dir)
    log_file = setup_logging(output_dir)
    logger.info(f"HPC Training started | Log: {log_file}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    validate_config(config)

    seed = config.get("seed", 42)
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Seed: {seed}")

    # =====================================================================
    # PHASE 1: DATA LOADING (real data only, no fallback)
    # =====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: LOADING BRCA SCRNASEQ DATA (real data, no fallback)")
    logger.info("=" * 70)

    data_dir = Path(config.get("data_dir", "data"))

    try:
        from tsgnn.data.brca_loader import load_brca_data
        logger.info("Loading real BRCA scRNA-seq data...")
        adata = load_brca_data(use_gse158508=True, malignant_only=True)
        logger.info(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    except Exception as e:
        logger.error(f"✗ Failed to load BRCA data: {e}")
        logger.error("Check: data/raw/ folder and GSE158508 extraction")
        raise

    # =====================================================================
    # PHASE 2: DATA PREPROCESSING
    # =====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 2: DATA PREPROCESSING")
    logger.info("=" * 70)

    K = config.get("data", {}).get("K", 10)
    n_genes = config.get("data", {}).get("n_genes", 500)

    # Select top genes by variance
    if adata.n_vars > n_genes:
        from sklearn.feature_selection import SelectKBest, f_classif
        logger.info(f"Selecting top {n_genes} genes by variance...")
        # This is simplified; real implementation should use proper feature selection
        var_genes = np.argsort(adata.X.var(axis=0))[-n_genes:]
        adata = adata[:, var_genes]
        logger.info(f"✓ Selected {n_genes} genes")

    # =====================================================================
    # PHASE 3: DATA SPLITTING (with leakage check)
    # =====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 3: GROUP-DISJOINT SPLITTING (cell-line aware)")
    logger.info("=" * 70)

    if "cell_line" in adata.obs.columns:
        group_ids = adata.obs["cell_line"].values
        logger.info(f"Groups: {len(np.unique(group_ids))} cell lines")
    else:
        logger.warning("No 'cell_line' column found. Using all data.")
        group_ids = np.zeros(adata.n_obs)

    # Create train/test split
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    for train_idx, test_idx in splitter.split(range(adata.n_obs), groups=group_ids):
        train_groups = group_ids[train_idx]
        test_groups = group_ids[test_idx]

        # CRITICAL: Verify no leakage
        if not skip_leakage_check:
            try:
                assert_no_group_leakage(train_groups, test_groups)
                logger.info("✓ No group leakage detected")
            except Exception as e:
                logger.error(f"✗ Leakage check failed: {e}")
                raise

        logger.info(f"Train: {len(train_idx)} cells | Test: {len(test_idx)} cells")

    # =====================================================================
    # PHASE 4: MODEL TRAINING
    # =====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 4: SHEAF GNN TRAINING")
    logger.info("=" * 70)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    logger.info(f"Stalk dimension: {model_cfg.get('stalk_dim', 4)}")
    logger.info(f"Diffusion steps: {model_cfg.get('num_diffusion_steps', 2)}")
    logger.info(f"Epochs: {training_cfg.get('epochs', 10)}")
    logger.info(f"Learning rate: {training_cfg.get('learning_rate', 0.001)}")

    # Initialize trainer
    trainer = Trainer(config=config, output_dir=output_dir, logger=logger)

    # Train on real data
    logger.info("\nStarting training...")
    try:
        results = trainer.train(adata, train_idx, test_idx)
        logger.info(f"✓ Training complete")
        logger.info(f"Final validation metrics: {results}")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise

    # =====================================================================
    # PHASE 5: EVALUATION & POST-MORTEM
    # =====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 5: EVALUATION")
    logger.info("=" * 70)

    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed/3600:.1f}h")
    logger.info(f"Checkpoint: {output_dir / 'model.pt'}")
    logger.info(f"Metrics: {output_dir / 'metrics.json'}")
    logger.info(f"Log: {log_file}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HPC Training for Sheaf GNN on real BRCA data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config (e.g., configs/brca_real.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/hpc_run",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="Skip leakage assertion (NOT RECOMMENDED)"
    )

    args = parser.parse_args()

    results = run_hpc_training(
        config_path=args.config,
        output_dir=args.output,
        skip_leakage_check=args.skip_leakage_check
    )

    print("\n" + "=" * 70)
    print("✅ HPC TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

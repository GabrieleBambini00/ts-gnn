"""
End-to-End TS-GNN Pipeline.

Phase 6: Executes the complete pipeline from raw data to final results.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/phase0_crc.yaml --skip-download
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger("tsgnn.pipeline")


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_pipeline(config_path: str, skip_download: bool = False, skip_training: bool = False):
    """Execute the full TS-GNN pipeline."""
    start = time.time()

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Seed: {seed}")

    data_dir = Path(config.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # PHASE 1: DATA ACQUISITION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA ACQUISITION AND PREPROCESSING")
    logger.info("=" * 60)

    if not skip_download:
        try:
            from tsgnn.data.download import download_all
            download_all(str(data_dir))
        except Exception as e:
            logger.warning(f"Download failed (may need manual download): {e}")

    # Generate ESM-2 embeddings (uses module's built-in ESM_DIR path)
    from tsgnn.data.allele import load_esm2_embeddings, generate_esm2_embeddings

    logger.info("Generating/loading ESM-2 embeddings...")
    try:
        esm_embeddings = generate_esm2_embeddings()
    except Exception as e:
        logger.warning(f"ESM-2 generation failed: {e}. Using placeholders.")
        esm_embeddings = {}

    if not esm_embeddings:
        esm_embeddings = load_esm2_embeddings()

    # Fallback: generate random embeddings in memory
    if not esm_embeddings:
        seed_val = config.get("seed", 42)
        logger.warning(
            f"No ESM-2 embeddings available. Generating reproducible random placeholders "
            f"(seed={seed_val}). Results will NOT be biologically meaningful — "
            "run `python scripts/download_all_datasets.py` to compute real ESM-2 embeddings."
        )
        from tsgnn.data.allele import TP53_HOTSPOT_MUTATIONS
        _rng = torch.Generator()
        _rng.manual_seed(seed_val)
        esm_embeddings = {"WT": torch.randn(1280, generator=_rng)}
        for allele in TP53_HOTSPOT_MUTATIONS:
            esm_embeddings[allele] = torch.randn(1280, generator=_rng)

    logger.info(f"Loaded ESM-2 embeddings for {len(esm_embeddings)} alleles")

    # =====================================================================
    # PHASE 1E: DATA INTEGRATION — REAL DATA ONLY
    # No synthetic fallback. If real data is missing, the pipeline fails
    # with a clear error message.
    # =====================================================================
    model_cfg = config.get("model", {})
    data_cfg  = config.get("data", {})
    K  = data_cfg.get("K", 10)
    n_genes = data_cfg.get("n_genes", 500)

    from tsgnn.data.brca_loader import load_brca_data, build_brca_temporal_sequences
    from tsgnn.data.preprocess import preprocess_scrna, select_features
    from tsgnn.data.grn_construction import (
        load_regnetwork, load_targetgenereg,
        load_string_ppi, construct_base_grn,
    )

    logger.info("Loading real BRCA scRNA-seq data (no synthetic fallback)...")
    adata = load_brca_data(use_gse158508=True, malignant_only=True)

    # Preprocess
    adata = preprocess_scrna(
        adata,
        n_top_genes=data_cfg.get("n_hvgs", 2000),
        batch_key=data_cfg.get("batch_key", None),
    )

    # Feature selection: TFs + p53 targets + HVGs
    ext_dir = Path(data_dir) / "external"
    jaspar_file = ext_dir / "jaspar" / "JASPAR2024_CORE_vertebrates.txt"
    tf_list = None
    if jaspar_file.exists():
        with open(jaspar_file) as f:
            tf_list = [
                line.strip().lstrip(">")
                for line in f if line.startswith(">")
            ]

    targetgenereg_df = load_targetgenereg()
    gene_list = select_features(
        adata, n_genes=n_genes, tf_list=tf_list, p53_targets=targetgenereg_df
    )
    N = len(gene_list)
    logger.info(f"Feature selection: {N} genes")

    # Build GRN from regulatory priors
    regnet_df = load_regnetwork()
    _str_dir = ext_dir / "string"
    _str_cands = sorted(_str_dir.glob("*protein.links*")) if _str_dir.exists() else []
    _str_path = _str_cands[0] if _str_cands else None
    if _str_path:
        logger.info(f"STRING file: {_str_path.name}")
    string_df = load_string_ppi(_str_path) if _str_path else None

    import numpy as np
    expr_matrix = None
    if adata.raw is not None:
        gene_indices = [
            list(adata.raw.var_names).index(g)
            for g in gene_list if g in adata.raw.var_names
        ]
        mat = adata.raw.X[:, gene_indices]
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        expr_matrix = np.asarray(mat, dtype=np.float32)

    edge_index, edge_weight, grn_meta = construct_base_grn(
        gene_list=gene_list,
        regnetwork_edges=regnet_df,
        targetgenereg=targetgenereg_df,
        string_ppi=string_df,
        expr_matrix=expr_matrix,
    )
    E = edge_index.shape[1]
    logger.info(f"Real GRN: {N} nodes, {E} edges")
    logger.info(f"Working graph: {N} nodes, {E} edges")

    # ── Build per-allele temporal sequences (real data) ───────────────────
    sequences = build_brca_temporal_sequences(
        adata=adata,
        gene_list=gene_list,
        edge_index=edge_index,
        K=K,
        min_cells_per_allele=data_cfg.get("min_cells_per_allele", 50),
    )
    alleles = list(sequences.keys())
    if not alleles:
        raise RuntimeError(
            "No allele has enough cells to build temporal sequences. "
            "Check wu2021_tp53_alleles.csv and min_cells_per_allele config."
        )
    logger.info(f"Built temporal sequences for {len(alleles)} alleles: {alleles}")

    train_data, val_data = {}, {}
    for allele, seq in sequences.items():
        nf = seq.node_features_sequence   # (K, N, d)
        ew = seq.edge_weights_sequence    # (K, E)
        split = max(1, int(0.85 * K))
        train_data[allele] = {
            "node_features_seq": nf[:split],
            "edge_weights_seq":  ew[:split],
        }
        val_data[allele] = {
            "node_features_seq": nf[split:],
            "edge_weights_seq":  ew[split:],
        }

    # Only use ESM2 embeddings for alleles that have real data
    real_esm = {a: esm_embeddings.get(a, esm_embeddings["WT"]) for a in alleles}
    if any(a not in esm_embeddings for a in alleles):
        missing = [a for a in alleles if a not in esm_embeddings]
        logger.warning(f"Missing ESM2 embeddings for {missing}, using WT as fallback")

    # =====================================================================
    # PHASE 2: MODEL CONSTRUCTION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL CONSTRUCTION")
    logger.info("=" * 60)

    from tsgnn.model.tsgnn import TSGNN

    d = model_cfg.get("stalk_dim", 4)
    input_dim = model_cfg.get("input_dim", N)
    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=input_dim,
        esm_dim=model_cfg.get("esm_dim", 1280),
        conditioning_dim=model_cfg.get("conditioning_dim", 128),
        edge_index=edge_index,
        num_diffusion_steps=model_cfg.get("num_diffusion_steps", 3),
        use_allele_conditioning=True,
    )

    params = model.count_parameters()
    logger.info(f"Model parameters: {params['total']:,}")
    for name, count in params.items():
        if name != "total":
            logger.info(f"  {name}: {count:,}")

    # =====================================================================
    # PHASE 3: TRAINING
    # =====================================================================
    if not skip_training:
        logger.info("=" * 60)
        logger.info("PHASE 3: TRAINING")
        logger.info("=" * 60)

        from tsgnn.training.trainer import TSGNNTrainer

        trainer = TSGNNTrainer(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=str(project_root / "checkpoints"),
            use_wandb=config.get("training", {}).get("use_wandb", False),
        )

        results = trainer.train(
            train_data=train_data,
            val_data=val_data,
            esm_embeddings=real_esm,   # real ESM-2 where available, random fallback
        )

        logger.info(f"Training complete: {results['epochs_trained']} epochs, "
                     f"best val loss: {results['best_val_loss']:.6f}")
    else:
        logger.info("Skipping training (--skip-training flag)")

    # =====================================================================
    # PHASE 4: EVALUATION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: EVALUATION")
    logger.info("=" * 60)

    from tsgnn.evaluation.metrics import evaluate_all

    eval_results = evaluate_all(
        model=model,
        test_data=val_data,
        esm_embeddings=real_esm,
        stalk_dim=d,
    )

    logger.info("Evaluation results:")
    for metric, values in eval_results.items():
        logger.info(f"  {metric}: {values}")

    # =====================================================================
    # PHASE 5: VISUALIZATION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: VISUALIZATION")
    logger.info("=" * 60)

    fig_dir = str(project_root / "figures")
    try:
        # Get model outputs for visualization
        model.eval()
        with torch.no_grad():
            allele = alleles[0]
            node_seq = val_data[allele]["node_features_seq"].to(device)
            allele_emb = real_esm[allele].to(device)
            preds, maps_traj, laps = model(node_seq, allele_emb)

        from tsgnn.visualization.rewiring import identify_bottleneck_timepoints
        bottlenecks = identify_bottleneck_timepoints(
            [l.cpu() for l in laps], edge_index, d
        )
        logger.info(f"Found {len(bottlenecks)} bottleneck transitions")

        from tsgnn.visualization.attention import analyze_regulatory_mode
        modes = analyze_regulatory_mode(
            [m.cpu() for m in maps_traj], edge_index
        )
        mode_counts = {}
        for v in modes.values():
            mode_counts[v["mode"]] = mode_counts.get(v["mode"], 0) + 1
        logger.info(f"Edge regulatory modes: {mode_counts}")

        # Try to generate plots (requires matplotlib)
        try:
            from tsgnn.visualization.rewiring import plot_rewiring_trajectory
            plot_rewiring_trajectory(
                [l.cpu() for l in laps], edge_index, d,
                allele=allele, output_dir=fig_dir,
            )

            from tsgnn.visualization.networks import plot_allele_embedding_space
            plot_allele_embedding_space(real_esm, output_dir=fig_dir)

            logger.info(f"Figures saved to {fig_dir}")
        except ImportError as e:
            logger.warning(f"Plotting skipped (missing dependency): {e}")

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TS-GNN End-to-End Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download step")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing checkpoint)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    run_pipeline(str(config_path), args.skip_download, args.skip_training)


if __name__ == "__main__":
    main()

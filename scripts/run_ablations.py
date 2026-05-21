"""
Run All Ablation Studies.

Usage:
    python scripts/run_ablations.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger("tsgnn.ablations")


def main():
    parser = argparse.ArgumentParser(description="TS-GNN Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/ablation_table.tex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from tsgnn.evaluation.ablation import AblationRunner
    import numpy as np

    # Mock train/eval for development (replace with real functions)
    def mock_train(cfg):
        return {"train_loss": np.random.uniform(0.5, 1.5),
                "val_loss": np.random.uniform(0.3, 1.0)}

    def mock_eval(cfg):
        return {"pearson_r": np.random.uniform(0.1, 0.5),
                "fisher_p": np.random.uniform(0.001, 0.05)}

    runner = AblationRunner(config, mock_train, mock_eval, seeds=[42, 123, 456])
    results = runner.run_all()

    # Generate LaTeX table
    latex = runner.generate_latex_table(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    logger.info(f"Ablation results saved to {output_path}")

    # Print summary
    for study, study_results in results.items():
        logger.info(f"\n{study}:")
        for variant, metrics in study_results.items():
            logger.info(f"  {variant}: {metrics}")


if __name__ == "__main__":
    main()

"""
TP53 allele extraction and ESM-2 embedding generation.

Phase 1D: Generate allele-specific protein embeddings from ESM-2.
Also handles allele assignment from WES/metadata to scRNA-seq cells.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

from tsgnn.data import DATA_DIR
ESM_DIR = DATA_DIR / "external" / "esm2_embeddings"

# ── TP53 wild-type protein sequence (UniProt P04637, canonical, 393 residues) ──
TP53_SEQUENCE = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEV"
    "RVCACPGRDRRTEEENLHKTTGIDSFLHSGAKVTCTYSPALNKMFCQLAKTCPVQLWVDST"
    "PPPGTRVRAMAIYKQSQHMTEVVRRCPHERCTPFHCDGAFHCPQSQSTRYHQTSCRHREK"
    "EILEGQGSCLQLTWKTSRSQNEFIPKRLCPELHQNTIPVTLSPDNLEVDVQHKTKRLKEI"
    "NLQHSSQGDQEQKDQSSTSRHKKLMFKTEGPDSD"
)

# Hotspot mutations: position (1-indexed), WT residue, mutant residue
TP53_HOTSPOT_MUTATIONS = {
    "R175H": (175, "R", "H"),
    "R273H": (273, "R", "H"),
    "R248W": (248, "R", "W"),
    "R248Q": (248, "R", "Q"),
    "R282W": (282, "R", "W"),
    "G245S": (245, "G", "S"),
    "Y220C": (220, "Y", "C"),
}

# Structural classification
CONTACT_MUTANTS = {"R273H", "R248W", "R248Q"}
CONFORMATIONAL_MUTANTS = {"R175H", "G245S", "R282W", "Y220C"}


def create_mutant_sequence(mutation: str) -> str:
    """Create mutant TP53 protein sequence by substituting the specified residue."""
    pos, wt_res, mut_res = TP53_HOTSPOT_MUTATIONS[mutation]
    seq = list(TP53_SEQUENCE)
    idx = pos - 1  # Convert to 0-indexed

    if idx >= len(seq):
        raise ValueError(f"Position {pos} is beyond sequence length {len(seq)}")
    if seq[idx] != wt_res:
        logger.warning(
            f"Expected {wt_res} at position {pos}, found {seq[idx]}. "
            "Sequence may differ from canonical. Proceeding with substitution."
        )

    seq[idx] = mut_res
    return "".join(seq)


def generate_esm2_embeddings(
    alleles: Optional[list] = None,
    model_name: str = "esm2_t33_650M_UR50D",
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate ESM-2 embeddings for WT and all specified TP53 alleles.

    Args:
        alleles: List of allele names (e.g., ["R175H", "R273H"]).
                 If None, generates for all hotspot mutations.
        model_name: ESM-2 model to use.
        device: Torch device. Auto-detects CUDA if available.

    Returns:
        Dict mapping allele name to embedding tensor of shape (1280,).
        Also saves to data/external/esm2_embeddings/.
    """
    if alleles is None:
        alleles = list(TP53_HOTSPOT_MUTATIONS.keys())

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ESM_DIR.mkdir(parents=True, exist_ok=True)

    # Check if embeddings already exist
    all_exist = all((ESM_DIR / f"{a}.pt").exists() for a in alleles)
    if all_exist and (ESM_DIR / "WT.pt").exists():
        logger.info("All ESM-2 embeddings already cached. Loading...")
        embeddings = {"WT": torch.load(ESM_DIR / "WT.pt", weights_only=True)}
        for a in alleles:
            embeddings[a] = torch.load(ESM_DIR / f"{a}.pt", weights_only=True)
        return embeddings

    logger.info(f"Loading ESM-2 model: {model_name} on {device}")
    try:
        # Task 10.1: Cache ESM-2 on Drive between Colab sessions
        cache_dir = Path(os.environ.get("TSGNN_DATA_DIR", ".")).parent / "cache" / "esm2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
    except ImportError:
        logger.error("fair-esm not installed. Install with: pip install fair-esm")
        logger.info("Generating random placeholder embeddings for development...")
        return _generate_placeholder_embeddings(alleles)

    # Prepare sequences
    sequences = [("WT", TP53_SEQUENCE)]
    for allele in alleles:
        mut_seq = create_mutant_sequence(allele)
        sequences.append((allele, mut_seq))

    logger.info(f"Computing ESM-2 embeddings for {len(sequences)} sequences...")

    embeddings = {}
    # Process one at a time to manage memory
    for name, seq in sequences:
        batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])

        # Mean pool over residues (excluding BOS/EOS tokens)
        token_repr = results["representations"][33]
        # tokens: [BOS, residue_1, ..., residue_N, EOS]
        embedding = token_repr[0, 1:-1, :].mean(dim=0).cpu()  # (1280,)

        embeddings[name] = embedding
        torch.save(embedding, ESM_DIR / f"{name}.pt")
        logger.info(f"  {name}: embedding shape {embedding.shape}, norm {embedding.norm():.2f}")

    # Also save the difference embeddings (mutant - WT)
    wt_emb = embeddings["WT"]
    for allele in alleles:
        diff = embeddings[allele] - wt_emb
        torch.save(diff, ESM_DIR / f"{allele}_diff.pt")

    logger.info(f"All embeddings saved to {ESM_DIR}")
    return embeddings


def _generate_placeholder_embeddings(alleles: list) -> Dict[str, torch.Tensor]:
    """Generate random placeholder embeddings for development without ESM-2."""
    torch.manual_seed(42)
    embeddings = {"WT": torch.randn(1280)}
    torch.save(embeddings["WT"], ESM_DIR / "WT.pt")
    for a in alleles:
        embeddings[a] = torch.randn(1280)
        torch.save(embeddings[a], ESM_DIR / f"{a}.pt")
        diff = embeddings[a] - embeddings["WT"]
        torch.save(diff, ESM_DIR / f"{a}_diff.pt")
    logger.warning("Using PLACEHOLDER embeddings. Install fair-esm for real embeddings.")
    return embeddings


def load_esm2_embeddings(alleles: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """Load pre-computed ESM-2 embeddings from disk."""
    if alleles is None:
        alleles = list(TP53_HOTSPOT_MUTATIONS.keys())

    embeddings = {}
    for name in ["WT"] + alleles:
        path = ESM_DIR / f"{name}.pt"
        if path.exists():
            embeddings[name] = torch.load(path, weights_only=True)
        else:
            logger.warning(f"Embedding not found for {name}. Run generate_esm2_embeddings() first.")

    return embeddings


def assign_alleles_to_cells(
    adata,
    mutation_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> None:
    """
    Assign TP53 allele labels to each cell in an AnnData object.

    Args:
        adata: AnnData with obs containing patient_col.
        mutation_df: DataFrame with columns [patient_id, tp53_allele].
        patient_col: Column name in adata.obs for patient ID.

    Adds adata.obs["tp53_allele"] with values like "R175H", "R273H", "WT", etc.
    """
    # Merge allele info onto cell metadata
    allele_map = mutation_df.set_index("patient_id")["tp53_allele"].to_dict()
    adata.obs["tp53_allele"] = adata.obs[patient_col].map(allele_map).fillna("unknown")

    # Report allele distribution
    counts = adata.obs["tp53_allele"].value_counts()
    logger.info("TP53 allele distribution across cells:")
    for allele, count in counts.items():
        logger.info(f"  {allele}: {count} cells")

    # Flag alleles with insufficient cells
    min_cells = 100
    for allele in counts.index:
        if allele != "unknown" and counts[allele] < min_cells:
            logger.warning(
                f"  WARNING: {allele} has only {counts[allele]} cells "
                f"(minimum recommended: {min_cells})"
            )


def get_structural_class(allele: str) -> str:
    """Return structural class of a TP53 allele."""
    if allele in CONTACT_MUTANTS:
        return "contact"
    elif allele in CONFORMATIONAL_MUTANTS:
        return "conformational"
    elif allele == "WT":
        return "wildtype"
    else:
        return "unknown"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embeddings = generate_esm2_embeddings()
    print(f"\nGenerated embeddings for {len(embeddings)} alleles:")
    for name, emb in embeddings.items():
        print(f"  {name}: shape={emb.shape}, norm={emb.norm():.3f}")

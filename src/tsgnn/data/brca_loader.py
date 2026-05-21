"""
BRCA scRNA-seq data loader for TS-GNN.

Reads GSE176078 (Wu et al. 2021, TNBC) and GSE158508 (paired scRNA+scATAC),
assigns TP53 alleles from TCGA BRCA mutation calls, and returns a preprocessed
AnnData object ready for GRN construction and temporal binning.

Input files (downloaded by tsgnn.data.download):
  data/raw/brca/GSE176078/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz
  data/raw/brca/GSE158508/GSE158508_normalized_counts.txt.gz
  data/raw/tcga/tcga_brca_tp53_mutations.csv
"""
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from tsgnn.data import RAW_DIR, EXTERNAL_DIR

# TP53 hotspot alleles we model (must match allele.py)
MODELLED_ALLELES = {"R175H", "R273H", "R248W", "R248Q", "R282W", "G245S", "Y220C"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GSE176078 — Wu et al. 2021 TNBC
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gse176078(brca_dir: Path):
    """
    Extract GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz if not yet done.

    The outer archive contains per-patient inner .tar files, each with:
      count_matrix_sparse.mtx
      count_matrix_barcodes.tsv
      count_matrix_genes.tsv
      metadata.csv
    """
    # Try both the preferred filename and the GEO RAW fallback
    _candidates = [
        brca_dir / "GSE176078" / "GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz",
        brca_dir / "GSE176078" / "GSE176078_RAW.tar",
    ]
    tar_path = next((p for p in _candidates if p.exists()), None)
    extract_dir = brca_dir / "GSE176078" / "extracted"

    if tar_path is None:
        raise FileNotFoundError(
            f"GSE176078 archive not found in {brca_dir / 'GSE176078'}.\n"
            "Expected one of: GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz  or  GSE176078_RAW.tar\n"
            "Run tsgnn.data.download.download_brca_scrna() first."
        )

    if not (extract_dir.exists() and any(extract_dir.iterdir())):
        logger.info(f"Extracting {tar_path.name} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        # Auto-detect compression: .tar.gz → "r:gz", plain .tar → "r:"
        open_mode = "r:gz" if str(tar_path).endswith(".gz") else "r:"
        with tarfile.open(tar_path, open_mode) as tf:
            tf.extractall(extract_dir)
        logger.info(f"Extracted outer archive to {extract_dir}")
    else:
        logger.info("GSE176078 already extracted.")

    # Also extract any inner per-patient .tar files (present when downloaded fresh from GEO).
    # Locally these may already be extracted as directories with .tar in their name — skip those.
    inner_tars = [p for p in extract_dir.glob("*.tar") if p.is_file()]
    if inner_tars:
        logger.info(f"Extracting {len(inner_tars)} inner per-patient .tar archives...")
        for inner_tar in inner_tars:
            patient_dir = extract_dir / inner_tar.stem  # e.g. CID3941
            if patient_dir.exists() and any(patient_dir.iterdir()):
                continue  # already extracted
            patient_dir.mkdir(exist_ok=True)
            try:
                with tarfile.open(inner_tar, "r:") as tf:
                    tf.extractall(patient_dir)
                logger.debug(f"  Extracted {inner_tar.name} → {patient_dir.name}/")
            except Exception as e:
                logger.warning(f"  Could not extract {inner_tar.name}: {e}")

    return extract_dir


def _load_patient_dir(pdir: Path):
    """
    Load a single patient directory produced by Wu et al. 2021 (GSE176078).

    Handles two layouts:
    A) Files directly in pdir:
         count_matrix_sparse.mtx  +  count_matrix_barcodes.tsv  +  count_matrix_genes.tsv
    B) Files nested one level deeper (pdir / patient_id / ...same files...)
    C) Standard 10x naming: matrix.mtx(.gz)  +  barcodes.tsv(.gz)  +  features.tsv(.gz)

    Returns (patient_id, AnnData) or None if nothing readable was found.
    """
    import anndata as ad
    import scipy.io
    import scipy.sparse

    # Derive clean patient_id (strip ".tar" suffix that Windows extraction may keep)
    patient_id = pdir.name
    if patient_id.endswith(".tar"):
        patient_id = patient_id[:-4]

    def _try_wu_layout(d: Path):
        """Try to read Wu-et-al. count files from directory d."""
        mtx  = d / "count_matrix_sparse.mtx"
        bars = d / "count_matrix_barcodes.tsv"
        genes = d / "count_matrix_genes.tsv"
        if not (mtx.exists() and bars.exists() and genes.exists()):
            return None
        try:
            barcodes  = pd.read_csv(bars,  header=None, sep="\t")[0].tolist()
            gene_list = pd.read_csv(genes, header=None, sep="\t")[0].tolist()
            mat = scipy.io.mmread(str(mtx))
            # Standard 10x MTX: rows=genes, cols=cells → transpose to cells×genes
            if mat.shape[1] == len(barcodes):
                X = scipy.sparse.csr_matrix(mat.T)
            elif mat.shape[0] == len(barcodes):
                X = scipy.sparse.csr_matrix(mat)
            else:
                logger.warning(
                    f"  {patient_id}: MTX shape {mat.shape} doesn't match "
                    f"barcodes ({len(barcodes)}) or genes ({len(gene_list)}), skipping"
                )
                return None

            obs_df = pd.DataFrame(index=barcodes)
            var_df = pd.DataFrame(index=gene_list)
            adata = ad.AnnData(X, obs=obs_df, var=var_df)
            adata.obs["patient_id"] = patient_id

            # Load cell-type annotations from metadata.csv
            meta_csv = d / "metadata.csv"
            if meta_csv.exists():
                try:
                    meta = pd.read_csv(meta_csv, index_col=0)
                    for col in ("celltype_major", "celltype_minor", "cell_type", "CellType"):
                        if col in meta.columns:
                            # metadata index may use "PATIENT_barcode" → match directly or strip prefix
                            matched = meta.reindex(adata.obs_names)[col]
                            if matched.notna().any():
                                adata.obs["cell_type"] = matched.values
                            else:
                                meta2 = meta.copy()
                                meta2.index = meta2.index.str.replace(
                                    f"^{patient_id}_", "", regex=True
                                )
                                adata.obs["cell_type"] = meta2.reindex(adata.obs_names)[col].values
                            break
                except Exception as em:
                    logger.debug(f"  {patient_id}: metadata load failed — {em}")

            logger.info(f"  {patient_id}: {adata.n_obs} cells, {adata.n_vars} genes (Wu MTX)")
            return adata
        except Exception as e:
            logger.warning(f"  {patient_id}: Wu MTX load failed — {e}")
            return None

    def _try_10x_layout(d: Path):
        """Try standard 10x layout."""
        if not next(d.glob("matrix.mtx*"), None):
            return None
        try:
            import scanpy as sc
            adata = sc.read_10x_mtx(d, var_names="gene_symbols", cache=False)
            adata.obs["patient_id"] = patient_id
            logger.info(f"  {patient_id}: {adata.n_obs} cells (10x format)")
            return adata
        except Exception as e:
            logger.warning(f"  {patient_id}: 10x MTX load failed — {e}")
            return None

    # A) Files directly in pdir
    adata = _try_wu_layout(pdir) or _try_10x_layout(pdir)
    if adata is not None:
        return patient_id, adata

    # B) Files nested one level deeper (e.g. pdir/CID3941/)
    for subdir in sorted(pdir.iterdir()):
        if subdir.is_dir():
            adata = _try_wu_layout(subdir) or _try_10x_layout(subdir)
            if adata is not None:
                return patient_id, adata

    logger.warning(f"  {patient_id}: no readable count data found, skipping")
    return None


def load_gse176078(brca_dir: Optional[Path] = None) -> "anndata.AnnData":
    """
    Load GSE176078 (Wu et al. 2021 TNBC) into AnnData.

    Returns:
        AnnData with:
          obs["patient_id"]  — patient barcode prefix (e.g. "CID3586")
          obs["cell_type"]   — cell type from Wu et al. annotations (if available)
          var_names          — gene symbols
    """
    import anndata as ad
    import scipy.sparse as sp

    if brca_dir is None:
        brca_dir = RAW_DIR / "brca"

    extract_dir = _extract_gse176078(brca_dir)

    # Wu et al. 2021 archive structure: one directory per patient
    # Each contains count_matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
    patient_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    if not patient_dirs:
        # Flat layout: files directly in extract_dir
        patient_dirs = [extract_dir]

    adatas = []
    for pdir in sorted(patient_dirs):
        result = _load_patient_dir(pdir)
        if result is not None:
            _, adata_p = result
            adatas.append(adata_p)

    if not adatas:
        raise RuntimeError(
            "Could not load any patient data from GSE176078.\n"
            f"Extracted directory: {extract_dir}\n"
            "Check archive contents manually."
        )

    import anndata as _ad
    # patient_id is already set per-cell in each adata.obs["patient_id"]
    # Use anndata.concat (AnnData ≥0.8) with a fallback to the legacy .concatenate()
    try:
        combined = _ad.concat(adatas, join="outer", label=None)
        combined.obs_names_make_unique()
    except Exception:
        import scanpy as sc
        combined = adatas[0].concatenate(
            adatas[1:],
            batch_key="_batch",
            index_unique="-",
        )
        # Drop the internal _batch column; patient_id is already in obs
        if "_batch" in combined.obs.columns:
            combined.obs.drop(columns=["_batch"], inplace=True)
        # Clean up any duplicated patient_id-* columns from legacy concatenate
        pid_cols = [c for c in combined.obs.columns if c.startswith("patient_id-")]
        if pid_cols:
            combined.obs["patient_id"] = combined.obs[pid_cols].bfill(axis=1).iloc[:, 0]
            combined.obs.drop(columns=pid_cols, inplace=True)

    logger.info(
        f"GSE176078 loaded: {combined.n_obs} cells, "
        f"{combined.n_vars} genes, "
        f"{combined.obs['patient_id'].nunique()} patients"
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GSE158508 — paired scRNA + scATAC
# ─────────────────────────────────────────────────────────────────────────────

def load_gse158508(brca_dir: Optional[Path] = None) -> Optional["anndata.AnnData"]:
    """
    Load GSE158508 (paired scRNA-seq + scATAC-seq) into AnnData.

    Returns None if file not found (non-blocking).
    """
    import anndata as ad

    if brca_dir is None:
        brca_dir = RAW_DIR / "brca"

    acc_dir = brca_dir / "GSE158508"
    count_file = acc_dir / "GSE158508_normalized_counts.txt.gz"

    if not count_file.exists():
        logger.warning(
            f"GSE158508 counts not found at {count_file}. "
            "Skipping — only GSE176078 will be used."
        )
        return None

    logger.info("Loading GSE158508 normalized counts...")
    try:
        df = pd.read_csv(count_file, sep="\t", index_col=0)
        adata = ad.AnnData(df.T)

        # Infer patient_id from cell barcode prefix (e.g. "BTOS2_AAACCTGAGAATGTGT-1")
        adata.obs["patient_id"] = adata.obs_names.str.split("_").str[0]
        logger.info(
            f"GSE158508 loaded: {adata.n_obs} cells, "
            f"{adata.n_vars} genes, "
            f"{adata.obs['patient_id'].nunique()} patients"
        )
        return adata
    except Exception as e:
        logger.warning(f"GSE158508 load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TCGA BRCA TP53 allele assignment
# ─────────────────────────────────────────────────────────────────────────────

def load_tcga_brca_tp53(tcga_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load TCGA BRCA TP53 mutation calls downloaded by download_tcga_brca().

    Returns:
        DataFrame with columns: [patient_id, tp53_allele]
        tp53_allele ∈ MODELLED_ALLELES | "other_mutation" | "WT"
    """
    if tcga_dir is None:
        tcga_dir = RAW_DIR / "tcga"

    mut_file = tcga_dir / "tcga_brca_tp53_mutations.csv"
    if not mut_file.exists():
        logger.warning(
            f"TCGA BRCA mutations not found at {mut_file}. "
            "All cells will be labelled 'unknown'. "
            "Run tsgnn.data.download.download_tcga_brca() first."
        )
        return pd.DataFrame(columns=["patient_id", "tp53_allele"])

    df = pd.read_csv(mut_file)
    logger.info(f"Loaded {len(df)} TCGA BRCA TP53 mutation records")

    # Normalise column names (cBioPortal API returns camelCase)
    df.columns = [c.lower() for c in df.columns]

    # Extract patient ID: TCGA-XX-XXXX (first 12 chars of sampleId)
    sample_col = next(
        (c for c in df.columns if c.lower() in ("tumor_sample_barcode", "sampleid", "sample_id", "patient_id")),
        None
    ) or next((c for c in df.columns if "sample" in c.lower() or "patient" in c.lower()), None)
    if sample_col is None:
        logger.warning("Cannot find patient ID column in TCGA mutations file.")
        return pd.DataFrame(columns=["patient_id", "tp53_allele"])

    df["patient_id"] = df[sample_col].str[:12]

    # Map protein change to allele name.
    # Prefer HGVSp_Short (e.g. "p.R175H") over HGVSp (e.g. "p.Arg175His")
    # because _map_allele uses single-letter codes.
    prot_col = next(
        (c for c in df.columns if c.lower() in ("hgvsp_short", "hgvsps", "proteinchange")),
        None
    ) or next(
        (c for c in df.columns if "protein" in c.lower() or "hgvsp" in c.lower() or "amino" in c.lower()),
        None
    )

    def _map_allele(prot_change: str) -> str:
        if pd.isna(prot_change):
            return "WT"
        s = str(prot_change).replace("p.", "").strip()
        # Match hotspot patterns: R175H, R273H, R248W, R248Q, R282W, G245S, Y220C
        for allele in MODELLED_ALLELES:
            if s.startswith(allele) or s == allele:
                return allele
        return "other_mutation"

    if prot_col:
        df["tp53_allele"] = df[prot_col].apply(_map_allele)
    else:
        df["tp53_allele"] = "other_mutation"

    result = df[["patient_id", "tp53_allele"]].drop_duplicates("patient_id")

    # Summary
    counts = result["tp53_allele"].value_counts()
    logger.info("TCGA BRCA TP53 allele distribution:")
    for allele, n in counts.items():
        logger.info(f"  {allele}: {n} patients")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main entry point: load + merge + assign alleles
# ─────────────────────────────────────────────────────────────────────────────

def load_brca_data(
    use_gse158508: bool = True,
    malignant_only: bool = True,
) -> "anndata.AnnData":
    """
    Load, merge, and allele-label BRCA scRNA-seq data.

    Steps:
    1. Load GSE176078 (Wu et al. 2021)
    2. Optionally merge GSE158508
    3. Load TCGA BRCA TP53 allele calls
    4. Assign tp53_allele to each cell via patient_id
    5. (Optional) Filter to malignant cells using cell_type annotations

    Returns:
        AnnData with obs["tp53_allele"] and obs["patient_id"] populated.
        Raw counts in adata.X; obs["is_malignant"] if annotations available.

    Examples:
        >>> from tsgnn.data.brca_loader import load_brca_data
        >>> adata = load_brca_data(malignant_only=True)
        >>> print(adata.obs['tp53_allele'].value_counts())
    """
    logger.info("Loading BRCA scRNA-seq data...")

    # ── GSE176078 primary cohort ──
    adata = load_gse176078()

    # ── GSE158508 supplementary cohort ──
    if use_gse158508:
        adata2 = load_gse158508()
        if adata2 is not None:
            import scanpy as sc
            # Align genes (intersection)
            common_genes = list(set(adata.var_names) & set(adata2.var_names))
            if len(common_genes) > 100:
                adata  = adata[:, common_genes].copy()
                adata2 = adata2[:, common_genes].copy()
                adata  = adata.concatenate(
                    adata2,
                    batch_key="geo_accession",
                    batch_categories=["GSE176078", "GSE158508"],
                    index_unique="-",
                )
                logger.info(
                    f"Merged GSE176078 + GSE158508: "
                    f"{adata.n_obs} cells, {adata.n_vars} genes"
                )
            else:
                logger.warning(
                    f"Only {len(common_genes)} genes in common — "
                    "skipping GSE158508 merge."
                )

    # ── Assign TP53 alleles ──────────────────────────────────────────────────
    # Priority:
    #   1. wu2021_tp53_alleles.csv  — curated mapping for the Wu et al. 2021
    #      cohort (GSE176078) derived from TCGA-BRCA allele frequency data
    #      applied per clinical subtype (TNBC→hotspot, ER+→WT).
    #   2. tcga_brca_tp53_mutations.csv — for future TCGA-native cohorts.
    from tsgnn.data.allele import assign_alleles_to_cells

    wu2021_allele_file = RAW_DIR / "brca" / "wu2021_tp53_alleles.csv"
    allele_df = None

    if wu2021_allele_file.exists():
        try:
            allele_df = pd.read_csv(wu2021_allele_file)[["patient_id", "tp53_allele"]]
            allele_df = allele_df.drop_duplicates("patient_id")
            logger.info(
                f"Loaded Wu et al. 2021 curated allele assignments: "
                f"{len(allele_df)} patients"
            )
            allele_counts = allele_df["tp53_allele"].value_counts().to_dict()
            logger.info(f"  Allele distribution: {allele_counts}")
        except Exception as e:
            logger.warning(f"Failed to load wu2021_tp53_alleles.csv: {e}")
            allele_df = None

    if allele_df is None or len(allele_df) == 0:
        # Fallback to TCGA (only works when cohort patient IDs match TCGA barcodes)
        allele_df = load_tcga_brca_tp53()

    if allele_df is not None and len(allele_df) > 0:
        assign_alleles_to_cells(adata, allele_df, patient_col="patient_id")
        n_assigned = (adata.obs["tp53_allele"] != "unknown").sum()
        if n_assigned == 0:
            raise RuntimeError(
                "All cells remain 'unknown' after allele assignment. "
                "Patient IDs in adata do not match any entry in the allele file. "
                "Check wu2021_tp53_alleles.csv patient_id column vs adata.obs['patient_id']."
            )
        logger.info(f"Allele assignment: {n_assigned}/{adata.n_obs} cells assigned.")
    else:
        raise RuntimeError(
            "No allele assignment file found. "
            "Expected: data/raw/brca/wu2021_tp53_alleles.csv"
        )

    # ── Malignant cell heuristic ──
    if malignant_only and "cell_type" in adata.obs.columns:
        non_malignant = {
            "T_cell", "T cell", "B_cell", "B cell",
            "Macrophage", "Fibroblast", "Endothelial",
            "Mast_cell", "Mast cell", "NK_cell", "NK cell",
            "Plasma_cell", "Plasma cell",
            # BRCA-specific stromal types
            "CAF",                    # Cancer-associated fibroblast
            "Luminal_progenitor",     # Normal luminal
            "Basal_normal",           # Normal basal
            "Pericyte", "Adipocyte",
        }
        adata.obs["is_malignant"] = ~adata.obs["cell_type"].isin(non_malignant)
        n_mal = adata.obs["is_malignant"].sum()
        n_tot = adata.n_obs
        logger.info(f"Malignant cells: {n_mal}/{n_tot} ({100*n_mal/n_tot:.1f}%)")
        adata = adata[adata.obs["is_malignant"]].copy()
        logger.info(f"Retained {adata.n_obs} malignant cells for analysis")
    else:
        adata.obs["is_malignant"] = True

    # ── Annotate BRCA subtype (Task 2.4) ──
    annotate_brca_subtype(adata)

    logger.info(
        f"\nFinal BRCA AnnData: {adata.n_obs} cells, "
        f"{adata.n_vars} genes\n"
        f"TP53 allele distribution:\n"
        f"{adata.obs['tp53_allele'].value_counts().to_string()}"
    )
    return adata


# ─────────────────────────────────────────────────────────────────────────────────
# 4b. BRCA subtype annotation (Task 2.4)
# ─────────────────────────────────────────────────────────────────────────────────

def annotate_brca_subtype(adata) -> None:
    """Assign BRCA molecular subtype to each cell based on marker expression.

    Adds adata.obs["brca_subtype"]: "TNBC" | "HER2+" | "LumA" | "LumB" | "unknown"

    Subtype logic (PAM50 proxy):
      HER2+  : ERBB2 > 1.0
      LumB   : ESR1 > 0.5 or PGR > 0.5, and MKI67 > 1.0 (high proliferation)
      LumA   : ESR1 > 0.5 or PGR > 0.5, and MKI67 <= 1.0
      TNBC   : none of the above
    """
    import scipy.sparse as sp

    markers = {"ESR1": "ER", "PGR": "PR", "ERBB2": "HER2", "MKI67": "Ki67"}
    present = {m: m in adata.var_names for m in markers}

    def _mean_expr(gene: str) -> np.ndarray:
        """Return per-cell mean expression for a gene (or zeros if absent)."""
        if not present[gene]:
            return np.zeros(adata.n_obs, dtype=np.float32)
        x = adata[:, gene].X
        if sp.issparse(x):
            x = np.asarray(x.todense())
        return np.asarray(x).ravel().astype(np.float32)

    er   = _mean_expr("ESR1")
    pr   = _mean_expr("PGR")
    her2 = _mean_expr("ERBB2")
    ki67 = _mean_expr("MKI67")

    subtypes = []
    for i in range(adata.n_obs):
        if her2[i] > 1.0:
            subtypes.append("HER2+")
        elif er[i] > 0.5 or pr[i] > 0.5:
            subtypes.append("LumB" if ki67[i] > 1.0 else "LumA")
        else:
            subtypes.append("TNBC")

    adata.obs["brca_subtype"] = subtypes
    counts = adata.obs["brca_subtype"].value_counts()
    logger.info(f"BRCA subtypes annotated: {counts.to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build per-allele TemporalGraphSequence from real AnnData
# ─────────────────────────────────────────────────────────────────────────────

def split_patients_by_allele(
    adata,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patients into train/val/test preserving per-allele distribution.

    Performs stratified patient-level splitting so that the val/test sets
    contain patients unseen during training (leave-one-patient-out).

    Args:
        adata:       AnnData with obs[\"tp53_allele\"] and obs[\"patient_id\"].
        train_ratio: Fraction of patients per allele going to train.
        val_ratio:   Fraction going to val (remainder goes to test).
        seed:        Random seed for reproducibility.

    Returns:
        train_mask, val_mask, test_mask — boolean arrays of length n_obs.
    """
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(adata.n_obs, dtype=bool)
    val_mask   = np.zeros(adata.n_obs, dtype=bool)
    test_mask  = np.zeros(adata.n_obs, dtype=bool)

    allele_col  = adata.obs["tp53_allele"].values
    patient_col = (
        adata.obs["patient_id"].values
        if "patient_id" in adata.obs.columns
        else np.array(["all"] * adata.n_obs)
    )

    for allele in np.unique(allele_col):
        patients = np.unique(patient_col[allele_col == allele])
        rng.shuffle(patients)
        n = len(patients)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(0, int(n * val_ratio))
        n_val   = min(n_val, n - n_train)   # ensure something for test if possible

        train_pts = set(patients[:n_train])
        val_pts   = set(patients[n_train : n_train + n_val])
        test_pts  = set(patients[n_train + n_val :])

        cell_allele_mask = allele_col == allele
        train_mask |= cell_allele_mask & np.isin(patient_col, list(train_pts))
        val_mask   |= cell_allele_mask & np.isin(patient_col, list(val_pts))
        test_mask  |= cell_allele_mask & np.isin(patient_col, list(test_pts))

    n_tr = train_mask.sum()
    n_va = val_mask.sum()
    n_te = test_mask.sum()
    logger.info(
        f"Patient-level split (seed={seed}): "
        f"train={n_tr}, val={n_va}, test={n_te} cells"
    )
    return train_mask, val_mask, test_mask


def build_brca_temporal_sequences(
    adata,
    gene_list:             List[str],
    edge_index,
    K:                     int = 10,
    min_cells_per_allele:  int = 50,
    split:                 Optional[str] = None,
    train_ratio:           float = 0.70,
    val_ratio:             float = 0.15,
    seed:                  int   = 42,
) -> Dict:
    """Full pipeline from preprocessed AnnData to per-allele temporal graph sequences.

    Args:
        adata:               Preprocessed AnnData (after preprocess_scrna).
        gene_list:           N gene names defining node order.
        edge_index:          (2, E) base GRN topology tensor.
        K:                   Number of pseudotime bins.
        min_cells_per_allele: Minimum cells required to include an allele.
        split:               Optional \"train\" | \"val\" | \"test\". If provided,
                             applies patient-level splitting before building
                             temporal sequences. None = use all cells.
        train_ratio:         Passed to split_patients_by_allele().
        val_ratio:           Passed to split_patients_by_allele().
        seed:                Random seed for patient-level split.

    Returns:
        Dict mapping allele_name -> TemporalGraphSequence.

    Examples:
        >>> from tsgnn.data.brca_loader import build_brca_temporal_sequences
        >>> seqs_train = build_brca_temporal_sequences(adata, genes, ei, split="train")
        >>> seqs_val   = build_brca_temporal_sequences(adata, genes, ei, split="val")
        >>> seqs_test  = build_brca_temporal_sequences(adata, genes, ei, split="test")
    """
    from tsgnn.data.temporal import (
        compute_pseudotime,
        orient_with_velocity,
        bin_cells_by_pseudotime,
        construct_temporal_graphs,
    )

    # ── Patient-level split (Task 2.3) ──
    if split is not None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")
        train_mask, val_mask, test_mask = split_patients_by_allele(
            adata, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        split_mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
        adata_split = adata[split_mask].copy()
        logger.info(
            f"Using split='{split}': {adata_split.n_obs}/{adata.n_obs} cells"
        )
    else:
        adata_split = adata

    logger.info("Computing pseudotime for BRCA cells...")
    pseudotime = compute_pseudotime(adata_split, method="diffusion")
    pseudotime = orient_with_velocity(adata_split, pseudotime)
    pseudotime_bins = bin_cells_by_pseudotime(pseudotime, K=K)

    # Find alleles with enough cells
    allele_counts = adata_split.obs["tp53_allele"].value_counts()
    valid_alleles = [
        a for a, n in allele_counts.items()
        if n >= min_cells_per_allele and a not in ("unknown", "other_mutation")
    ]

    if not valid_alleles:
        logger.warning(
            f"No allele has >= {min_cells_per_allele} cells. "
            "Using all available alleles with relaxed threshold."
        )
        valid_alleles = [
            a for a in allele_counts.index
            if a not in ("unknown",) and allele_counts[a] >= 10
        ]

    logger.info(f"Building temporal sequences for alleles: {valid_alleles}")

    sequences = {}
    for allele in valid_alleles:
        allele_mask = (adata_split.obs["tp53_allele"] == allele).values
        n_cells = allele_mask.sum()
        logger.info(f"  {allele}: {n_cells} cells")

        seq = construct_temporal_graphs(
            adata=adata_split,
            gene_list=gene_list,
            edge_index=edge_index,
            pseudotime_bins=pseudotime_bins,
            K=K,
            allele=allele,
            allele_mask=allele_mask,
        )
        sequences[allele] = seq

    return sequences


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    adata = load_brca_data()
    print(f"\nLoaded: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Alleles: {adata.obs['tp53_allele'].value_counts().to_dict()}")

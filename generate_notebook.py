import json
from pathlib import Path

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# TS-GNN Data Acquisition for Google Colab\n",
        "\n",
        "This notebook assumes it is located inside your Google Drive project folder (e.g., `MyDrive/Sheaf Neural Networks/ts-gnn/`). It will download all the necessary datasets (Breast Cancer, Colorectal Cancer, Regulatory Priors) exactly to the relative `./data/` folder.\n",
        "\n",
        "**Instructions:**\n",
        "1. Mount your Google Drive.\n",
        "2. Make sure your working directory is correctly set to the `ts-gnn` root folder.\n",
        "3. Run all cells."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import os\n",
        "# ⚠️ IMPORTANT: Change this path to match exactly where your 'ts-gnn' folder is in your Drive\n",
        "ROOT_DIR = '/content/drive/MyDrive/Sheaf Neural Networks/ts-gnn'\n",
        "os.chdir(ROOT_DIR)\n",
        "print(f\"Current working directory: {os.getcwd()}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Install Dependencies"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install --quiet scanpy anndata GEOparse fair-esm torch torchvision torchaudio"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 1. Download Core Datasets and Priors\n",
        "First, we run the automated download script which prepares Breast Cancer matrices, Colorectal Cancer matrices, and external priors (RegNetwork, TargetGeneReg, JASPAR, etc.)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python src/tsgnn/data/download.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 2. Download Gambardella 2022 (Breast Cancer Cell Line Atlas)\n",
        "This dataset is crucial for maintaining a pristine ground truth for the model's training phase."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import requests\n",
        "from pathlib import Path\n",
        "\n",
        "def download_file(url, dest):\n",
        "    print(f\"Downloading {url} to {dest}...\")\n",
        "    resp = requests.get(url, stream=True, timeout=300)\n",
        "    resp.raise_for_status()\n",
        "    with open(dest, \"wb\") as f:\n",
        "        for chunk in resp.iter_content(chunk_size=8192):\n",
        "            f.write(chunk)\n",
        "    print(f\"Done: {dest.stat().st_size / 1e6:.1f} MB\")\n",
        "\n",
        "gamb_dir = Path(os.getcwd()) / \"data\" / \"raw\" / \"breast_gambardella\"\n",
        "os.makedirs(gamb_dir, exist_ok=True)\n",
        "\n",
        "# File 1: RAW.UMI.counts.BC.cell.lines.rds\n",
        "url1 = \"https://ndownloader.figshare.com/files/28893384\"\n",
        "dest1 = gamb_dir / \"RAW.UMI.counts.BC.cell.lines.rds\"\n",
        "if not dest1.exists(): download_file(url1, dest1)\n",
        "\n",
        "# File 2: GFICF.processed.counts.gficf\n",
        "url2 = \"https://ndownloader.figshare.com/files/33943715\"\n",
        "dest2 = gamb_dir / \"GFICF.processed.counts.gficf.tar\"\n",
        "if not dest2.exists(): download_file(url2, dest2)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 3. Generate ESM-2 Embeddings for TP53 Alleles\n",
        "Finally, we generate the mutation embeddings directly into the Google Drive caching folder."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python src/tsgnn/data/allele_embeddings.py"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

with open("c:\\Users\\02gab\\OneDrive - Università Commerciale Luigi Bocconi\\Desktop\\Sheaf Neural Networks\\ts-gnn\\Colab_Data_Downloader.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully.")

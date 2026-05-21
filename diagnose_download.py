import requests
import sys
import os
from pathlib import Path

def test_url(name, url):
    print(f"Testing {name}...")
    try:
        r = requests.get(url, timeout=10, stream=True)
        print(f"  Status: {r.status_code}")
        # Read just 1 byte
        content = next(r.iter_content(1))
        print(f"  Connectivity: OK (received data)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

urls = {
    "STRING": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "JASPAR": "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "GitHub_TCGA": "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_tcga_pan_can_atlas_2018/data_mutations.txt",
    "GEO_FTP": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
}

print("=== TS-GNN Connectivity Diagnostics ===")
for name, url in urls.items():
    test_url(name, url)

print("\n=== Environment Diagnostics ===")
print(f"CWD: {os.getcwd()}")
print(f"Python: {sys.version}")

# Check if we can write to the data directory
try:
    test_file = Path("data/test_write.tmp")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w") as f:
        f.write("test")
    print("Write Permission: OK")
    test_file.unlink()
except Exception as e:
    print(f"Write Permission: FAILED ({e})")

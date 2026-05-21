import requests
import urllib3
urllib3.disable_warnings()

urls = [
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "https://version-12-0.string-db.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "https://regnetworkweb.org/download/human.zip",
    "https://github.com/gersteinlab/RegNetwork/raw/main/network/human.zip",
    "https://ndownloader.figshare.com/files/51254970",
    "https://depmap-public.s3.amazonaws.com/depmap-releases/24Q4/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
]

for u in urls:
    try:
        r = requests.head(u, allow_redirects=True, timeout=5, verify=False)
        print(f'{r.status_code} {u}')
    except Exception as e:
        print(f'ERR {u}: {e}')

import requests
import re
import urllib3
urllib3.disable_warnings()

with open('src/tsgnn/data/download.py', 'r', encoding='utf-8') as f:
    text = f.read()

urls = re.findall(r'(https?://[^\s\"\',]+)', text)
urls = list(set(urls))

for u in urls:
    if 'ncbi.nlm.nih.gov' in u or 'jaspar.elixir.no' in u or 'encode' in u or 'cbioportal' in u or 'string' in u or 'regnetwork' in u or 'depmap' in u or 'figshare' in u:
        try:
            r = requests.head(u, allow_redirects=True, timeout=10, verify=False)
            print(f'{r.status_code} {u}')
        except Exception as e:
            print(f'ERR {u}: {e}')

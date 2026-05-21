import requests
import json
output = {}
try:
    r = requests.get('https://www.cbioportal.org/api/v2/molecular-profiles?studyId=brca_tcga_pan_can_atlas_2018')
    profiles = r.json()
    muts = [p['molecularProfileId'] for p in profiles if 'mutation' in p['molecularProfileId'].lower()]
    output['tcga_mut'] = muts
except Exception as e:
    output['tcga_mut'] = str(e)

for gse in ['GSE176078', 'GSE158508']:
    prefix = gse[:-3] + 'nnn'
    url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/'
    try:
        r = requests.get(url)
        import re
        files = re.findall(r'href="([^"]+)"', r.text)
        output[gse] = [f for f in files if f.endswith('gz') or f.endswith('tar') or f.endswith('h5ad') or f.endswith('h5')]
    except Exception as e:
        output[gse] = str(e)

with open('output_geo.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

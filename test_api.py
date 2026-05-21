import requests

try:
    print('Searching Figshare for DepMap...')
    r = requests.get('https://api.figshare.com/v2/articles?search=DepMap+24Q2+Public', timeout=10)
    articles = r.json()
    if articles:
        article_id = articles[0]['id']
        print('Article ID:', article_id)
        r2 = requests.get(f'https://api.figshare.com/v2/articles/{article_id}/files', timeout=10)
        files = r2.json()
        for f in files:
            if 'CRISPRGeneEffect.csv' in f['name'] or 'Model.csv' in f['name']:
                print(f['name'], '->', f['download_url'])
    else:
        print('No articles found')
except Exception as e:
    print('Error:', e)

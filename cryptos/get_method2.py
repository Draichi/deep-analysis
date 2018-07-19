from urllib.request import Request, urlopen
import pandas as pd
import json

url = 'https://api.coingecko.com/api/v3/coins/rupaya/market_chart?vs_currency=usd&days=1'
headers = {
    'User-Agent': 'Mozilla/5.0'
}
req = Request(url, headers=headers)
print('-- downloading')
res = urlopen(req).read()
# r = pd.read_json(response.json())
# r.to_picke('fred.pkl')
# print(res)
print('----------------------------------')
print(pd.read_json(res, orient='list'))
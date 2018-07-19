import requests
import pandas as pd

url = 'https://api.coingecko.com/api/v3/coins/rupaya/market_chart?vs_currency=usd&days=1'
headers = {
    'User-Agent': 'Mozilla/5.0'
}
response = requests.get(url, headers=headers)
print('-- downloading')
r = pd.DataFrame(response.json())
r.to_csv('rupaya.csv')
print(r)
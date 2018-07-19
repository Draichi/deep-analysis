import pandas as pd
import datetime
import plotly.offline as offline
import plotly.graph_objs as go
import requests
import pandas as pd

# https://plot.ly/python/time-series/

coins = ['fantasy-gold', 'rupaya']
keys = ['prices', 'market_caps', 'total_volumes']

def get_coin_data(coin):
    try:
        f = open('{}.csv'.format(coin), 'rb')
        df = pd.read_csv(f)
        print('--- loading {} from cache'.format(coin))
    except (OSError, IOError) as e:
        print('--- downloading {}'.format(coin))
        url = 'https://api.coingecko.com/api/v3/coins/{}/market_chart?vs_currency=usd&days=1'.format(coin)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.DataFrame(response.json())
        df.to_csv('{}.csv'.format(coin))
        print('--- caching {}'.format(coin))
    return df

def get_df(item, coin, key):
    current_item = item.replace('[', '').replace(']', '').split(',')
    date = current_item[0]
    val = current_item[1]
    dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
    coin_data[coin]['date'] = dt
    coin_data[coin][key] = val
    return pd.DataFrame(coin_data[coin], index=[0])


coin_data = {}

for coin in coins:
    data = get_coin_data(coin)
    coin_data[coin] = data
    
    for key in keys:
        for item in coin_data[coin][key]:
            df = get_df(item, coin, key)
            
print(df)
quit()
trace_price = go.Scatter(
    x=df['fantasy-gold']['date'],
    y=df['fantasy-gold']['prices'],
    name = "fantasy-gold prices",
)

trace_cap = go.Scatter(
    x=coin_data['fantasy-gold']['date'],
    y=coin_data['fantasy-gold']['market_caps'],
    name = "fantasy-gold caps",
)

trace_vol = go.Scatter(
    x=coin_data['rupaya']['date'],
    y=coin_data['rupaya']['prices'],
    name = "rupaya prices",
)

data = [trace_cap, trace_price]

offline.plot(data)











# for i, item in enumerate(df['prices']):
#     current_item = item.replace('[', '').replace(']', '').split(',')
#     date = current_item[0]
#     price = current_item[1]
#     dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
#     df.loc[i, 'date'] = dt
#     df.loc[i, 'price'] = price

# for i, item in enumerate(df['market_caps']):
#     current_item = item.replace('[', '').replace(']', '').split(',')
#     date = current_item[0]
#     cap = current_item[1]
#     dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
#     df.loc[i, 'date'] = dt
#     df.loc[i, 'cap'] = cap

# for i, item in enumerate(df['total_volumes']):
#     current_item = item.replace('[', '').replace(']', '').split(',')
#     date = current_item[0]
#     vol = current_item[1]
#     dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
#     df.loc[i, 'date'] = dt
#     df.loc[i, 'vol'] = vol

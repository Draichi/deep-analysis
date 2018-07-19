import pandas as pd
import datetime
import plotly.offline as offline
import plotly.graph_objs as go

df = pd.read_csv('ali.csv')
# df['index'] = df['prices']
for i, item in enumerate(df['prices']):
    current_item = item.replace('[', '').replace(']', '').split(',')
    date = current_item[0]
    price = current_item[1]
    dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
    df.loc[i, 'date'] = dt
    df.loc[i, 'price'] = price

df_prices = df[['date', 'price']]
# df_prices.reset_index(drop=True)
df_prices.set_index('date')

data = [go.Scatter(
        x=df_prices['date'],
        y=df_prices['price']
    )]  

offline.plot(data)
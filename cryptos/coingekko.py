import pandas as pd
import datetime

df = pd.read_csv('ali.csv')
# df['index'] = df['prices']
for i, item in enumerate(df.prices):
    current_item = item.replace('[', '').replace(']', '').split(',')
    date = current_item[0]
    price = current_item[1]
    df.loc[i, 'index'] = date
    df.loc[i, 'price'] = price

df_prices = df[['index', 'price']]
df_prices.set_index('index')
print(df_prices)
quit()

df.columns = ['index', 'market_cap', 'prices', 'total_volumes']
df.set_index('index')
item = df['total_volumes'][0]
current_item = item.replace('[', '').replace(']', '').split(',')
date = current_item[0]
price = current_item[1]

dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')

label_arr = list(df)

series_arr = list(map(lambda col: df[col], label_arr))
print(df)
quit()

print('date:', dt)
print('price:', price)
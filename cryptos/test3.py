import pandas as pd
import datetime

df = pd.read_csv('ali.csv')
df.set_index('Unnamed: 0')
item = df['total_volumes'][0]
current_item = item.replace('[', '').replace(']', '').split(',')
date = current_item[0]
price = current_item[1]

dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')

print('date:', dt)
print('price:', price)
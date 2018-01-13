import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# creating the csv file 
#start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)
#df = web.DataReader('TSLA', 'yahoo', start, end)
#df.to_csv('tsla.csv')

# column 0 (dates) will be the index column
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

# moving average will take the last 100 day prices and take the average of them today, min_periods will avoid NaN
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

print(df.tail())
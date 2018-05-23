import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# creating the csv file 
#start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)
#df = web.DataReader('TSLA', 'yahoo', start, end)
#df.to_csv('tsla.csv')

# column 0 (dates) will be the index column
df = pd.read_csv('datasets/tsla.csv', parse_dates=True, index_col=0)

# moving average will take the last 100 day prices and take the average of them today, min_periods will avoid NaN
# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# resample
# OpenHighLowClose           ex: 10Min, 6Min
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
# data based on 10 days
#print(df_ohlc.head())

# reset index and convert dates to mdates number so we can get the values
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# ploting w/ matplotlib and taking the values w/ pandas
# rows x columns    gridsize  staring
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
# this will display beatiful dates
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
# x = df_volume.index.map(mdates.date2num), y = df_volume.values, fill from 0 to y
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()
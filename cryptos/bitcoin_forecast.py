
# pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
import pandas as pd
import pandas_datareader.data as web
import quandl, math, pickle
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
from mpl_finance import candlestick_ohlc

style.use('ggplot')

#df = quandl.get('BTER/ZECBTC')

try:
    f = open('datasets/BCHARTS-BITSTAMPUSD.pkl', 'rb')
    df = pickle.load(f)
    print('-- data loaded from cache')
except (OSError, IOError) as e:
    print('-- downloading data from quandl')
    df = quandl.get('BITSTAMP/USD', returns="pandas")
    with open('datasets/BCHARTS-BITSTAMPUSD.pkl', 'wb') as ff:
        pickle.dump(df, ff)
    print('-- cached data')

# print(df.head())
# quit()

"""
percent of change => ( Close - Open ) / Open * 100
high low percent  => ( High - Close ) / Close * 100
"""

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Close', 'HL_PCT', 'PCT_Change', 'Volume (Currency)']]

# print(df.tail())

df.fillna(-99999, inplace = True)

forecast_col = 'Close'

forecast_out = int(math.ceil(0.001*len(df)))

# print(df.tail())


df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

# print(df['label'])
# quit()
X = X[:-forecast_out]

df.dropna(inplace=True)

print(df.head())
print(df.tail())

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
# with open('datasets/bitstamp.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('datasets/bitstamp.pkl', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print('---  Accuracy:', accuracy)
print('---  Forecast out:', forecast_out, 'days')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# df_ohlc = df['Last'].resample('10D').ohlc()
# df_volume = df['Volume'].resample('10D').sum()
# df_ohlc.reset_index(inplace=True)
# df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
# ax1.xaxis_date()

# candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')

# ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# df['Forecast'].plot()

# plt.show()

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
import pandas as pd
import pandas_datareader.data as web
import quandl, math, pickle
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
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

"""
percent of change => ( Close - Open ) / Open * 100
high low percent  => ( High - Close ) / Close * 100
"""
df['Last day close'] = df['Close'].shift(1)

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100

# df = df[['Close', 'HL_PCT', 'PCT_Change', 'Volume (Currency)']]

df.fillna(-99999, inplace = True)

# print(df.tail())
# quit()

# print('timestamp', lastdate)
# print('last_date', last_date)
# print('last_unix',last_unix)
# print('next_unix',next_unix)
# quit()
# for i in range(3):
#     next_date = dt.datetime.fromtimestamp(next_unix)
#     print('next date', next_date)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns))]
#     print(df.loc[next_date])


# print(x_df.tail())
# next_date = dt.datetime.fromtimestamp(next_unix)
# [np.nan for _ in range(len(df.columns)-1)] + [i]
# print(x_df.loc[next_date])
# print(x_df.tail())
# quit()

# -------
# adding a row
# df2 = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
# for i in range(5):
#     df2.loc[i] = [np.random.randint(-1,1) for n in range(3)]
# print(df2)

# -------

x_df = df.drop(['Close'], 1)
y_df = df['Close']

X = np.array(x_df)
X = preprocessing.scale(X)

df.dropna(inplace=True)

y = np.array(y_df)

# print('x',X[0:10])
# print('y',y[0:10])
# quit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=3)
clf.fit(X_train, y_train)
with open('datasets/bitstamp.pkl', 'wb') as f:
    pickle.dump(clf, f)

# pickle_in = open('datasets/bitstamp.pkl', 'rb')
# clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

print('accuracy', accuracy)


last_date = df.iloc[-1].name
# print(last_date)
# quit()
lastdate = dt.datetime.fromtimestamp(1528858800.0)

last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day*2


x_df_predict = x_df[-3:]
# print(x_df_predict)
# for i in range(len(x_df_predict)):
    
# quit()

for i in range(3):
    next_date = dt.datetime.fromtimestamp(next_unix)
    # print('next date', next_date)
    next_unix += one_day
    # x_df_predict.loc[next_date] = [np.nan for _ in range(len(x_df_predict.columns))]
    x_df_predict[next_date] = x_df_predict['Open'][i]
    print(x_df_predict[next_date])
    # print(x_df_predict.loc[next_date])
print(x_df_predict.tail())
quit()
predict = clf.predict([

])

quit()
forecast_set = clf.predict(next_days)

quit()

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
    print(df.loc[next_date])

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
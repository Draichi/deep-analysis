# pip install https://github.com/matplotlib/mpl_finance/archive/master.zip


import pandas as pd
import quandl, math, datetime
# numpy is better to use arrays
import numpy as np
# preprocessing is used to scaling the features data
# cross-validation will be used to train and test
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# dataframe
# RUPX/USD
df = pd.read_csv('datasets/rupaya_11-08-2017_16-05-2018.csv', decimal=",")

# df = df[['Data','Open', 'High', 'Low', 'Close', 'Volume',]]

# print(df.head())
# print(df.tail())
# quit()

# High x Low percent
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
# Daily percet change
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Featues are the attribuites who made the label, 
# and label is some sort of prediction into the future
# LABEL => 'Adj. Close'
# FEATURES => Everything else


forecast_col = 'Close'

# fill 'not avaible' data with -99999
df.fillna(-99999, inplace=True)

df['label'] = df[forecast_col].shift(1)

# if we wnat to print the tail
# we'll need to drop the 'na's before
df.dropna(inplace=True)
# print(df.head())
# print(df.tail())
# print(df[-2:-1].drop(['label'], 1))
# quit()

X = np.array(df.drop(['label'], 1))
# X = preprocessing.scale(X)

# X_lately = X[-forecast_out:]
# we made that shift so here we want to
# make sure that we only have X's where
# we have values for y
df.dropna(inplace=True)
# X = X[:-forecast_out]

y = np.array(df['label'])

# 20% of data we gonna use as test data
# this will shuffe the data maintaining the correlation
# between X's and y's
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

################

# classifier:
# n_jobs=10 will run 10 jobs at a time
# n_jobs=-1 will run as many jobs as possible
clf = LinearRegression(n_jobs=10)
# clf = svm.SVR()
# fit is synonimus of train
clf.fit(X_train, y_train)
# we pickle here to avoid to train everytime
with open('rupx.pickle', 'wb') as f:
    pickle.dump(clf, f)

##################

# load the alredy trained data
# pickle_in = open('rupx.pickle', 'rb')
# clf = pickle.load(pickle_in)

# score is synonimus of test
accuracy = clf.score(X_test, y_test)
row = df[-2:-1].drop(['label'], 1)
predict_this = np.array(row)
forecast_set = clf.predict(predict_this)
# print(df.tail())

print(predict_this, forecast_set)
print(X[-2:-1], y[-2:-1])
print('\x1b[1;33;40m   ---  Accuracy:', accuracy, '\x1b[0m')
quit()

# df['Forecast'] = np.nan

# last_date = df.iloc[-1].Data
# last_unix = last_date.timestamp()
# one_day = 86400
# next_unix = last_unix + one_day

# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# df['Close'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
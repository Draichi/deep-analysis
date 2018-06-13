import pandas as pd
import quandl, math, datetime, pickle, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style

warnings.filterwarnings("ignore", category=DeprecationWarning)
style.use('ggplot')

try:
    f = open('datasets/GOOGL-27-05.pkl', 'rb')
    df = pickle.load(f)
    print('-- data loaded from cache')
except (OSError, IOError) as e:
    print('-- downloading data from quandl')
    df = quandl.get('WIKI/GOOGL', returns="pandas")
    with open('datasets/GOOGL-27-05.pkl', 'wb') as ff:
        pickle.dump(df, ff)
    print('-- cached data')

# dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

# High x Low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Daily percet change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Featues are the attribuites who made the label, 
# and label is some sort of prediction into the future
# LABEL => 'Adj. Close'
# FEATURES => Everything else


forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

# let's say the length of df was a number that was
# return a decimal point like 0.2,
# math.ceil will round that up to 1 (float)
# this will be the number of days out,
# we gonna try to predict out 1% of dataframe 

forecast_out = int(math.ceil(0.001*len(df)))
# print(forecast_out)
# quit()

# df['label'] = df[forecast_col].shift(-forecast_out)

# if we wnat to print the tail
# we'll need to drop the 'na's before
# df.dropna(inplace=True)

# last_date = df.iloc[-1].name
# last_unix = last_date.timestamp()
# one_day = 86400
# next_unix = last_unix + one_day

# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]



# print(df.tail())
# print(last_date)
# quit()

X = np.array(df.drop(['Adj. Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

# print(last_unix)
# print('o')
# quit()

# X_lately
# [[ 1.63788031 -1.63575383 -0.53357709]
#  [ 2.51713951 -1.55184968 -0.65539548]
#  [-0.50488861  0.23733959 -0.55124957]
#  [ 4.10537475 -3.54051775 -0.59144015]]

# we made that shift so here we want to
# make sure that we only have X's where
# we have values for y
# X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['Adj. Close'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


################
 
# classifier:
# n_jobs=10 will run 10 jobs at a time
# n_jobs=-1 will run as many jobs as possible
clf = LinearRegression(n_jobs=10)
# clf = svm.SVR()
clf.fit(X_train, y_train)
with open('datasets/model_GOOGL-21-05.pkl', 'wb') as f:
    pickle.dump(clf, f)

##################

# load the alredy trained data
# pickle_in = open('datasets/model_GOOGL-21-05.pkl', 'rb')
# clf = pickle.load(pickle_in)

##################


accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print('\x1b[1;33;40m   ---  Accuracy:', accuracy, '\x1b[0m')
print('\x1b[1;33;40m   ---  Forecast out:', forecast_out, 'days \x1b[0m')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
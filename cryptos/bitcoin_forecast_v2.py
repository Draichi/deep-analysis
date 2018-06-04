import numpy as np
# import matplotlib.pyplot as plt
from pandas import read_csv
# from keras.models import Sequential, load_model
# from keras.layers import Dense
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math, pickle, quandl
import os

# hide tensorflow runtime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(36)

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

df = df.iloc[::-1]
df = df.drop(['Open','High','Low','Volume (Currency)', 'Volume (BTC)', 'Weighted Price'], axis=1)
dataset = df.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dateset = scaler.fit_transform(dataset)

def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(dataset[i + 1])
    return np.asarray(dataX), np.asarray(dataY)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, shuffle=False)

#                   samples,  time steps,   features
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# source activate tensorflow
# dectivate
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math, pickle, quandl
import os

# hide tensorflow runtime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(36)

def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(dataset[i + 1])
    return np.asarray(dataX), np.asarray(dataY)

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

X, y = create_dataset(dataset)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, shuffle=False)

#                   samples,  time steps,   features
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

model.save('datasets/savedBitcoinModel')

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

traiScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('-- train score: %.2f RMSE' % (traiScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('-- test score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict):len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

print("Price for last 5 days: ")
print(testPredict[-5:])
futurePredict = model.predict(np.asarray([[testPredict[-1]]]))
futurePredict = scaler.inverse_transform(futurePredict)
print("Bitcoin price for tomorrow: ", futurePredict)
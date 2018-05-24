import plotly.offline as py
import plotly.graph_objs as go
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from termcolor import *
import colorama
colorama.init()
# from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web

style.use('ggplot')
print_title = lambda x: cprint(x, 'magenta')
print_table = lambda x: cprint(x, 'green')

# creating the csv file 
#start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)
#df = web.DataReader('TSLA', 'yahoo', start, end)
#df.to_csv('tsla.csv')

# column 0 (dates) will be the index column
df = pd.read_csv('datasets/tsla.csv', parse_dates=True, index_col=0)
df_2 = web.DataReader('aapl', 'morningstar').reset_index()
# moving average will take the last 100 day prices and take the average of them today, min_periods will avoid NaN
# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# resample
# OpenHighLowClose           ex: 10Min, 6Min
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()


trace = go.Ohlc(x=df_2.Date,
                open=df_2.Open,
                low=df_2.Low,
                close=df_2.Close)


layout = go.Layout(
    xaxis = dict(
        rangeslider = dict(
            visible = True
        )
    )
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='OHLC without Rangeslider')
quit()

# data based on 10 days
print_title('\n--- Dataframe open high low close (raw)')
print_table(df_ohlc.head())

# reset index and convert dates to mdates number so we can get the values
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

print_title('\n--- Dataframe open high low close (normalized)')
print_table(df_ohlc.head())

print_title('\n--- Dataframe volume')
print_table(df_volume[:5])

# --------------------------------------------------------------->
# generate a scatter plot of the entire dataframe
def df_scatter(df,
                title,
                separate_y_axis=False,
                y_axis_label='',
                scale='linear',
                initial_hide=False
               ):
        label_arr = list(df)
        series_arr = list(map(lambda col: df[col], label_arr))
        
        layout = go.Layout(
            plot_bgcolor='#010008',
            paper_bgcolor='#010008',
            title=title,
            legend=dict(orientation="h"),
            xaxis=dict(type='date'),
            yaxis=dict(
                title=y_axis_label,
                showticklabels= not separate_y_axis,
                type=scale
            )
        )
        
        y_axis_config = dict(
            overlaying='y',
            showticklabels=False,
            type=scale
        )
        
        visibility = 'visible'
        if initial_hide:
            visibility = 'legendonly'
            
        # form trace for each series
        trace_arr = []
        for index, series in enumerate(series_arr):
            trace = go.Scatter(
                x=series.index,
                y=series,
                name=label_arr[index],
                visible=visibility
            )
            
            # add separate axis
            if separate_y_axis:
                trace['yaxis'] = 'y{}'.format(index + 1)
                layout['yaxis{}'.format(index + 1)] = y_axis_config
            
            trace_arr.append(trace)
            
        fig = go.Figure(data=trace_arr, layout=layout)
        py.iplot(fig)
        
df_scatter(df_ohlc, 'Dataframe open-high-low-close')



# # ploting w/ matplotlib and taking the values w/ pandas
# # rows x columns    gridsize  staring
# ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
# # this will display beatiful dates
# ax1.xaxis_date()

# candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
# # x = df_volume.index.map(mdates.date2num), y = df_volume.values, fill from 0 to y
# ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# plt.show()
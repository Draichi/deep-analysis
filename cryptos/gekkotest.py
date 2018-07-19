import os, pickle, quandl
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime
from urllib.request import Request, urlopen

def merge_dfs_on_column(dataframes, labels):
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index]
    return pd.DataFrame(series_dict)

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
                x=series.index*1000,
                y=series,
                name=label_arr[index],
                visible=visibility
            )
            
            # add separate axis
            if separate_y_axis:
                trace['yaxis'] = 'y{}'.format(index + 1)
                layout['yaxis{}'.format(index + 1)] = y_axis_config
            
            trace_arr.append(trace)
            
        offline.plot(
            {
                'data': trace_arr, 
                'layout': layout
            }, 
            image = 'png',
            filename = '{}.html'.format(title.replace(" ", "_")),
            image_filename = title
        )
        
def get_json_data(json_url):
    df = pd.read_json(json_url)
    return df

base_url = 'https://api.coingecko.com/api/v3/coins/{}/market_chart?vs_currency=usd&days=1'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
currency = 'usd'
days=1
end_date = datetime.now()
# daily, 86400 sec/day
period = 86400
user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'

def get_crypto_data(coin_name):
    # retrive crypto data from poloniex
    json_url = base_url.format(coin_name)
    print('-- downloading {}'.format(coin_name))
    req = Request(json_url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    # print(webpage)
    # quit()
    data_df = get_json_data(webpage)
    # data_df = data_df.set_index('prices')
    return data_df

altcoins = ['rupaya']

altcoin_data ={}
df_test ={} 
for altcoin in altcoins:
    # coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(altcoin)
    # web = crypto_price_df['prices']
    # for price in web:
    altcoin_data[altcoin] = crypto_price_df['prices']

    for date, price in altcoin_data[altcoin]:
        # print(idx, item)
        df_test[date] = price
    #     altcoin_data[altcoin]['Date'] = price[0]
# print(list(df_test.keys()))
# quit()
df_ = pd.DataFrame.from_dict({'date': list(df_test.keys()), 'price': list(df_test.values())})
# df_.index.name = 'date'
# df_.reset_index()
df_.to_csv('test2.csv')
print(df_.tail())
quit()
# df = pd.DataFrame(altcoin_data)
# df.to_csv("test.csv")
print(altcoin_data['absolute'])
quit()
# print(altcoin_data)
# quit()
# for altcoin in altcoin_data.keys():
#     # print('altcoin:', altcoin)
#     for item in altcoin_data[altcoin]:
        # print('item:',item[1])
        
        # altcoin_data[altcoin]['Price_BTC'] = item[1]
# quit()
# print(list(altcoin_data.values()))
# print(list(altcoin_data.keys()))
# quit()

combined_df = merge_dfs_on_column(
    list(altcoin_data.values()), 
    list(altcoin_data.keys())
)
combined_df.to_pickle('tst.pkl')

print(combined_df)
quit()

def correlation_heatmap(df, title, absolute_bounds=True):
    '''plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient')
    )
    layout = go.Layout(title=title)
    
    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0
    
    offline.plot(
        {
            'data': [heatmap], 
            'layout': layout
        }, 
        image = 'png',
        filename = '{}.html'.format(title.replace(" ", "_")),
        image_filename = title
    )

correlation_heatmap(combined_df.pct_change(), "Correlation 2018")
df_scatter(
    combined_df,
    'CRYPTO PRICES (USD)',
    separate_y_axis=False,
    y_axis_label='Coin Value (USD)',
    scale='linear'
)
combined_df.to_csv('datasets/gekkp.csv')
import os, pickle, quandl
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime
from urllib.request import Request, urlopen




    
# merge USD price of each altcoin into single dataframe
# combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'Price_USD')

# # add BTC price to the dataframe
# combined_df['BTC'] = btc_usd_datasets['MEAN_PRICE']



# --------------------------------------------------------------->
# data source:
# https://blog.quandl.com/api-for-bitcoin-data
# def get_quandl_data(quandl_id):
#     path = '{}.pkl'.format(quandl_id).replace('/','-')
#     cache_path = 'datasets/' + path
#     try:
#         f = open(cache_path, 'rb')
#         df = pickle.load(f)
#         print('-- loaded {} from cache'.format(quandl_id))
#     except (OSError, IOError) as e:
#         print('-- downloading {} from quandl'.format(quandl_id))
#         df = quandl.get(quandl_id, returns="pandas")
#         df.to_pickle(cache_path)
#         print('-- cached {} at {}'.format(quandl_id, cache_path))
#     return df

# # --------------------------------------------------------------->
# # gettin pricing data for 3more BTC exchanges
# exchanges = ['COINBASE', 'BITSTAMP', 'ITBIT', 'KRAKEN']
# exchange_data = {}

# for exchange in exchanges:
#     exchange_code = 'BCHARTS/{}USD'.format(exchange)
#     btc_exchange_df = get_quandl_data(exchange_code)
#     exchange_data[exchange] = btc_exchange_df

# --------------------------------------------------------------->
# merge a single column of each dataframe
def merge_dfs_on_column(dataframes, labels, col):
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
    return pd.DataFrame(series_dict)

# --------------------------------------------------------------->
# merge BTC price dataseries into a single dataframe
# btc_usd_datasets = merge_dfs_on_column(
#     list(exchange_data.values()),
#     list(exchange_data.keys()),
#     'Weighted Price'
# )
# btc_usd_datasets.replace(0, np.nan, inplace=True)
# btc_usd_datasets['MEAN_PRICE'] = btc_usd_datasets.mean(axis=1)
# uncomment the folowing to see the merged data
#btc_usd_datasets.tail()

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
    # download and cache json data, return as dataframe
    # pkl = '{}.pkl'.format(path)
    # cache_path = 'datasets/' + pkl
    # try:
    #     f = open(pkl, 'rb')
    #     df = pickle.load(f)
    #     print('-- loaded {} from cache'.format(path))
    # except (OSError, IOError) as e:
    #     print('-- downloading {}'.format(path))
    #     df = pd.read_json(json_url)
    #     df.to_pickle(pkl)
    #     print('-- cached {} at {}'.format(json_url, pkl))
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
    # data_df = data_df.set_index(0)
    return data_df

altcoins = ['rupaya']

altcoin_data ={}
for altcoin in altcoins:
    # coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(altcoin)
    altcoin_data[altcoin] = crypto_price_df
    
# for altcoin in altcoin_data.keys():
#     altcoin_data[altcoin]['Price_USD'] = altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['MEAN_PRICE']


# print(crypto_price_df.tail())
# print(altcoin_data)
# quit()
# merge USD price of each altcoin into single dataframe
combined_df = merge_dfs_on_column(
    list(altcoin_data.values()), 
    list(altcoin_data.keys()),
    0
)
combined_df.to_pickle('tst.pkl')

print(combined_df)
quit()

# add BTC price to the dataframe
# combined_df['BTC'] = btc_usd_datasets['MEAN_PRICE']

# scale can be 'linear' or 'log'
# df_scatter(combined_df,
#            'CRYPTO PRICES (USD)',
#            separate_y_axis=False,
#            y_axis_label='Coin Value (USD)',
#            scale='log')

# combined_df_2016 = combined_df[combined_df.index.year == 2016]
# combined_df_2016.pct_change().corr(method='pearson')

# combined_df_2017 = combined_df[combined_df.index.year == 2017]
# combined_df_2017.pct_change().corr(method='pearson')

# combined_df_2018 = combined_df[combined_df.index.year == 2018]
# combined_df_2018.pct_change().corr(method='pearson')

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

correlation_heatmap(combined_df_2018.pct_change(), "Correlation 2018")
df_scatter(
    combined_df,
    'CRYPTO PRICES (USD)',
    separate_y_axis=False,
    y_axis_label='Coin Value (USD)',
    scale='linear'
)
combined_df.to_csv('datasets/gekkp.csv')
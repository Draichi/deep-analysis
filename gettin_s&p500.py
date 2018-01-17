import bs4 as bs
import datetime as dt
import os
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pickle
import requests
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def save_sp500_list():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    # [1:] exclude the fist row (title rows)
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        # translate all '.' to '-' before append
        mapping = str.maketrans(".", "-")
        ticker = ticker.translate(mapping)
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

# run one time
#save_sp500_list()

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_list()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    # just getting the first 20st values
    for ticker in tickers[:20]:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# run one time
#get_data_from_yahoo()

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        # just getting the first 20st values
        tickers = pickle.load(f)[:20]
    main_df = pd.DataFrame()
    # we can count where we are on this list
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        # now the column will be 'MMM', 'XLL' with the 'close' ticker price
        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        # now we're gonna start to join this df together
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        # if counter divider by 10 is the remainder zero
        # we gonna see '10 , 20 , 30'
        if count % 10 == 0:
            print('Count:',count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

# run one time
#compile_data()

def vizualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AKAM'].plot()
    #plt.show()
    # this will make a correlation table of all our columns
    df_corr = df.corr()
    print(df_corr.head())
    data = df_corr.values
    fig = plt.figure()
    # 1 by 1 plot number 1
    ax = fig.add_subplot(1, 1, 1)
    # this cmap is a range from Red to Yellow to Green (negative , neutral, positive)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    # we're aranging ticks at every half marks
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    # we do this to remove the gap at the top of matplolib graphs
    ax.invert_yaxis()
    # put the ticks to the top
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    # set the color limits of this heatmap minimun x maximum
    heatmap.set_clim(-1,1)
    # clean the things a lil bit
    plt.tight_layout()
    plt.show()

vizualize_data()
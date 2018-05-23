import numpy as np 
import pandas as pd
from collections import Counter 
import pickle

#------------------------------------------------------------->

HOW_MANY_DAYS = 7
REQUIREMENT = 0.028

#------------------------------------------------------------->

df = pd.read_csv('datasets/sp500_joined_closes.csv', index_col=0)
tickers = df.columns.values
df.fillna(0, inplace=True)

#------------------------------------------------------------->
def process_data_for_labels(ticker):
    
    for i in range(1, HOW_MANY_DAYS+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0, inplace=True)
    return tickers, df

#------------------------------------------------------------->
def buy_sell_hold(*args):
    cols = [c for c in args]
    
    for col in cols:
        if col > REQUIREMENT:
            return 1
        if col < -REQUIREMENT:
            return -1
    return 0

#------------------------------------------------------------->
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                    df['{}_1d'.format(ticker)],
                                    df['{}_2d'.format(ticker)],
                                    df['{}_3d'.format(ticker)],
                                    df['{}_4d'.format(ticker)],
                                    df['{}_5d'.format(ticker)],
                                    df['{}_6d'.format(ticker)],
                                    df['{}_7d'.format(ticker)]))

    vals     = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = df['{}_target'.format(ticker)].values.tolist()

    return x, y, df

extract_featuresets('ALK')
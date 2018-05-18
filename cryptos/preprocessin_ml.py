import numpy as np
import pandas as pd
import pickle

# labels are our taget, it's basically a classification
def process_data_for_labels (ticker):
    # how many days in the future
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    # +1 because range starts on zero, we want to start on 1
    for i in range(1, hm_days+1):
        # e.g: XOM day 2 = 'XOM_2d' (in the future)
        # the value is the price in 2 days from now, minus today price, divide by todays price, times 100
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

process_data_for_labels('ABT')


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        # buy
        if col > requirement:
            return 1
        # sell
        if col < -requirement:
            return -1
    # hold
    return 0
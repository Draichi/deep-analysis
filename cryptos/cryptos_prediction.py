import argparse
import numpy as np 
import pandas as pd
from collections import Counter 
import pickle, warnings
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from termcolor import cprint
#------------------------------------------------------------->

warnings.filterwarnings("ignore", category=DeprecationWarning)
#------------------------------------------------------------->

parser = argparse.ArgumentParser(description='Deep analysis of cryptocurrencies')
parser.add_argument('-d', '--days', type=int, default=0, help='7')
parser.add_argument('-c', '--change', type=float, default=0, help='0.02')
parser.add_argument('-$', '--coin', type=str, default=0, help='BTC')
args = parser.parse_args()

#------------------------------------------------------------->
HOW_MANY_DAYS      = args.days
REQUIREMENT        = args.change
DATABASE           = 'datasets/altcoins_joined_closes_20181405.csv'
COIN               = args.coin
DATABASE_INDEX_COL = 0
#------------------------------------------------------------->
df = pd.read_csv(DATABASE, index_col=DATABASE_INDEX_COL)
tickers = df.columns.values
df.fillna(0, inplace=True)
#------------------------------------------------------------->
cprint(
    '\n\n              {} changing {}% in {} days:\n\n'.format(
        COIN, 
        REQUIREMENT*100, 
        HOW_MANY_DAYS
    ),
    'yellow'
)
#------------------------------------------------------------->
def buy_sell_hold(*args):
    cols = [c for c in args]
    
    for col in cols:
        if col > REQUIREMENT:
            return 'BUY'
        if col < -REQUIREMENT:
            return 'SELL'
    return 'HOLD'

#------------------------------------------------------------->
def process_data_for_labels(ticker):
    
    for i in range(1, HOW_MANY_DAYS+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0, inplace=True)
    return tickers, df

#------------------------------------------------------------->
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    for i in range(1, HOW_MANY_DAYS+1):
        df['{}_target'.format(COIN)] = list(map(
            buy_sell_hold,
            df['{}_{}d'.format(COIN,i)]
        ))

    vals     = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print_div()
    cprint('~~> Data spread: {}'.format(Counter(str_vals)), 'magenta')

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = df['{}_target'.format(ticker)].values.tolist()

    return x, y, df

#------------------------------------------------------------->
def train_the_clf(ticker):
    x, y, df = extract_featuresets(ticker)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=0.2
    )
    clf = VotingClassifier([
        ('lsvc', svm.LinearSVC()),
        ('knn', neighbors.KNeighborsClassifier()),
        ('rfor', RandomForestClassifier())
    ])
    clf.fit(x_train, y_train)

    confidence  = clf.score(x_test, y_test)
    predictions = clf.predict(x_test)

    cprint(
        '\n~~> Spread prediction: {}'.format(Counter(predictions)),
        'magenta'
    )
    cprint(
        '\n~~> Accuracy: {0:.3f} %'.format(confidence*100), 
        'magenta'
    )
    print_div()
    
    return confidence

#------------------------------------------------------------->
def print_div():
    cprint('~'*70, 'cyan')

train_the_clf(COIN)

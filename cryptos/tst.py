import os, pickle, quandl
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime
from urllib.request import Request, urlopen

df_ = pd.read_csv('test.csv', index_col=0)
print(df_.head())

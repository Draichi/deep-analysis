import pandas as pd
import os, time
from datetime import datetime

# change this path to the path to YOUR database
path = 'C:/Users/Lucas/Documents/intraQuarter'

def Key_Stats(gather='Total Debt/Equity (mrq)'):
    stats_path = path+'/_KeyStats'
    stock_list = [x[0] for x in os.walk(stats_path)]
    #print('\n', stock_list[:10])
    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        #print(each_file)
        # disconsider empty dirs
        if len(each_file) > 0:
          for file in each_file:
            date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
            unix_time = time.mktime(date_stamp.timetuple())
            #print(date_stamp, unix_time)

Key_Stats()
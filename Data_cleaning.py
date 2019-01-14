import pandas as pd
import re
import os

WORKING_FOLDER = os.curdir
#####################################################
# Put the DMSC.csv directly under WORKING_FOLDER
#####################################################

raw_data = pd.read_csv(WORKING_FOLDER + '/DMSC.csv')

for i in range(len(raw_data)):
    comment = raw_data.Comment[i]
    filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_，。、:！,.!《》<> ]')
    filtrate_str = filtrate.sub(r'', comment)
    print(filtrate_str)
    raw_data.loc[i, 'Comment'] = filtrate_str

raw_data.drop_duplicates()
raw_data.dropna()

raw_data.to_csv(WORKING_FOLDER + "/DMSC_cleaned.csv", encoding='utf-8')

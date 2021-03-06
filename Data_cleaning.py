import pandas as pd
import re
import os
from opencc import OpenCC

WORKING_FOLDER = os.curdir
#####################################################
# Put the DMSC.csv directly under WORKING_FOLDER
#####################################################

raw_data = pd.read_csv(WORKING_FOLDER + '/DMSC.csv')
cc = OpenCC('t2s')

for i in range(len(raw_data)):
    comment = raw_data.Comment[i]
    # filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_，。、:！,.!《》<>]')
    # filtrate_str = filtrate.sub(r' ', comment)
    raw_data.loc[i, 'Comment'] = cc.convert(comment)
    print(i)

raw_data.drop_duplicates()
raw_data.dropna()

raw_data.to_csv(WORKING_FOLDER + "/DMSC_cleaned.csv", encoding='utf-8')

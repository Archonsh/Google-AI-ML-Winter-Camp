import pandas as pd
import re

WORKING_FOLDER = "/Users/Archon/Downloads/"
#####################################################
# Put the DMSC.csv directly under WORKING_FOLDER
#####################################################

raw_data = pd.read_csv(WORKING_FOLDER + '/DMSC.csv')

raw_data.drop_duplicates()
raw_data.dropna()

for i in range(10):
    comment = raw_data.Comment[i]
    filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_，。、:！,.!《》<> ]')
    filtrate_str = filtrate.sub(r'', comment)
    print(filtrate_str)
    raw_data.loc[i, 'Comment'] = filtrate_str

raw_data.to_csv(WORKING_FOLDER + "/DMSC_cleaned.csv", encoding='utf-8')

import pandas as pd
import os

WORKING_FOLDER = os.curdir
raw_data = pd.read_csv(WORKING_FOLDER + '/DMSC.csv')

train = raw_data.sample(frac=0.8, random_state=12345678)
test = raw_data.drop(train.index)

train.to_csv(WORKING_FOLDER + "/DMSC_train.csv", encoding='utf-8')
test.to_csv(WORKING_FOLDER + "/DMSC_test.csv", encoding='utf-8')


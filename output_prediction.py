from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, \
    BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import sys
import csv

WORKING_FOLDER = os.curdir
EMBEDDING_NAME = 'sgns.weibo.word'
MAX_NB_WORDS = 500  # length of seq
EMBEDDING_DIM = 300  # length of embedding
MAX_NB_FEATURES = 500000  # max number of words in use

########################################################################
def test_acc():
    print("Working folder %s" % WORKING_FOLDER)

    train_df = pd.read_csv(WORKING_FOLDER + '/DMSC_train.csv')
    test_df = pd.read_csv(WORKING_FOLDER + '/DMSC_test.csv')
    print("Data read successfully")

    tk = Tokenizer()
    tk.fit_on_texts(train_df['Comment'])

    test_tokenized = tk.texts_to_sequences(test_df['Comment'])
    print("Tokenize complete")

    X_test = pad_sequences(test_tokenized, maxlen=MAX_NB_WORDS)

    M = ['CNN_GRU_20ep_MODEL.hdf5','CNN_ONLY_MODEL_SOFTMAX_L2_conv64.hdf5 ', 'LTSM_CNN_MODEL.hdf5', 'PARALLEL_LTSM_GRU_BEST_MODEL.hdf5']

    for MODEL_NAME in M:
        pred_model = load_model(MODEL_NAME)
        pred = pred_model.predict(X_test, batch_size=384, verbose=1)
        arg_rank = np.argsort(-pred, axis=1)

        with open('test.csv', 'w') as f:
            wt = csv.writer(f)
            for i in range(len(arg_rank)):
                wt.writerow(arg_rank[i])


if __name__ == '__main__':
    test_acc()

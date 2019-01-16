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

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax



WORKING_FOLDER = os.curdir
EMBEDDING_NAME = 'sgns.weibo.word'
MAX_NB_WORDS = 500  # length of seq
EMBEDDING_DIM = 300  # length of embedding
MAX_NB_FEATURES = 500000  # max number of words in use
MODEL_NAME = "LSTM_CNN_BEST_MODEL"

########################################################################

print("Working folder %s" % WORKING_FOLDER)

train_df = pd.read_csv(WORKING_FOLDER + '/DMSC_train.csv')
test_df = pd.read_csv(WORKING_FOLDER + '/DMSC_test.csv')
print("Data read successfully")

tk = Tokenizer()
tk.fit_on_texts(train_df['Comment'])

train_tokenized = tk.texts_to_sequences(train_df['Comment'])
test_tokenized = tk.texts_to_sequences(test_df['Comment'])
print("Tokenize complete")

X_test = pad_sequences(test_tokenized, maxlen=MAX_NB_WORDS)

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(test_df['Star'].values.reshape(-1,1))
print('One hot encoding complete')

pred_model = load_model(MODEL_NAME + '.hdf5')
pred = pred_model.predict(X_test, batch_size=384, verbose=1)
print(y_ohe[5], argmax(pred[5]), test_df['Star'][5])
wrong_indices = [i for i, v in enumerate(pred) if argmax(pred[i])+1
                 != test_df['Star'][i]]


acc = 1 - len(wrong_indices) / float(len(y_ohe))
print("Test accuracy: %f on %s" % (acc, MODEL_NAME))
print("Below is selections of wrongly predicted points:")

for i in wrong_indices[:10]:
    print(test_df['Comment'][i])

pred_model.fit(X_test,y_ohe,batch_size=384,verbose=1)




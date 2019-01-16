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



WORKING_FOLDER = os.curdir
EMBEDDING_NAME = 'sgns.weibo.word'
MAX_NB_WORDS = 500  # length of seq
EMBEDDING_DIM = 300  # length of embedding
MAX_NB_FEATURES = 500000  # max number of words in use
MODEL_NAME = "LTSM_CNN_BEST_MODEL"

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

LSTM_CNN_model = load_model(MODEL_NAME + '.hdf5')
loss, acc = LSTM_CNN_model.evaluate(X_test, y=y_ohe, batch_size=384, verbose=1)
print("Test loss: %f, accuracy: %f on %s" % (loss, acc, MODEL_NAME))





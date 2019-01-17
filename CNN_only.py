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

X_train = pad_sequences(train_tokenized, maxlen=MAX_NB_WORDS)  # pad all sentence to same length
X_test = pad_sequences(test_tokenized, maxlen=MAX_NB_WORDS)

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(train_df['Star'].values.reshape(-1, 1))
print('One hot encoding complete')

word_index = tk.word_index  # num of words appeared
nb_words = min(MAX_NB_FEATURES, len(word_index))
print("word index = %d" % len(word_index))
print("# of words = %d" % nb_words)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(WORKING_FOLDER + r'/' + EMBEDDING_NAME))

embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    # if i >= max_features:
    #     continue
    embedding_vector = embedding_index.get(word)  #
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print("Embedding matrix completed")


def build_model_CNN_only(lr=1e-3, lr_d=1e-10, spatial_dr=0.3, kernel_size2=2, kernel_size3=3, kernel_size5=5,
                         dense_units=128, dr=0.4, conv_size=32):
    file_path = "CNN_ONLY_MODEL_SOFTMAX_L2_maxpool.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    inp = Input(shape=(MAX_NB_WORDS,))
    x = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='same', kernel_initializer='he_uniform')(x1)
    # avg_pool3_conv2 = GlobalAveragePooling1D()(x2)
    # max_pool3_conv2 = GlobalMaxPooling1D()(x2)
    max_pool3_conv2 = MaxPooling1D()(x2)

    x3 = Conv1D(conv_size, kernel_size=kernel_size3, padding='same', kernel_initializer='he_uniform')(x1)
    # avg_pool3_conv3 = GlobalAveragePooling1D()(x3)
    # max_pool3_conv3 = GlobalMaxPooling1D()(x3)
    max_pool3_conv3 = MaxPooling1D()(x3)

    x5 = Conv1D(conv_size, kernel_size=kernel_size5, padding='same', kernel_initializer='he_uniform')(x1)
    # avg_pool3_conv5 = GlobalAveragePooling1D()(x5)
    # max_pool3_conv5 = GlobalMaxPooling1D()(x5)
    max_pool3_conv5 = MaxPooling1D()(x5)

    # x = concatenate([avg_pool3_conv2, max_pool3_conv2, avg_pool3_conv3, max_pool3_conv3,
    #                  avg_pool3_conv5, max_pool3_conv5])
    x = concatenate([max_pool3_conv2, max_pool3_conv3, max_pool3_conv5])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x))
    x = Dense(5, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["categorical_accuracy"])
    print(model.summary())
    history = model.fit(X_train, y_ohe, batch_size=128, epochs=20, validation_split=0.1,
                        verbose=1, callbacks=[check_point, early_stop])
    model = load_model(file_path)
    return model

CNN_model = build_model_CNN_only()


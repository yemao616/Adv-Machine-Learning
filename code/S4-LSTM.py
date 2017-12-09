import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import scikitplot.plotters as skplt

import nltk

# from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam



# Use the Keras tokenizer
num_words = 2000
text = np.load("/Users/Ye/ymao4/722/imdb_text_shuffle.npy")[:30000]
y = np.load("/Users/Ye/ymao4/722/imdb_class_shuffle.npy")[:30000]

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(text)
# Pad the data
X = tokenizer.texts_to_sequences(text)
X = pad_sequences(X, maxlen=2000)
print X
# Build out our simple LSTM
embed_dim = 128
lstm_out = 196

# Model saving callback
ckpt_callback = ModelCheckpoint('keras_model',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

model = Sequential()

model.add(Embedding(X.shape[0], embed_dim, input_length=X.shape[1]))
model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
print(model.summary())

Y = y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])

model = load_model('keras_model')
pred = model.predict(X_test)
RMSE = mean_squared_error(pred, Y_test) ** 0.5
print('RMSE: ', RMSE)
# skplt.plot_confusion_matrix(classes[np.argmax(Y_test, axis=1)], preds)

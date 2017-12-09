import os
import scipy.io
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


def evaluate_nlp(nlp, x_train, x_test, y_train, y_test):
    names = ["Nearest Neighbors", "RBF SVM", "Linear",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]
    Regressors = [
        KNeighborsRegressor(5),
        SVR(kernel="rbf", degree=3, gamma='auto'),
        LinearRegression(normalize=True),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_estimators=80),
        MLPRegressor(),
        AdaBoostRegressor()]
    rmse = []
    for name, clf in zip(names, Regressors):
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        RMSE = mean_squared_error(y_predict, y_test) ** 0.5
        # NRMSE = RMSE / (max(Val_Label[:,0]) - min(Val_Label[:,0]))
        rmse.append(RMSE)
        print('The RMSE of ' + name + ' is : ' + str(RMSE))
        # print('The NRMSE of ' + name + ' is : ' + str(NRMSE))
    return rmse



if __name__ == '__main__':
    y = np.load("/home/ymao4/722/imdb_class_shuffle.npy")[:30000]
    bag_of_words = np.load("/home/ymao4/722/bag_of_words.npy")
    tfidf = np.load("/home/ymao4/722/tfidf.npy")
    w2v = np.load("/home/ymao4/722/word2vec.npy")

    name_nlp = [" Bag of Words ", " Tf-idf ", " Word 2 Vector"]
    nlps = [bag_of_words, tfidf, w2v]

    noise = [200, 2000, 5000, 10000, 15000]
    for i in noise:
        print "------------------>>>>>>>>>>>>>>>>>"
        print i
        print "------------------>>>>>>>>>>>>>>>>>"
        s = np.arange(21000)
        np.random.shuffle(s)
        detail = []
        for nlp, each in zip(name_nlp, nlps):

            x = each
            train = range(int(y.shape[0]*0.7))
            test = range(int(y.shape[0]*0.7), y.shape[0], 1)
            x_test, y_test = x[test], y[test]
            x_train, y_train = x[train], y[train]

            s = s[:i]
            for each in s:
                x_train[each] = random.sample(xrange(1, 10), 1)[0]
       
            print "NLP : "+str(nlp)+ " in noise " + str(i)+ "--------"

            rmse = evaluate_nlp(nlp, x_train, x_test, y_train, y_test)
            detail.append(rmse)
        scipy.io.savemat('%s.mat' %i, mdict={'detail': detail})




  







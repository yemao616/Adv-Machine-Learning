import os
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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


class MySentences(object):
    """MySentences is a generator to produce a list of tokenized sentences 
    
    Takes a list of numpy arrays containing documents.
    
    Args:
        arrays: List of arrays, where each element in the array contains a document.
    """
    def __init__(self, *arrays):
        self.arrays = arrays
 
    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document.decode('utf-8')):
                    yield nltk.word_tokenize(sent)

def get_word2vec(sentences, location):
    """Returns trained word2vec
    
    Args:
        sentences: iterator for sentences
        
        location (str): Path to save/load word2vec
    """
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model


class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document.decode('utf-8')):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


def evaluate_features(X, y, clf=None):
    """General helper function for evaluating effectiveness of passed features in ML model

    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation

    Args:
        X (array-like): Features array. Shape (n_samples, n_features)

        y (array-like): Labels array. Shape (n_samples,)

        clf: Classifier to use. If None, default Log reg is use.
    """
    if clf is None:
        clf = LinearRegression()

    predicted = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8), n_jobs=-1, verbose=2)
    preds = np.round(predicted).astype(int)
    # for i in xrange(len(y)):
    #     print y[i], predicted[i], preds[i]
    #print('Log loss: {}'.format(log_loss(y, preds)))
    print('MSE: {}'.format(mean_squared_error(y, predicted)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
   
    # skplt.plot_confusion_matrix(y, preds)



if __name__ == '__main__':
    text = np.load("/home/ymao4/722/imdb_text_shuffle.npy")[:30000]
    y = np.load("/home/ymao4/722/imdb_class_shuffle.npy")[:30000]


    """ bag of words"""
    print "======== bag_of_words ========"
    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)
    bag_of_words = count_vectorizer.fit_transform(text)

    svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
    truncated_bag_of_words = svd.fit_transform(bag_of_words)
    # evaluate_features(truncated_bag_of_words, y)
    np.save('bag_of_words', truncated_bag_of_words)


    """ tfidf """
    print "======== tfidf ========"
    count_vectorizer = TfidfVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=None)    
    tfidf = count_vectorizer.fit_transform(text)
    #print len(count_vectorizer.get_feature_names())

    svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
    truncated_tfidf = svd.fit_transform(tfidf)
    # evaluate_features(truncated_tfidf, y)
    #evaluate_features(tfidf, y, SVC(kernel='linear'))
    np.save('tfidf', truncated_tfidf)


    """ word2Vector """
    print "======== w2vec ========"
    w2vec = get_word2vec(MySentences(text),'w2vmodel')
    mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)
    mean_embedded = mean_embedding_vectorizer.fit_transform(text)
    # evaluate_features(mean_embedded, y)
    np.save('word2vec', mean_embedded)



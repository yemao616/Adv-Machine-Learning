import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path = '/Users/Ye/ymao4/722/aclImdb/'
text = {}


def pretreat(data):
    rootpath = path+data
    for fn in os.listdir(rootpath):
        filename = os.path.join(rootpath, fn)
        with open(filename, 'rb') as f:
            text.append(f.read())
            string = data.replace('/', '_')
    np.save(string, text)


def tfIdf(d):
    string = d.replace('/', '_')+'.npy'
    data = np.load(string)
    corpus = list(data)
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    tf = X.toarray()
    terms_index = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    Y = transformer.fit_transform(tf)
    tfidf = Y.toarray()
    s = d.replace('/', '_')+'_tfidf'
    s = path+s
    #np.save('tf', tf)
    np.save(s, tfidf)
    #np.save('terms_index', terms_index)


if __name__ == '__main__':
    #l = ['train/neg', 'train/pos', 'test/neg', 'test/pos']
    #for i in l:
    #pretreat('test/neg')
    tfIdf('test/neg')
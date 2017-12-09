import os
import numpy as np


def pretreat(data):
    rootpath = path+data
    for fn in os.listdir(rootpath):
        key = fn.split('_')
        filename = os.path.join(rootpath, fn)
        with open(filename, 'rb') as f:
            text.append(f.read())
            y.append(key[-1])
    string = data.replace('/', '_')


if __name__ == '__main__':
    path = '/Users/Ye/ymao4/722/aclImdb/'
    text = []
    y = []
    l = ['train/neg', 'train/pos', 'test/neg', 'test/pos']
    for i in l:
        pretreat(i)
    np.save('/Users/Ye/ymao4/722/imdb_text', text)
    np.save('/Users/Ye/ymao4/722/imdb_class', y)
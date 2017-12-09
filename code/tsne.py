import numpy as np
from utils import feature_vis

def main():
    bw = np.load('bag_of_words.npy',encoding='bytes')
    tfidf = np.load('tfidf.npy',encoding='bytes')
    word2vec = np.load('word2vec.npy',encoding='bytes')
    labels = np.load('imdb_class_shuffle.npy',encoding='bytes')

    fea_vis = feature_vis(mode='classification')


    print (np.shape(tfidf))
    print (np.shape(word2vec))

    index = np.arange(1500)
    fea_vis.feature_vis_tsne(word2vec[index], labels[index], 'word2vec')
    fea_vis.feature_vis_tsne(bw[index], labels[index], 'bag of words')
    fea_vis.feature_vis_tsne(tfidf[index], labels[index], 'tfidf')
    

if __name__ == "__main__":
    main()


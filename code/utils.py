
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.decomposition import SparsePCA as PCA

class feature_vis(object):
    def __init__(self, mode='regression'):
        if mode not in ['classification', 'regression']:
            raise ValueError('only support classification or regression mode')
        self.mode = mode

    def feature_vis_tsne(self, features, targets, title='Figure'):
        if self.mode == 'regression':
            bins = np.array(np.linspace(np.min(targets), np.max(targets), num=10))
            targets = np.digitize(targets, bins)

        print(np.shape(features))
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
        tsne = TSNE(n_components=2)
        features = tsne.fit_transform(features)
        num_classes = max(np.unique(targets)) + 1
        print(num_classes)
        print(np.shape(targets))
        print(np.shape(features))

        cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
        colors = cmap[targets]
        x = features[:, 0]
        y = features[:, 1]
        plt.scatter(x, y, color=colors)
        handle = []
        for i in np.unique(targets):
            patch = mpatches.Patch(color=cmap[i], label=i)
            handle.append(patch)
        plt.legend(handles=handle)
        plt.title(title)
        plt.show()

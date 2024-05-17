import os

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import umap


def pca_plot(data, labels, title, save_path, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data = pca.transform(data)
    plt.figure(figsize=(12, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title(title)
    path = os.path.join(save_path, title + '_pca.png')
    plt.savefig(path, format='png')
    plt.close()


def t_sne_plot(data, labels, title, save_path, n_components=2, perplexity=30):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    data = tsne.fit_transform(data)
    plt.figure(figsize=(12, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title(title)
    path = os.path.join(save_path, title + '_tsne.png')
    plt.savefig(path, format='png')
    plt.close()


def umap_plot(data, labels, title, save_path, n_components=2, n_neighbors=5):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    data = reducer.fit_transform(data)
    plt.figure(figsize=(12, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title(title)
    path = os.path.join(save_path, title + '_umap.png')
    plt.savefig(path, format='png')
    plt.close()


if __name__ == "__main__":
    import numpy as np

    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 10, 100)
    pca_plot(data, labels, 'test', './')
    t_sne_plot(data, labels, 'test', './')
    umap_plot(data, labels, 'test', './')

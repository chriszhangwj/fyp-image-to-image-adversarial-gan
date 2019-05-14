import torch
import pytorch_ssim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsne_plot(data, label, dim=2):
    #data is the feature vector
    #label is the corresponding label
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)

    tsne = TSNE(n_components=dim, verbose = 1)
    tsne_data = tsne.fit_transform(pca_data)
    
    color_map = label
    
    plt.figure(figsize=(10,10))
    for cl in range(10):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_data[indices,0], tsne_data[indices,1], label=cl)
    plt.legend()
    plt.show()
    return None

feature_vec = np.genfromtxt("feature_200_mnist.csv", delimiter=',')
label_vec = np.genfromtxt("mnist_train_label_numpy.csv", delimiter=',')
print(np.shape(feature_vec))
print(np.shape(label_vec))
tsne_plot(feature_vec,label_vec,dim=2)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_min_dimensions(flattened):
    pca = PCA().fit(flattened)
    # plt.figure()
    # plt.title('PCA')
    # plt.xlabel('Dimensions')
    # plt.ylabel('Variance Retention')
    # plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
    # plt.show()
    min_dim = (np.where(pca.explained_variance_ratio_.cumsum() > 0.95))[0][0]
    print('Minimum dimensions required for 95% retention = ', min_dim)
    return min_dim

def doPCA(flattened):
    dim = get_min_dimensions(flattened)
    pca = PCA(dim)
    projected = pca.fit_transform(flattened)
    return projected

# MDA
def doMDA(X, Y, dim):
    mda = LinearDiscriminantAnalysis(n_components=dim)
    projected = mda.fit_transform(X,Y)
    return projected

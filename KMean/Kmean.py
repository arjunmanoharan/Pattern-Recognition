import numpy as np
from sklearn.cluster import KMeans
import math

def doKmeans(data,n_clusters1):
	kmeans = KMeans(n_clusters=n_clusters1, random_state=0).fit(data)
	dum=kmeans.cluster_centers_#get the cluster centers

	return dum.ravel()





def extractFeatures(vector, feat_indicator):

    x = np.array([float(vector[i]) for i in range(vector.shape[0]) if i % 2 == 0])
    y = np.array([float(vector[i]) for i in range(vector.shape[0]) if i % 2 != 0])

    # Number of points
    N = x.shape[0]

    def getDerivatives(x, y):
        dx, dy = [], []
        for i in range(x.shape[0])[2 : -2]:
            dx_, dy_ = 0, 0
            for j in range(2):
                dx_ += j * (x[i + j] - x[i - j])
                dy_ += j * (y[i + j] - y[i - j])
            dx_ /= (2 * (1 * 1 + 2 * 2))
            dy_ /= (2 * (1 * 1 + 2 * 2))
            dx.append(dx_)
            dy.append(dy_)
        dx = np.array(dx)
        dy = np.array(dy)
        return dx, dy

    # First derivatives
    dx, dy = getDerivatives(x, y)
    
    # Second derivatives
    d2x, d2y = getDerivatives(dx, dy)

    # Curvature
    kt = []
    epsilon = 1e-9
    for i in range(d2x.shape[0]):
        kt.append( (dx[i + 2] * d2y[i] - d2x[i] * dy[i + 2]) / np.power(dx[i + 2] * dx[i + 2] + dy[i + 2] * dy[i + 2] + epsilon, 3.0 / 2.0) )
    kt = np.array(kt)

    # Assemble features
    feat_map = {'x' : x[ 4 : -4], 'dx' : dx[2 : -2], 'd2x' : d2x, 'y' : y[4 : -4], 'dy' : dy[2 : -2], 'd2y' : d2y, 'kt' : kt}
    features = []
    for i in range(kt.shape[0]):
        feats_ = []
        for f in feat_indicator.split(','):
            feats_.append(feat_map[f][i])
        features.append(feats_)

    return np.array(features)



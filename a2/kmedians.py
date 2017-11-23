import numpy as np
import utils

class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        k = self.k
        y = np.ones(N)

        centers = np.zeros((k, D))
        for kk in range(k):
            i = np.random.randint(N)
            centers[kk] = X[i]

        while True:
            y_old = y

            # Compute L1 distance to each cluster center
            dist = np.zeros((N,k))
            for i in range(N):
                for kk in range(k):
                    dist[i][kk] = np.sum(np.abs(X[i] - centers[kk]))
            y = np.argmin(dist, axis=1)

            # Update medians
            for kk in range(k):
                if X[y==kk].size != 0:
                    centers[kk] = np.median(X[y==kk], axis=0)

            # Stop if no point changed cluster
            changes = np.sum(y != y_old)
            print('Running K-medians, changes in cluster assignment = {}'.format(changes))
            if changes == 0:
                break

        self.centers = centers

    def predict(self, X):
        centers = self.centers
        k = self.k

        N = X.shape[0]
        k = centers.shape[0]
        dist = np.zeros((N,k))
        for i in range(N):
            for kk in range(k):
                dist[i][kk] = np.sum(np.abs(X[i] - centers[kk]))
        return np.argmin(dist, axis=1)

    def error(self, X):
        centers = self.centers
        k = self.k

        y = self.predict(X)

        tot_dist_error = 0
        for kk in range(centers.shape[0]):
            X_k = X[y==kk]
            for i in range(len(X_k)):
                tot_dist_error += np.sum(np.abs(X_k[i] - centers[kk]))

        return tot_dist_error

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class KMeans:

    # where to use distributed computation?
    # What is the time and space complexity of the Kmeans algorithm?

    def __init__(self, k, max_iter = 100, tol = 1e-5, n_thread = 4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_thread = n_thread
        self.centers = None
        self.labels = None

    def fit(self, X):
        n_row, n_feat = X.shape

        self.centers = X[np.random.choice(n_row, self.k, replace = False)]

        for _ in range(self.max_iter):
            # Need a function to calculate distance
            distance = self._distance(X)

            self.labels = np.argmin(distance, axis = 1)

            new_centers = self._compute_new_centers(X)

            if np.linalg.norm(new_centers - self.centers) < self.tol:
                break
            self.centers = new_centers

        return

    def _distance(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centers, axis = 2)

    def _compute_new_centers(self, X):

        new_centers = np.zeros_like(self.centers)

        with ThreadPoolExecutor(max_workers=self.n_thread) as excecutor:
            futures = {
                excecutor.submit(self._compute_center, X, i): i
                for i in range(self.k)
            }

            for future in futures:
                i = futures[future]
                new_centers[i] = future.result()

        return new_centers

    def _compute_center(self, X, cluster_index):

        cluster_points = X[self.labels == cluster_index]
        if len(cluster_points) == 0:
            return self.centers[cluster_index]

        return cluster_points.mean(axis = 0)

    def predict(self, X):
        dis = self._distance(X)
        return np.argmin(dis, axis = 1)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    # import matplotlib.pyplot as plt

    # Create synthetic data
    data, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    # Initialize and fit the KMeans model
    kmeans = KMeans(k=3, n_thread=4)
    kmeans.fit(data)

    # Get the results
    labels = kmeans.labels
    centroids = kmeans.centers

    print(centroids)

    # Visualize results
    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    # plt.title("K-Means Clustering with Multi-threading")
    # plt.legend()
    # plt.show()
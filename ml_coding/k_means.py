import numpy as np
class KMeans:
    # Is K means a supervised or unsupervised learning algorithm?
        # K means is an unsupervised learning algorithm
    # What is the input and output of the Kmeans algorithm?
        # Input: X_train, K
        # Output: X_train_pred
    # Do we need a predict function for K-means?

    def __init__(self, num_clusters, num_iter = 100, tol = 1e-5):
        self.num_clusters = num_clusters
        self.num_iter = num_iter
        self.tol = tol

        # What is the dimension of self.center?
        self.center = None
        self.label = None

    def cal_dist(self, X):
        row, col = X.shape
        dist_matrix = np.zeros((row, self.num_clusters))
        for i in range(self.num_clusters):
            diff = X - self.center[i]
            # What norm is used here?
            dist_matrix[:,i] = np.linalg.norm(diff, axis = 1)
        return dist_matrix

    def fit(self, X):
        row, col = X.shape

        # Does initialization matters here?
        init_index = np.random.choice(row, self.num_clusters, replace = False)
        self.center = X[init_index]

        for _ in range(self.num_iter):
            dist = self.cal_dist(X)
            new_label = np.argmin(dist, axis = 1)       # What is argmin, argmax
            new_center = np.array([X[new_label == center].mean(axis = 0) for center in range(self.num_clusters)])

            if np.all(np.linalg.norm((new_center - self.center), axis = 1) ) < self.tol:
                break
            self.center = new_center
        self.labels = new_label

    def pred(self, X):
        diff = self.cal_dist(X)
        return np.argmin(diff, axis = 1)



if __name__ == "__main__":
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ])

    kmeans = KMeans(num_clusters=2)

    # print(kmeans.cal_dist(X))
    kmeans.fit(X)

    new_data = np.array([[0.0, 0.0], [10.0, 10.0]])
    predictions = kmeans.pred(new_data)
    print(predictions)
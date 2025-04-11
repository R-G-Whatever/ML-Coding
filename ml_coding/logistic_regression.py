import numpy as np
class LogisticRegression:
    def __init__(self, lr= 0.01, num_iter = 100, tol = 1e-6):
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.w = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def init_para(self, n_features):
        self.w = np.zeros(n_features)

    def predict_prod(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.bias)

    def predict(self, X, threshold = 0.5):
        proba = self.predict_prod(X)
        return (proba >= threshold).astype(int)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.init_para(n_features)

        for i in range(self.num_iter):
            model = np.dot(X, self.w) + self.bias

            predict = self.sigmoid(model)

            dw = np.dot(X.T, (predict - y)) / n_samples
            db = np.sum(predict - y) / n_samples

            self.w -= self.lr * dw
            self.bias -= self.lr * db

            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

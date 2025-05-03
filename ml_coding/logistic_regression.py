import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100, tol=1e-6):
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol

        self.w = None
        self.bias = None

    def sigmoid(self, z):
        # pay attention when divide something
        # can 1 + np.exp(-Z) close to 0?
        return 1 / (1 + np.exp(-z))

    def init_para(self, n_features):
        # Does initialition matters here? --- NO because the loss function logistic regression is convex
        # When convex it is guarantee to have a solution then it will have a closed form
        self.w = np.zeros(n_features)

    def predict_prob(self, X):
        # X shape (row, num_features)
        # self.w shape (num_features, 1)
        # X @ self.w
        return self.sigmoid(np.dot(X, self.w) + self.bias)

    def predict(self, X, threshold=0.5):
        proba = self.predict_prob(X)
        return (proba >= threshold).astype(int)

    def fit(self, X, y):
        # fit function gradient descent
        n_samples, n_features = X.shape

        self.init_para(n_features)

        for i in range(self.num_iter):
            predict = self.sigmoid(np.dot(X, self.w) + self.bias)

            dw = (
                np.dot(X.T, (predict - y)) / n_samples
            )  # based on derivative of the loss function

            db = (
                np.sum(predict - y) / n_samples
            )  # based on derivative of the loss function

            # Homework 2: derive function for logistic regression
            # Step1: sigmoid derivative
            # Step2: loss function - chain rule for derivative
            self.w -= self.lr * dw
            self.bias -= self.lr * db

            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break
                # [1,2,3,4] --- L2 norm = np.sqrt(1**2 + 2**2 + 3**2 + 4**2) ---> scaler
                #           --- L1 norm = (abs(1) + abs(2) + abs(3) + abs(4))

        # homework 3 --- Use closed form to calculate the LR results


# General traditional ML class structure
# class LogisticRegression:
#     def __init__(self, lr = 0.01, num_iter = 500, tol = 1e-5):
#         self.lr = lr
#         self.num_iter = num_iter
#         self.tol = tol
#
#     def fit(self, X, y):
#         return
#
#     def predict(self, X):
#         return y


if __name__ == "__main__":
    lr = LogisticRegression()
    print(lr.bias)

    # K4 - pay extra attention over float problem when divide by close to zero
    A = 99.9e-300
    A = [1e-300, 5e100]
    print(1 / A)
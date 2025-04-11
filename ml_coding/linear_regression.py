import numpy as np
class LinearRegressionClosed:
    def __init__(self):
        self.w = None
        self.bias = None

    def fit(self, X, y):

        X_b = np.hstack([np.ones((X.shape[0], 1) ), X])
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.bias = theta_best[0]
        self.w = theta_best[1:]

    def predict(self, X):
        if not self.w or not self.bias:
            raise ValueError("The model is not yet fitted")
        X_b = np.hstack([np.zeros((X.shape[0], 1)), X])
        return X_b.dot(np.hstack([self.bias, self.w]))

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u /v


class LinearRegression:
    def __init__(self, num_iter = 100, tol = 1e-5, lr = 0.1):
        self.num_iter = num_iter
        self.tol = tol
        self.lr = lr

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        m, n = X.shape
        for _ in range(self.num_iter):
            y_pred = X @ self.w + self.b
            dw = (1/ m) * X.T @ (y_pred - y)
            db = (1/ m) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

    def predict(self, X):
        return X @ self.w + self.b  # Make predictions

if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.2, 2.4, 3.1, 4.1, 5.1])

    model = LinearRegression()
    model.fit(X,y)
    predict = model.predict(np.array([[6], [7]]))
    print(predict)
    # print(model.score(X,y))

import numpy as np



class LogisticRegression:
    def __init__(self, num_iters=1000, tol=1e-5, lr=0.1):
        self.num_iters = num_iters
        self.tol = tol
        self.lr = lr

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0


        for _ in range(self.num_iters):
            Z = X @ self.w + self.b
            y_pred = self.sigmoid(Z)

            dw = X.T @ (y_pred - y) / m
            db = np.mean(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if np.abs(np.mean(y_pred - y)) <= self.tol:  # Fix stopping condition
                break

        print("Final Weights:", self.w)
        print("Final Bias:", self.b)

    def pred(self, X):
        return (self.sigmoid(X @ self.w + self.b) >= 0.5).astype(int)


if __name__ == "__main__":
    from collections import Counter
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, size=(100,))  # Ensure y is a column vector

    print("Training Data (X):")
    print(X)
    print("Labels (y):")
    print(y)

    LR = LogisticRegression()
    LR.fit(X, y)

    print("Predictions:")
    print(LR.pred(X))

    res = LR.pred(X)
    print(y == res)
    print(Counter(y == res))
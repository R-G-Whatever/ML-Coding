import numpy as np


def confusion_matrix(y, y_pred):
    TP = np.sum((y == 1) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    FP = np.sum((y == 0) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))
    return np.array([[TP, FN], [FP, TN]])




if __name__ == "__main__":
    np.random.seed(42)
    y = np.random.randint(0, 2, 6)
    y_pred = np.random.randint(0, 2, 6)
    print(confusion_matrix(y, y_pred))
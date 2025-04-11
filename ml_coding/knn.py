import numpy as np
import torch.nn.functional as F
import torch

class KNN:
    def __init__(self, k, temp):
        self.k = k
        self.temp = temp

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def pred(self, X_pred):
        # Normalize the input vectors to ensure cosine similarity
        X_train_norm = self.X_train / self.X_train.norm(dim=-1, keepdim=True)
        X_pred_norm = X_pred / X_pred.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between X_pred and X_train
        cosine_sim = torch.matmul(X_pred_norm, X_train_norm.T)  # (n_pred, n_train)

        # Sort based on cosine similarity (highest first)
        sorted_indices = torch.argsort(cosine_sim, dim=-1, descending=True)  # (n_pred, k)

        # Get the labels of the k-nearest neighbors
        nearest_labels = self.y_train[sorted_indices[:, :self.k]]  # (n_pred, k)

        # Perform majority voting (mode) over the k nearest labels
        majority_labels = torch.mode(nearest_labels, dim=1).values
        return majority_labels

if __name__ == "__main__":
    knn = KNN(k=3, temp=0.1)
    X_train = torch.rand(10, 5)  # 10 training samples, 5 features each
    y_train = torch.randint(0, 3, (10, 1))  # 10 training labels, random integers (0, 1, 2)

    knn.fit(X_train, y_train)

    X_pred = torch.rand(2, 5)  # 2 prediction samples, 5 features each
    predictions = knn.pred(X_pred)

    print(predictions)
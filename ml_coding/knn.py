import numpy as np
import torch.nn.functional as F
import torch
from idna import valid_label_length


class KNN:
    # Is KNN a supervised or unsupervised learning algorithm?
        # KNN is a supervised learning algorithm
    # What is the input and output of the KNN algorithm?
        # X_train, y_train, K, temp
    # What are the trainable parameters in KNN?
    # Do we need a predict function for KNN?

    # Clear define distance
    # distance metrics: Euclidean, Manhattan, Cosine
    # Euclidean distance: sqrt(sum((x1 - x2)^2))
    # Manhattan distance: sum(abs(x1 - x2))
    # Cosine distance: 1 - (x1 * x2) / (||x1|| * ||x2||)

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
        # X_train: (m, n)
        # X_pred: (m_pred, n)
        # Time complexity: O(m_train * m_pred * n)

        cosine_sim = torch.matmul(X_pred_norm, X_train_norm.T)  # (n_pred, n_train)

        # A, B
        # cosine_sim = (A, A-A1, A_A2) (B, B_A1, B_A2)

        # Sort based on cosine similarity (highest first)
        sorted_indices = torch.argsort(cosine_sim, dim=-1, descending=True)  # (n_pred, k)

        # Get the labels of the k-nearest neighbors
        nearest_labels = self.y_train[sorted_indices[:, :self.k]]  # (n_pred, k)

        # Perform majority voting (mode) over the k nearest labels
        majority_labels = torch.mode(nearest_labels, dim=1).values
        # [1,2,3,4,4,2,4,2] - Mode 2 or 4 is mode?
        # torch.mode 2
        return majority_labels  # shape: (n_pred,)

    def predict_prob(self, X_pred):
        # Normalize to get cosine similarity
        X_train_norm = self.X_train / self.X_train.norm(dim=-1, keepdim=True)
        X_pred_norm = X_pred / X_pred.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        cosine_sim = torch.matmul(X_pred_norm, X_train_norm.T)  # (n_pred, n_train)

        # For each prediction, find top-k neighbors
        topk_sim, topk_indices = torch.topk(cosine_sim, self.k, dim=-1)  # (n_pred, k)

        # Apply temperature scaling
        scaled_sim = topk_sim / self.temp

        # Convert to softmax weights
        weights = F.softmax(scaled_sim, dim=-1)
        # topk_sim = [4, 3, 3] ---> [0.4, 0.3, 0.3]
        # topk_index = [1, 2, 1]
        # class 1: 0.4 + 0.3
        # class 2: 0.3
        # (n_pred, k)

        # Gather labels of top-k neighbors
        topk_labels = self.y_train[topk_indices]  # (n_pred, k)

        # Compute class probabilities
        n_classes = torch.max(self.y_train).item() + 1
        pred_probs = torch.zeros(X_pred.size(0), n_classes, device=X_pred.device)

        for i in range(X_pred.size(0)):
            for j in range(self.k):
                label = topk_labels[i, j].item()
                pred_probs[i, label] += weights[i, j]

        return pred_probs  # shape: (n_pred, n_classes)

if __name__ == "__main__":
    # knn = KNN(k=3, temp=0.1)
    # X_train = torch.rand(100, 5)  # 10 training samples, 5 features each
    # y_train = torch.randint(0, 3, (100, 1))  # 10 training labels, random integers (0, 1, 2)
    #
    # knn.fit(X_train, y_train)
    #
    # X_pred = torch.rand(2, 5)  # 2 prediction samples, 5 features each
    # predictions = knn.pred(X_pred)
    # print(predictions)
    #
    # pred_prob = knn.predict_prob(X_pred)
    # print(pred_prob)

    A = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(torch.argsort(A, dim=-1, descending=True))
    # print(torch.topk(A, 2, dim=-1))



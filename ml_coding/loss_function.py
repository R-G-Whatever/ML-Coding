import numpy as np


def BinaryCrossEntropy(y, y_pred, epsilon = 1e-12):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    res = y*np.log(y_pred) + (1-y)*np.log(1-y_pred)
    return -np.mean(res)

def MeanSquareLossError(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def CrossEntropyLoss(y, y_pred, epsilon = 1e-12):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    res = -np.sum(y * np.log(y_pred)) / y.shape[0]
    return res

def CrossEntropyLossIndex(y, y_pred, epsilon = 1e-12):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    res = -np.mean(np.log(y_pred[np.arange(len(y)), y]))
    return res

import torch.nn.functional as F
import torch
def ContrastiveLoss(X1, X2, label, margin=0.5):
    cosine_sim = F.cosine_similarity(X1, X2, dim=1)
    cosine_distance = 1 - cosine_sim
    loss = (label * torch.pow(cosine_distance, 2) + (1-label) * torch.pow(torch.clamp(cosine_distance - margin, min = 0.0), 2))
    return loss.mean()
def Softmax(X):
    exp_x_shift = np.exp(X - np.max(X, axis = 1, keepdims=True))
    return exp_x_shift / np.sum(exp_x_shift, axis = 1)

if __name__ == "__main__":

    # Test MSE loss
    y = np.random.rand(4,)
    y_pred = np.random.rand(4,)
    print(MeanSquareLossError(y, y_pred))

    # Test BCE loss
    y = np.random.randint(0, 2, 5)
    y_pred = np.random.rand(5,)
    print(BinaryCrossEntropy(y, y_pred))

    y = np.array([
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1]  # Class 2
    ])

    y_pred = np.array([
        [0.9, 0.05, 0.05],  # Model predicts class 0 with high confidence
        [0.1, 0.8, 0.1],  # Model predicts class 1
        [0.2, 0.2, 0.6]  # Model predicts class 2
    ])
    print(CrossEntropyLoss(y, y_pred))

    y_index = [0,1,2]
    print(CrossEntropyLossIndex(y_index, y_pred))

    print(Softmax(y_pred))

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    x2 = torch.tensor([[1.1, 2.1], [5.0, 6.0]], dtype=torch.float32)
    label = torch.tensor([1, 0], dtype=torch.float32)  # First pair is similar, second is dissimilar

    loss = ContrastiveLoss(x1, x2, label)
    print(loss.item())
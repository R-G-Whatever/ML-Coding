import numpy as np


def BinaryCrossEntropy(y, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # set boundries for the array
    res = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    return -np.mean(res)

def CrossEntropyLoss(y, y_pred, epsilon=1e-12):
    # y -> [[1,0,0], [0,1,0]]
    # y_pred -> [[0.5, 0.1, 0.4], [0.2, 0.7, 0.1]]

    # one hot encoding
    # y = [A, B, C]
    # y_1 = [Is_A] = 1
    # y_2 = [Is_B] = 1
    # y = [[1,0,0], [0,1,0], [0,0,1]]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Review np.sum default is item wise sum
    print(np.sum(y * np.log(y_pred), axis = 0))
    print(np.sum(y * np.log(y_pred), axis = 1))
    res = -np.sum(y * np.log(y_pred)) / y.shape[0]
    return res

def CrossEntropyLossIndex(y, y_pred, epsilon=1e-12):
    # y_pred -> [[0.5, 0.1, 0.4], [0.2, 0.7, 0.1]]
    # y = [0, 1, 2, 2, 1] vector (m * n) to (m * 1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    print(y_pred)
    print(np.arange(len(y)))
    print([np.arange(len(y)), y])
    print(y_pred[np.arange(len(y)), y])
    res = -np.mean(np.log(y_pred[np.arange(len(y)), y])) # how it works? select element (0,0), (1,1), (2,3) and (3,2)
    return res

def MeanSquareLossError(y, y_pred):
    return np.mean((y - y_pred) ** 2)   # np.sum np.mean
                                        # when not mention axis it means by default element level operation

def CrossEntropyLossIndex(y, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    print(y_pred)
    print(np.arange(len(y)))
    print([np.arange(len(y)), y])
    print(y_pred[np.arange(len(y)), y])
    res = -np.mean(np.log(y_pred[np.arange(len(y)), y])) # how it works? select element (0,0), (1,1), (2,3) and (3,2)
    return res

import torch.nn.functional as F
import torch


def ContrastiveLoss(X1, X2, label, margin=0.5):
    cosine_sim = F.cosine_similarity(X1, X2, dim=1)
    cosine_distance = 1 - cosine_sim
    loss = label * torch.pow(cosine_distance, 2) + (1 - label) * torch.pow(
        torch.clamp(cosine_distance - margin, min=0.0), 2
    )
    return loss.mean()


def Softmax(X):
    exp_x_shift = np.exp(X - np.max(X, axis=1, keepdims=True))  # What if we don't write keepdims = True?
    # Counter example X = [1e6, 1e6, 1e6]
    # print(exp_x_shift)
    return exp_x_shift / np.sum(exp_x_shift, axis=1, keepdims = True)


if __name__ == "__main__":

    # A = [[1,2,3], [2,3,4], [3,4,5], [3,4,6]]
    # print(np.mean(A))
    # print(np.mean(A, axis = 0))
    # print(np.mean(A, axis = 1))

    # # Test MSE loss
    # y = np.random.rand(
    #     4,
    # )
    # y_pred = np.random.rand(
    #     4,
    # )
    # print(MeanSquareLossError(y, y_pred))
    #
    # # Test BCE loss
    # y = np.random.randint(0, 2, 5)
    # y_pred = np.random.rand(
    #     5,
    # )
    # print(BinaryCrossEntropy(y, y_pred))

    # y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Class 0  # Class 1  # Class 2
    #
    # y_pred = np.array(
    #     [
    #         [0.9, 0.05, 0.05],  # Model predicts class 0 with high confidence
    #         [0.1, 0.8, 0.1],  # Model predicts class 1
    #         [0.2, 0.2, 0.6],  # Model predicts class 2
    #         [0.1, 0.3, 0.6],
    #     ]
    # )
    # # # print(CrossEntropyLoss(y, y_pred))
    # #
    # y_index = [0, 1, 2, 2]
    # print(CrossEntropyLossIndex(y_index, y_pred))
    #
    # print(Softmax(y_pred))
    #
    # x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    # x2 = torch.tensor([[1.1, 2.1], [5.0, 6.0]], dtype=torch.float32)
    # label = torch.tensor(
    #     [1, 0], dtype=torch.float32
    # )  # First pair is similar, second is dissimilar
    #
    # loss = ContrastiveLoss(x1, x2, label)
    # print(loss.item())

    # X = [[1,2,3], [2,3,4], [4,5,6], [7,8,9]]
    # print(np.max(X, axis=1, keepdims=True))
    # print(np.max(X, axis=1))
    # X = [[1e6, 1e6, 1e6], [1e6, 1e6, 2e6]]
    # X = [[1,2,3], [2,3,4]]
    # print(Softmax(X))
    X = [[1e6, 1e6, 1e6], [1e6, 1e6, 1e6]]
    def mock_softmax(X):
        return np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)

    print(Softmax(X))
    print(mock_softmax(X))

    # print(np.clip(X, 1, 3))


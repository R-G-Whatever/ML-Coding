import numpy as np
import torch
class EvalMetrics:
    def __init__(self):
        self.metrics = []

    @staticmethod
    def cal_conf_matrix(y_pred: np.array, y: np.array):
        # Numpy array is different from list especially for masking logic
        # Numpy array is also similar to torch tensor
        TP = np.sum((y_pred == True) & (y == True))
        # use np.sum as a counter
        TN = np.sum((y_pred == False) & (y == False))
        FN = np.sum((y_pred == False) & (y == True))
        FP = np.sum((y_pred == True) & (y == False))
        print(f"TP: {TP}    FN: {FN}")
        print(f"FP: {FP}    TN: {TN}")
        return [[TP, FN], [FP, TN]]

    @staticmethod
    def cal_recall(y_pred, y):
        # recall = TP / TP + FN
        # True positive rate - what is the percentage of true positive in all true positives
        # What is the range of precision and recall?
        # From 0 to 1
        TP = np.sum((y_pred == True) & (y == True))
        # Think why TP = np.sum((y_pred == True) and (y == Ture)) does not work?
        FN = np.sum((y_pred == False) & (y == True))
        # Divider cannot be zero
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    @staticmethod
    def cal_precision(y_pred, y):
        # precision = TP / TP + FP what is the percentage of true positive in all predicted positives
        TP = np.sum((y_pred == True) & (y == True))
        FP = np.sum((y_pred == True) & (y == False))
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)


    @staticmethod
    def recall_multi_class(y_pred, y):
        # output is not a number
        # one hot encoding [1,2,3] ---> is_1; is_2, is_3
        recall = {} # recall for class 1; recall for class 2; recall for class 3;
        # how many class we have? ---> depend on the kind of distinct element within y
        distinct_class = set(y)
        for item in distinct_class:
            TP = np.sum((y_pred == item) & (y == item))
            FN = np.sum((y_pred != item) & (y == item))
            if TP + FN == 0:
                recall[item] = 0    # dictionary key is the class name, item is the recall value
            else:
                recall[item] = TP / (TP + FN)
        return recall

    # Homework 1:
    # PR-AUC
    # ROC-AUC


if __name__ == "__main__":
    y_pred = np.array([0, 1, 1, 0, 1])
    y = np.array([0, 0, 1, 0, 1])

    y_pred_list = [0, 1, 1, 0, 1]
    print(y_pred_list == True)
    print(y_pred == True)

    # K1 - use numpy array as filter; filter is used as a mask
    y_pred_multi_class = np.array([1,2,3,3,3,3,1,2])
    filter = (y_pred_multi_class == 2)
    print(y_pred_multi_class * filter)

    eval = EvalMetrics()
    eval.cal_conf_matrix(y_pred, y)

    # K2 - how to use numpy.sum, different axis for adding

    matrix = np.array([[1,2,3,4,5], [2,3,4,5,6]])
    print(matrix)
    print(np.sum(matrix)) # default item wise adding
    print(np.sum(matrix, axis = 1)) # axis = 1 column wise adding
    print(np.sum(matrix, axis = 0)) # axis = 0 row wise adding
    print(np.sum(matrix, axis = -1)) # in this case it is column wise adding by default shape (2,5)

    recall = eval.cal_recall(y_pred, y)
    print(recall)           # What is the range of recall and precision?
                            # What does recall == 1 means?

    precision = eval.cal_precision(y_pred, y)
    print(precision)

    # K3 - matrix multiplication dot product; matrix division is item wise operation
    A = np.array([1,2,3])
    B = np.array([0,3,4])
    # print(A/B)
    print(A*B)

    # K4 - matrix multiplication use @ or np.matmul
    A = np.array([[1,2,3],[2,3,4]])     # shape (2,3)
    B = np.array([[1,2],[2,3], [3,4]])  # shape (3,2)

    print(A@B)
    print(np.matmul(A,B))

    # K5 - multi class eval metrics such as recall and precision
    y_pred = np.array([1,2,3,2,2,2,2])
    y = np.array([1,2,3,3,3,2,2])
    # how to calculate recall?
    eval = EvalMetrics()
    print(eval.recall_multi_class(y_pred, y))

    # Homework write multi_class calculation for Precision
import numpy as np


def roc_auc(y, y_pred):
    index_sort = np.argsort(y_pred)[::-1]
    TP, FP = 0, 0
    FN = np.sum(y)
    TN = len(y) - FN
    y = y[index_sort]
    tpr = [0]
    fpr = [0]

    for i in range(len(index_sort)):
        if y[i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)

    # Compute AUC using Trapezoidal Rule
    auc = np.trapz(tpr, fpr)

    return tpr, fpr, auc



if __name__ == "__main__":
    y = np.random.randint(0,2,10)
    y_pred = np.random.rand(10,)
    # print(y)
    # print(y_pred)
    roc_auc(y, y_pred)
    print(roc_auc(y, y_pred))
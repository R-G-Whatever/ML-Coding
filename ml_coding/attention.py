import numpy as np

class SingleHeadAttention:
    def __init__(self, d_k):
        self.d_k = d_k


    def compute_attention(self, Q, K, V, mask = None):

        scores = np.matmul(Q, K.transpose(0, 2, 1))

        scores = scores / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_w = np.exp(scores - np.max(scores, axis = -1, keepdims = True))
        attention_w = attention_w / np.sum(attention_w, axis = -1, keepdims= True)

        output = np.matmul(attention_w, V)

        return output, attention_w

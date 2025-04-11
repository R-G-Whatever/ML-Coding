import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.w_k = nn.Linear(d_in, d_out)
        self.w_q = nn.Linear(d_in, d_out)
        self.w_v = nn.Linear(d_in, d_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        Q = self.w_q(X) #(b, n_seq, d_out)
        K = self.w_k(X)
        V = self.w_v(X)

        atten_score = torch.matmul(Q, K.transpose(1,2)) / (self.d_out ** 0.5)
        if mask is not None:
            atten_score.masked_fill(mask == 0, -float('inf'))
        atten_weight = F.softmax(atten_score, dim = 2) # (b, n_seq, n_seq)
        atten_weight = self.dropout(atten_weight)
        return atten_weight @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, num_head, dropout = 0.1):
        super().__init__()
        assert d_out % num_head == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // num_head
        self.d_in = d_in
        self.d_out = d_out
        self.num_head = num_head

        self.w_k = nn.Linear(d_in, d_out)
        self.w_q = nn.Linear(d_in, d_out)
        self.w_v = nn.Linear(d_in, d_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask = None):
        batch_size, n_seq, _ = X.shape
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        Q = Q.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1,2)

        atten_score = torch.matmul(Q, K.transpose(-1,-2)) / (self.head_dim ** 2)
        if mask is not None:
            atten_score.masked_fill(mask == 0, float("-inf"))

        atten_weight = F.softmax(atten_score, dim = -1)
        atten_weight = self.dropout(atten_weight)
        output = atten_weight @ V
        output = output.transpose(1,2).contiguous().view(batch_size, n_seq, self.d_out)

        return output


if __name__ == "__main__":
    x = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
         [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
         [0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6],
         [0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5]],

        [[0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
         [0.6, 0.5, 0.8, 0.7, 0.2, 0.1, 0.4, 0.3],
         [0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2],
         [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    ], dtype=torch.float32)

    mask = torch.tensor([
        [[1, 1, 1, 0],  # First sequence: last token is masked
         [1, 1, 1, 0],  # Second sequence: last token is masked
         [1, 1, 1, 0],  # Third sequence: last token is masked
         [1, 1, 1, 1]],  # Fourth sequence: no masking

        [[1, 1, 0, 0],  # First sequence: last two tokens are masked
         [1, 1, 1, 0],  # Second sequence: last token is masked
         [1, 1, 1, 0],  # Third sequence: last token is masked
         [1, 1, 1, 1]]  # Fourth sequence: no masking
    ], dtype=torch.float32)

    atten = SelfAttention(8, 6, 0.1)
    print(atten(x, mask = mask))

    atten_mul = MultiHeadSelfAttention(8, 6, 2, 0.1)
    print(atten_mul(x, mask=mask))
import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X,W_q)
    K = np.dot(X,W_k)
    V = np.dot(X,W_v)
    return Q, K, V


def softmax(x):
    x_max = np.max(x)
    return np.exp(x-x_max)/np.sum(np.exp(x-x_max), axis=-1)  # remember to use axis here


def self_attention(Q, K, V):
    d_k = K.shape[-1]
    attention = (np.dot(Q, K.T)/np.sqrt(d_k))
    attention = softmax(attention)
    attention = np.dot(attention, V)
    return attention


X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
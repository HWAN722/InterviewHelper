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
    d_k = K.shape[-1]  # d_k for self attention
    attention = (np.dot(Q, K.T)/np.sqrt(d_k))
    attention = softmax(attention)
    attention = np.dot(attention, V)
    return attention

def multi_head_attention(Q, K, V, n_heads):
    d_model = K.shape[-1]
    d_k = d_model // n_heads  # d_k for multihead

    Q = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)

    attention = []

    for i in range(n_heads):
        attention.append(self_attention(Q[i],K[i],V[i]))
    return attention

Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 0], [0, 1]])
n_heads = 2
output = multi_head_attention(Q, K, V, n_heads)

print(output)



import numpy as np


def softmax(z):
    z = np.exp(z - np.max(z, axis=-1, keepdims=True)) # prevent overflow
    return z/np.sum(z, axis=-1, keepdims=True)


def cross_entropy(y_pred, y_true):
    z = softmax(y_pred)

    eps = 1e-15 # prevent log(0)
    z = np.clip(z, eps, 1-eps)

    return -np.mean(np.sum(y_true * np.log(z), axis=-1))


y_pred = [2, 1, 0.1]
y_true = [0, 1, 0]
y_pred = np.array(y_pred)
y_true = np.array(y_true)
print(cross_entropy(y_pred, y_true))
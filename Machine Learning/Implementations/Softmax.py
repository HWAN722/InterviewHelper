import numpy as np

def softmax(z):
    z = np.exp(z - np.max(z, axis=0, keepdims=True)) # prevent overflow
    return z/np.sum(z, axis=0, keepdims=True)

z = [2, 1, 1]
z = np.array(z)
print(softmax(z))
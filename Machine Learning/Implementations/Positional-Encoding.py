import numpy as np


def pos_encoding(position: int, d_model: int):
    pos_encoding = np.zeros([position,d_model])
    for pos in range(len(pos_encoding)):
        for i in range(len(pos_encoding[0])):
            if i % 2 == 0:
                pos_encoding[pos][i] = np.sin(pos/10000 ** (i/d_model))
            else:
                pos_encoding[pos][i] = np.cos(pos / 10000 ** ((i-1) / d_model))
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding


print(pos_encoding(2,8))
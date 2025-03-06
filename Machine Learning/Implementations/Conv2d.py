import numpy as np


def simple_conv2d(input_matrix, kernel, padding: int, stride: int):
    if padding:
        input_matrix = np.pad(input_matrix, padding)

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    height = (input_height - kernel_height) // stride + 1
    width = (input_width - kernel_width) // stride + 1
    output_matrix = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            output = input_matrix[i*stride:i*stride+kernel_width,j*stride:j*stride+kernel_height]
            output_matrix[i,j] = np.sum(output * kernel)


    return output_matrix


input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train_neuron(features, labels, weights, bias: float, learning_rate: float, epochs: int):
    weights = np.array(weights)
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predicts = sigmoid(z)

        mse = np.mean((predicts-labels) ** 2)
        mse_values.append(round(mse,4))

        loss = predicts - labels
        weights_gradient = (2/len(labels)) * np.dot(features.T, loss * np.mean(predicts) * (1 - predicts))
        bias_gradient = (2/len(labels)) * np.sum(loss * np.mean(predicts) * (1 - predicts))

        weights -= weights_gradient * learning_rate
        bias -= bias_gradient * learning_rate

        weights = np.round(weights, 4)
        bias = np.round(bias, 4)

    return weights, bias, mse_values

print(train_neuron([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], weights = [0.1, -0.2], bias = 0.0, learning_rate = 0.1, epochs = 2))
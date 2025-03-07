import numpy as np

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    hidden_state = np.array(initial_hidden_state)
    Wh = np.array(Wh)
    Wx = np.array(Wx)
    b = np.array(b)

    for seq in input_sequence:
        seq = np.array(seq)
        hidden_state = np.tanh(np.dot(Wh,hidden_state) + np.dot(Wx,seq) + b)
        print(hidden_state)

    hidden_state = np.round(hidden_state, 4)

    return hidden_state.tolist()

input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]
Wx = [[0.5]]  # Input to hidden weights
Wh = [[0.8]]  # Hidden to hidden weights
b = [0.0]     # Bias
output = rnn_forward(input_sequence,initial_hidden_state,Wx,Wh,b)
print(output)
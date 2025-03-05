import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

outputs = np.array([[0], [1], [1], [0]])

np.random.seed(1)
weights_input_hidden = 2 * np.random.random((2, 2)) - 1
weights_hidden_output = 2 * np.random.random((2, 1)) - 1

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_layer_input)
    
    error = outputs - final_output
    
    delta_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = delta_output.dot(weights_hidden_output.T)
    delta_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += inputs.T.dot(delta_hidden_layer) * learning_rate

print(final_output)

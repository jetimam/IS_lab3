import numpy as np

#input nodes
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(input)

#output nodes
expected_output = np.array([[0], [1], [1], [0]])
print(expected_output, '\n')

#weights
weights = np.array([[0.1], [0.2]])

#bias
bias = 0.3

#activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#derivate of the activation fuction
def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))

#training
for epoch in range(10000):
	#forward propogation
	weights_sum = np.dot(input, weights) + bias
	output = sigmoid(weights_sum)

	error = output - expected_output #gap between expected output and output
	total_error = np.square(np.subtract(output, expected_output)).mean()

	derivative = error * sigmoid_derivative(output)

	t_input = input.T
	final_derivative = np.dot(t_input, derivative)

	#backward propogation
	weights = weights - 0.05 * final_derivative
	for i in derivative:
		bias = bias - 0.05 * i

pred = np.array([1, 1])
print('predicting (1,1):')

result = np.dot(pred, weights) + bias

res = sigmoid(result)

print(res)
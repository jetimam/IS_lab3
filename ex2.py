import numpy as np
import matplotlib.pyplot as plt

#input nodes
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(input)

loss_arr = []
epoch_arr = []
for i in range(1,10001):
	epoch_arr.append(i)

#output nodes
expected_output = np.array([[0], [1], [1], [1]])
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

	#backward propogation
	error = output - expected_output #gap between expected output and output
	total_error = np.square(np.subtract(output, expected_output)).mean()
	print('Training:', error)
	loss_arr.append(error[0])

	derivative = error * sigmoid_derivative(output)

	t_input = input.T
	final_derivative = np.dot(t_input, derivative)
	weights = weights - 0.05 * final_derivative
	for i in derivative:
		bias = bias - 0.05 * i

#prediction
pred = np.array([0, 0])
print('predicting (0, 0):', end=' ')
result = np.dot(pred, weights) + bias
res = sigmoid(result)
print(res)

# plt.plot(epoch_arr, loss_arr)
# plt.xlabel('Epochs')
# plt.ylabel('Expected Output - Epoch Output')
# plt.show()
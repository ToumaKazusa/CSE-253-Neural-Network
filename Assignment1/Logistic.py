from Loader import MNIST
import numpy as np


[ims, labels] = MNIST().load_training()


ones  = np.ones((1, 20000))
train = np.array(ims[0:20000])
train = np.concatenate((train.T, ones), axis = 0).T

weight = np.ones((785, 1))
for i in range(0, 785):
 	weight[i] = 0.01

for loop in range(0, 1000):
	y = train.dot(weight)
	err = 0
	for i in range(0, 20000):
		if labels[i] == 2:
			err += 1 - y[i]
			for j in range(0, 785):
				weight[j] += 0.001 * (y[i] - 1) * train[i][j]
		if labels[i] == 3:
			err += - 1 - y[i]
			for j in range(0, 785):
				weight[j] += 0.001 * y[i] * train[i][j]
	print(err)



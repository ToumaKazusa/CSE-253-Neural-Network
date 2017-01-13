from Loader import MNIST
import numpy as np


[ims, labels] = MNIST().load_training()


ones  = np.ones((1, 20000))
train = np.array(ims[0:20000])
train = np.concatenate((train.T, ones), axis = 0).T

weight = np.ones((785, 1))
for i in range(0, 785):
 	weight[i] = 0.001

sums = np.zeros((785, 1))
for i in range(0, 20000):
	if labels[i] == 2 | labels[i] == 3:
		for j in range(0, 785):
			sums[j] += 1


for loop in range(0, 1000):
	y = train.dot(weight)
	grad = 0
	for i in range(0, 20000):
		if labels[i] == 2:
			grad += 1 - y[i]
		if labels[i] == 3:
			grad += -1 - y[i]
	for i in range(0, 785):
		weight[i] += 0.000000001*grad
	print(grad)



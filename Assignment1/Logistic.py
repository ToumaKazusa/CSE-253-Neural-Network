from Loader import MNIST
import numpy as np


[ims, labels] = MNIST().load_training()



data = []
tar = []
for i in range(20000):
	ims[i].append(1)
	if labels[i] == 2:
		data.append(ims[i])
		tar.append(1)
	if labels[i] == 3:
		data.append(ims[i])
		tar.append(0)

weight = []
det = []
for i in range(785):
	weight.append(0)
	det.append(0)

for loop in range(1000):
	error = 0
	for j in range(785):
		det[j] = 0
	for i in range(len(data)):
		y = 0
		for j in range(785):
			y += data[i][j] * weight[j]
		error += tar[i] - y
		for j in range(785):
			det[j] += (tar[i] - y) * data[i][j]
	for j in range(785):
		weight[j] += 0.00000000003 * loop * det[j]
	print(error)








#ones  = np.ones((1, 20000))
#train = np.array(ims[0:20000])
#train = np.concatenate((train.T, ones), axis = 0).T

#weight = np.zeros((785, 1))


#for loop in range(0, 1000):
#	y = train.dot(weight)
#	err = 0
#	for i in range(0, 20000):
#		if labels[i] == 2:
#			err += 1 - y[i]
#			for j in range(0, 785):
#				weight[j] -= 0.0000001 * (1 - y[i]) * train[i][j]
#		if labels[i] == 3:
#			err += - 1 - y[i]
#			for j in range(0, 785):
#				weight[j] -= 0.0000001 * (- y[i]) * train[i][j]
#	print(weight)
#	print(err)



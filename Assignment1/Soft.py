from Loader import MNIST
import numpy as np
import math

# Fix step length, load data
step = .00001
regular = .01
[ims, labels] = MNIST().load_training()


# Create training set and verification set
for i in range(20000):
	ims[i].append(1)

ims_t = []
ims_v = []
lbs_t = []
lbs_v = []

for i in range(18000):
	temp = np.array(ims[i]).reshape(1, 785) / 255.0
	ims_t.append(temp)
	temp = np.zeros((1, 10))
	temp[0][labels[i]] = 1
	lbs_t.append(temp)

for i in range(2000):
	temp = np.array(ims[i + 18000]).reshape(1, 785) / 255.0
	ims_v.append(temp)
	temp = np.zeros((1, 10))
	temp[0][labels[i + 18000]] = 1
	lbs_v.append(temp)

# Initial weight matrix of Softmax Classifier
weight = np.zeros((785, 10))

for loop in range(1000):
	
	det = np.zeros((785, 10))
	for i in range(18000):
		a = ims_t[i].dot(weight)
		y = np.zeros((1, 10))
		for j in range(10):
			y[0][j] = math.exp(a[0][j])
		y /= np.sum(y)
		e = lbs_t[i] - y
		det += ims_t[i].T.dot(e)	
	weight = np.add((1 - regular) * weight, step * det)

	false = 0
	error = 0
	for i in range(2000):
		a = ims_t[i].dot(weight)
		for j in range(10):
			y[0][j] = math.exp(a[0][j])
		y /= np.sum(y)
		m = 0
		for j in range(10):
			if y[0][j] > y[0][m]:
				m = j
			if lbs_t[i][0][j] == 1:
				error += 1 - y[0][j]
		if lbs_t[i][0][m] == 0:
			false += 1
	print("false:" + str(false))
	print("error:" + str(error))
	print("---------------")

	





from Loader import MNIST
import numpy as np

# Fix step length, load data
step = 0.00000000015
[ims, labels] = MNIST().load_training()


# Create nparray for every sample with label '2' & '3', and mark label as 1 for '2' and 0 for '3'
image = []
label = []
for i in range(20000):
	ims[i].append(1)
	if labels[i] == 2:
		image.append(np.array(ims[i]).reshape((1, 785)))
		label.append(1)
	if labels[i] == 3:
		image.append(np.array(ims[i]).reshape((1, 785)))
		label.append(0)

# Divide into two saperate part, the first 90 percent to be training set, and remain to be verification set
n = len(image)
m = int(0.9 * n)

ims_t = image[0:m]
ims_v = image[m:n]
lab_t = label[0:m]
lab_v = label[m:n]

weight = np.zeros((785, 1))

# Fixed loop, using gradient descent
for loop in range(1000):
	# Update weight
	det = np.zeros((785, 1))
	for i in range(m):
		y = ims_t[i].dot(weight)
		det += (lab_t[i] - y[0][0]) * ims_t[i].T
	weight = np.add(weight, step * det)

	# Calculate error rate on verification set
	fn = 0
	fp = 0
	err = 0
	for i in range(n - m):
		y = ims_v[i].dot(weight)
		if y[0][0] - lab_v[i] > 0.5:
			fn += 1
		if lab_v[i] - y[0][0] > 0.5:
			fp += 1
		err += abs(lab_v[i] - y[0][0])
	print("false negative:" + str(fn))
	print("false positive:" + str(fp))
	print("error:" + str(err))
	print("---------------")
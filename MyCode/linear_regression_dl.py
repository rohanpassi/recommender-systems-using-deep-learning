import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from loadData import loadMovieLens

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
x_train, y_train, x_test, y_test =  loadMovieLens()
# x_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

# n_samples = x_train.shape[0]
n_samples = len(x_train)
print(n_samples)

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name = "weight")
b = tf.Variable(rng.randn(), name = "bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean Squared Error
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2*n_samples)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing all the variables
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# Fit all training data
	for epoch in range(training_epochs):
		for (x, y) in zip(x_train, y_train):
			sess.run(optimizer, feed_dict={X:x, Y:y})

		# Display logs per epoch step
		if (epoch + 1) % display_step == 0:
			c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
			print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(c), "W = ", sess.run(W), "b = ", sess.run(b))
	print("Optimization finished")
	training_cost = sess.run(cost, feed_dict={X:x_train, Y:y_train})
	print("Training cost = ", training_cost, "W = ", sess.run(W), "b = ", sess.run(b))

	# Graphic Display
	plt.plot(x_train, y_train, 'ro', label="Original Data")
	plt.plot(x_train, sess.run(W)*x_train + sess.run(b), label="Fitted Line")
	plt.legend()
	plt.show()
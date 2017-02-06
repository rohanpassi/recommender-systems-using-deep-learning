import tensorflow as tf


x = tf.Variable([[1, 2], [3, 4]], name='x')
init = tf.initialize_all_variables()
with tf.Session() as sess:
	print(sess.run(x))
import tensorflow as tf
from tensorflow_models import *

with tf.Session() as sess:
	train_data, test_data = merge_data()
	# model_dir = "/home/rohan/Documents/MTPS/Rohan/MyCode/wide_n_deep_model"
	model_dir = "/home/rohan/Documents/MTPS/Rohan/MyCode/deep_model"
	# model_dir = "/home/rohan/Documents/MTPS/Rohan/MyCode/wide_model"

	classifier = build_estimator(model_dir)
	results = classifier.predict(input_fn=lambda: input_fn(test_data))
	results = min_max_normalization(results, 0, 1)
	checkAccuracy(results, test_data)
	print(RMSE(results, test_data[LABEL_COLUMN]))
import tensorflow as tf
from tensorflow_models import *

with tf.Session() as sess:
	train_data,test_data = merge_data()
	model_dir = "/home/rohan/Documents/MTPS/Rohan/MyCode/model"

	classifier = build_estimator(model_dir)
	results = classifier.predict(input_fn=lambda: input_fn(test_data))
	results = min_max_normalization(results, 1, 5)
	print(RMSE(results, test_data[LABEL_COLUMN]))
	print(MAE(results, test_data[LABEL_COLUMN]))


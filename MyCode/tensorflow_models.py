from __future__ import absolute_import, division, print_function


import sys
import time
import pprint
import argparse
import tempfile
from math import sqrt
from dateutil import parser
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.learn as tflearn

tf.logging.set_verbosity(tf.logging.ERROR)


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 500, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")


COLUMNS = ['user_id', 'movie_date', 'age', 'sex', 'occupation', 'zip_code', 'rating', 'rating_date', 'movie_id', 'unknown', 'action', 'adventure', 
	'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']

LABEL_COLUMN = "rating"

CATEGORICAL_COLUMNS = ['sex', 'occupation', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
	'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']

CONTINUOUS_COLUMNS = ['age','time_diff']



def RMSE(y_pred, y_test):
	diff = y_pred - y_test
	diff = diff ** 2
	diff = np.mean(diff)
	diff = np.sqrt(diff)
	return diff

def MAE(y_pred, y_test):
	diff = np.absolute(y_pred - y_test)
	diff = np.mean(diff)
	return diff

def min_max_normalization(data, new_min, new_max):
	result = []
	maxE = np.max(data)
	minE = np.min(data)
	for i in range(0,len(data)):
		tmp = new_min + ((data[i]-minE) * (new_max - new_min) / (maxE-minE))
		result.append(tmp)
	result = np.array(result)
	return result

def z_score_normalization(data):
	std = np.std(data)
	mean = np.mean(data)
	data = (data - mean) / std
	return data

def changeDateFromString(date):
	dt = parser.parse(date)
	return dt

def changeDateFromTimestamp(date):
	dt = datetime.fromtimestamp(date)
	return dt

def dateDiff(movieDate, ratingDate):
	diff = ratingDate - movieDate
	return int(abs(diff.total_seconds()))

def load_users():
	u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
	users = pd.read_csv('../ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1', engine="python")
	users['age'] = min_max_normalization(users['age'], 0, 1)
	return users

def load_ratings(filename):
	r_cols = ['user_id', 'movie_id', 'rating', 'rating_date']
	ratings = pd.read_csv('../ml-100k/' + filename, sep='\t', names=r_cols, encoding='latin-1', engine="python")
	return ratings

def load_movies():
	m_cols = ['movie_id', 'movie_date', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
	'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']
	cols = range(5, 24)
	cols.insert(0, 2)
	cols.insert(0, 0)
	movies = pd.read_csv('../ml-100k/u.item', sep='|', names=m_cols, usecols=cols, encoding='latin-1', engine="python")
	movies = movies.dropna()
	di = {0:'N', 1:'Y'}
	for k in m_cols:
		if k=='movie_id' or k=='movie_date':
			continue
		else:
			movies = movies.replace({k:di})
	return movies

def load_time(filename):
	col_name = ['time_diff']
	time = pd.read_csv(filename, names=col_name, encoding='latin-1', engine='python')
	train_data['time_diff'] = (train_data["rating"].astype(int))
	train_data['time_diff'] = min_max_normalization(train_data['time_diff'], 0, 1)
	return time

def merge_data():
	users = load_users()
	movies = load_movies()
	train_ratings = load_ratings('u1.base')
	train_movie_ratings = pd.merge(movies, train_ratings)
	train_data = pd.merge(train_movie_ratings, users)
	time_diff = load_time('train_time.csv')
	train_data = pd.concat([train_data, time_diff], axis=1)


	test_ratings = load_ratings('u1.test')
	test_ratings['rating'] = min_max_normalization(test_ratings['rating'], 0, 1)
	test_movie_ratings = pd.merge(movies, test_ratings)
	test_data = pd.merge(test_movie_ratings, users)
	time_diff = load_time('test_time.csv')
	test_data = pd.concat([test_data, time_diff], axis=1)

	return train_data, test_data


def build_estimator(model_dir):
	"""Build an Estimator"""
	
	# Sparse base Columns
	gender = tf.contrib.layers.sparse_column_with_keys(column_name="sex", keys=["F", "M"])
	unknown = tf.contrib.layers.sparse_column_with_keys(column_name="unknown", keys=['Y', 'N'])
	action = tf.contrib.layers.sparse_column_with_keys(column_name="action", keys=['Y', 'N'])
	adventure = tf.contrib.layers.sparse_column_with_keys(column_name="adventure", keys=['Y', 'N'])
	animation = tf.contrib.layers.sparse_column_with_keys(column_name="animation", keys=['Y', 'N'])
	children = tf.contrib.layers.sparse_column_with_keys(column_name="children", keys=['Y', 'N'])
	comedy = tf.contrib.layers.sparse_column_with_keys(column_name="comedy", keys=['Y', 'N'])
	crime = tf.contrib.layers.sparse_column_with_keys(column_name="crime", keys=['Y', 'N'])
	documentary = tf.contrib.layers.sparse_column_with_keys(column_name="documentary", keys=['Y', 'N'])
	drama = tf.contrib.layers.sparse_column_with_keys(column_name="drama", keys=['Y', 'N'])
	fantasy = tf.contrib.layers.sparse_column_with_keys(column_name="fantasy", keys=['Y', 'N'])
	filmnoir = tf.contrib.layers.sparse_column_with_keys(column_name="filmnoir", keys=['Y', 'N'])
	horror = tf.contrib.layers.sparse_column_with_keys(column_name="horror", keys=['Y', 'N'])
	musical = tf.contrib.layers.sparse_column_with_keys(column_name="musical", keys=['Y', 'N'])
	mystery = tf.contrib.layers.sparse_column_with_keys(column_name="mystery", keys=['Y', 'N'])
	romance = tf.contrib.layers.sparse_column_with_keys(column_name="romance", keys=['Y', 'N'])
	scifi = tf.contrib.layers.sparse_column_with_keys(column_name="scifi", keys=['Y', 'N'])
	thriller = tf.contrib.layers.sparse_column_with_keys(column_name="thriller", keys=['Y', 'N'])
	war = tf.contrib.layers.sparse_column_with_keys(column_name="war", keys=['Y', 'N'])
	western = tf.contrib.layers.sparse_column_with_keys(column_name="western", keys=['Y', 'N'])
	occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="occupation", hash_bucket_size=1000)

	# Continuous Columns
	age = tf.contrib.layers.real_valued_column("age")
	time_diff = tf.contrib.layers.real_valued_column("time_diff")

	# Transformations
	age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

	# Wide Columns and Deep Columns
	wide_columns = [gender, occupation, age_buckets, unknown, action, adventure, animation, children, comedy, crime, documentary,
		drama, fantasy, filmnoir, horror, musical, mystery, romance, scifi, thriller, war, western,
		tf.contrib.layers.crossed_column([gender, occupation], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, action], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, animation], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, children], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, crime], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, drama], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, fantasy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, horror], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, musical], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, mystery], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, romance], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, scifi], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([gender, thriller], hash_bucket_size=int(1e4)),

		tf.contrib.layers.crossed_column([age_buckets, gender], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, action], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, animation], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, children], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, crime], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, drama], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, fantasy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, horror], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, musical], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, mystery], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, romance], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, scifi], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, thriller], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, gender, romance], hash_bucket_size=int(1e4)),

		tf.contrib.layers.crossed_column([age_buckets, action, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, animation, children], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, animation, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, children, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, animation, children, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, crime, action, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, drama, mystery], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, scifi, thriller], hash_bucket_size=int(1e4)),

		tf.contrib.layers.crossed_column([action, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([animation, children], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([animation, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([children, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([animation, children, comedy], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([crime, action, adventure], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([drama, mystery], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([scifi, thriller], hash_bucket_size=int(1e4))
		]

	deep_columns = [
		tf.contrib.layers.embedding_column(gender, dimension=8),
		tf.contrib.layers.embedding_column(occupation, dimension=8),
		tf.contrib.layers.embedding_column(unknown, dimension=8),
		tf.contrib.layers.embedding_column(action, dimension=8),
		tf.contrib.layers.embedding_column(adventure, dimension=8),
		tf.contrib.layers.embedding_column(animation, dimension=8),
		tf.contrib.layers.embedding_column(children, dimension=8),
		tf.contrib.layers.embedding_column(comedy, dimension=8),
		tf.contrib.layers.embedding_column(crime, dimension=8),
		tf.contrib.layers.embedding_column(documentary, dimension=8),
		tf.contrib.layers.embedding_column(drama, dimension=8),
		tf.contrib.layers.embedding_column(fantasy, dimension=8),
		tf.contrib.layers.embedding_column(filmnoir, dimension=8),
		tf.contrib.layers.embedding_column(horror, dimension=8),
		tf.contrib.layers.embedding_column(musical, dimension=8),
		tf.contrib.layers.embedding_column(mystery, dimension=8),
		tf.contrib.layers.embedding_column(romance, dimension=8),
		tf.contrib.layers.embedding_column(scifi, dimension=8),
		tf.contrib.layers.embedding_column(thriller, dimension=8),
		tf.contrib.layers.embedding_column(war, dimension=8),
		tf.contrib.layers.embedding_column(western, dimension=8),
		time_diff,
		age]

	# # Optimizers
	# linear_optimizer = tf.train.FtrlOptimizer(learning_rate=0.1,
	# 	l1_regularization_strength=0.01, l2_regularization_strength=0.01)

	# dnn_optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
	# 	l1_regularization_strength=0.001, l2_regularization_strength=0.001)

	if FLAGS.model_type == "wide":
		m = tflearn.LinearRegressor(model_dir=model_dir, feature_columns=wide_columns)
		# m = tflearn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
	elif FLAGS.model_type == "deep":
		m = tflearn.DNNRegressor(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])
		# m = tflearn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])
	elif FLAGS.model_type == "logistic":
		m = tflearn.LogisticRegressor()
	else:
		m = tflearn.DNNLinearCombinedRegressor(model_dir=model_dir,
			linear_feature_columns=wide_columns,
			dnn_feature_columns=deep_columns, dnn_hidden_units=[64, 32, 16])
		# m = tflearn.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns, 
		# 	dnn_feature_columns=deep_columns, dnn_hidden_units=[100, 50])

	return m

def input_fn(df):
	"""Input builder function"""
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a tf.SparseTensor
	continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

	# Creates a dictionary mapping from each categorical feature column name (k)
	# to the values of that column stored in a tf.SparseTensor
	categorical_cols = {
		k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)], values=df[k].values, shape=[df[k].size, 1])
		for k in CATEGORICAL_COLUMNS
	}

	# Merges the two dictionaries into one
	feature_cols = dict(continuous_cols)
	feature_cols.update(categorical_cols)
	# Converts the label column into a constant Tensor
	label = tf.constant(df[LABEL_COLUMN].values)

	# Returns the feature columns and the label
	return feature_cols, label

def train_and_eval():
	"""Train and evaluate the model"""
	train_data, test_data = merge_data()
	
	# train_data = train_data.dropna(how='any', axis=0)
	# test_data = test_data.dropna(how='any', axis=0)

	train_data[LABEL_COLUMN] = (train_data["rating"].astype(float))
	test_data[LABEL_COLUMN] = (test_data["rating"].astype(float))

	model_dir = "/home/rohan/Documents/MTPS/Rohan/MyCode/test_model"
	print("model directory = %s" % model_dir)

	test_rmse = np.zeros(FLAGS.train_steps)

	m = build_estimator(model_dir)
	for i in range(FLAGS.train_steps):
		m.fit(input_fn=lambda: input_fn(train_data), steps=1)
		# results = m.evaluate(input_fn=lambda: input_fn(test_data), steps=1)
		# for key in sorted(results):
		# 	print("%s : %s" % (key, results[key]))
		results = m.predict(input_fn=lambda: input_fn(test_data))
		results = min_max_normalization(results, 0, 1)
		test_rmse[i] = RMSE(results, test_data[LABEL_COLUMN])

	outFile = open("rmse_with_time.csv","wb")
	for i in range(0, len(test_rmse)):
		outFile.write(str(test_rmse[i]))
	outFile.close()


def main(_):
	start = time.time()
	train_and_eval()
	end = time.time()
	print("Time of Execution %s" % (end - start))

if __name__ == "__main__":
	tf.app.run()
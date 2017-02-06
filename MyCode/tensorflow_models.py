from __future__ import absolute_import, division, print_function


import sys
import argparse
import tempfile


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.metrics import streaming_mean_absolute_error, streaming_mean_squared_error


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")


COLUMNS = ['user_id', 'age', 'sex', 'occupation', 'zip_code', 'rating', 'timestamp', 'movie_id', 'unknown', 'action', 'adventure', 
	'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']

LABEL_COLUMN = "rating"

CATEGORICAL_COLUMNS = ['sex', 'occupation', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
	'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']

CONTINUOUS_COLUMNS = ['age', 'timestamp']



def RMSE(y_pred, y_test):
	return tf.sqrt(streaming_mean_squared_error(y_pred, y_test))

def MAE(y_pred, y_test):
	return streaming_mean_absolute_error(y_pred, y_test)

def load_users():
	u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
	users = pd.read_csv('../ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1', engine="python")
	print("USERS LOADED")
	return users

def load_ratings(filename):
	r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
	ratings = pd.read_csv('../ml-100k/'+filename, sep='\t', names=r_cols, encoding='latin-1', engine="python")
	print("RATINGS LOADED")
	return ratings

def load_movies():
	m_cols = ['movie_id', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
	'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
	'romance', 'scifi', 'thriller', 'war', 'western']
	cols = range(5, 24)
	cols.insert(0, 0)
	movies = pd.read_csv('../ml-100k/u.item', sep='|', names=m_cols, usecols=cols, encoding='latin-1', engine="python")
	print("MOVIES LOADED")
	return movies

def merge_data():
	users = load_users()
	movies = load_movies()
	train_ratings = load_ratings('u1.base')
	train_movie_ratings = pd.merge(movies, train_ratings)
	train_data = pd.merge(train_movie_ratings, users)

	test_ratings = load_ratings('u1.test')
	test_movie_ratings = pd.merge(movies, test_ratings)
	test_data = pd.merge(test_movie_ratings, users)
	print("DATA LOADED")
	return train_data, test_data


def build_estimator(model_dir):
	"""Build an Estimator"""
	
	# Sparse base Columns
	gender = tf.contrib.layers.sparse_column_with_keys(column_name="sex", keys=["F", "M"])
	unknown = tf.contrib.layers.sparse_column_with_keys(column_name="unknown", keys=[0, 1])
	action = tf.contrib.layers.sparse_column_with_keys(column_name="action", keys=[0, 1])
	adventure = tf.contrib.layers.sparse_column_with_keys(column_name="adventure", keys=[0, 1])
	animation = tf.contrib.layers.sparse_column_with_keys(column_name="animation", keys=[0, 1])
	children = tf.contrib.layers.sparse_column_with_keys(column_name="children", keys=[0, 1])
	comedy = tf.contrib.layers.sparse_column_with_keys(column_name="comedy", keys=[0, 1])
	crime = tf.contrib.layers.sparse_column_with_keys(column_name="crime", keys=[0, 1])
	documentary = tf.contrib.layers.sparse_column_with_keys(column_name="documentary", keys=[0, 1])
	drama = tf.contrib.layers.sparse_column_with_keys(column_name="drama", keys=[0, 1])
	fantasy = tf.contrib.layers.sparse_column_with_keys(column_name="fantasy", keys=[0, 1])
	filmnoir = tf.contrib.layers.sparse_column_with_keys(column_name="filmnoir", keys=[0, 1])
	horror = tf.contrib.layers.sparse_column_with_keys(column_name="horror", keys=[0, 1])
	musical = tf.contrib.layers.sparse_column_with_keys(column_name="musical", keys=[0, 1])
	mystery = tf.contrib.layers.sparse_column_with_keys(column_name="mystery", keys=[0, 1])
	romance = tf.contrib.layers.sparse_column_with_keys(column_name="romance", keys=[0, 1])
	scifi = tf.contrib.layers.sparse_column_with_keys(column_name="scifi", keys=[0, 1])
	thriller = tf.contrib.layers.sparse_column_with_keys(column_name="thriller", keys=[0, 1])
	war = tf.contrib.layers.sparse_column_with_keys(column_name="war", keys=[0, 1])
	western = tf.contrib.layers.sparse_column_with_keys(column_name="western", keys=[0, 1])
	occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="occupation", hash_bucket_size=1000)

	# Continuous Columns
	age = tf.contrib.layers.real_valued_column("age")
	timestamp = tf.contrib.layers.real_valued_column("timestamp")
	# unknown = tf.contrib.layers.real_valued_column(column_name="unknown")
	# action = tf.contrib.layers.real_valued_column(column_name="action")
	# adventure = tf.contrib.layers.real_valued_column(column_name="adventure")
	# animation = tf.contrib.layers.real_valued_column(column_name="animation")
	# children = tf.contrib.layers.real_valued_column(column_name="children")
	# comedy = tf.contrib.layers.real_valued_column(column_name="comedy")
	# crime = tf.contrib.layers.real_valued_column(column_name="crime")
	# documentary = tf.contrib.layers.real_valued_column(column_name="documentary")
	# drama = tf.contrib.layers.real_valued_column(column_name="drama")
	# fantasy = tf.contrib.layers.real_valued_column(column_name="fantasy")
	# filmnoir = tf.contrib.layers.real_valued_column(column_name="filmnoir")
	# horror = tf.contrib.layers.real_valued_column(column_name="horror")
	# musical = tf.contrib.layers.real_valued_column(column_name="musical")
	# mystery = tf.contrib.layers.real_valued_column(column_name="mystery")
	# romance = tf.contrib.layers.real_valued_column(column_name="romance")
	# scifi = tf.contrib.layers.real_valued_column(column_name="scifi")
	# thriller = tf.contrib.layers.real_valued_column(column_name="thriller")
	# war = tf.contrib.layers.real_valued_column(column_name="war")
	# western = tf.contrib.layers.real_valued_column(column_name="western")

	# Transformations
	age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

	# Wide Columns and Deep Columns
	wide_columns = [gender, occupation, age_buckets,
					tf.contrib.layers.crossed_column([gender, occupation], hash_bucket_size=int(1e4))]

	deep_columns = [
		tf.contrib.layers.embedding_column(gender, dimension=8),
		tf.contrib.layers.embedding_column(occupation, dimension=8),
		timestamp, age, unknown, action, adventure, animation, children, comedy, crime, documentary,
		drama, fantasy, filmnoir, horror, musical, mystery, romance, scifi, thriller, war, western]

	if FLAGS.model_type == "wide":
		m = tflearn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
	elif FLAGS.model_type == "deep":
		m = tflearn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])
	else:
		m = tflearn.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns, 
			dnn_feature_columns=deep_columns, dnn_hidden_units=[100, 50])

	return m

def input_fn(df):
	"""Input builder function"""
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a tf.SparseTensor
	continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
	for k in CONTINUOUS_COLUMNS:
		print(df[k].unique())

	# Creates a dictionary mapping from each categorical feature column name (k)
	# to the values of that column stored in a tf.SparseTensor
	categorical_cols = {
		k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)], values=str(df[k].values), shape=[df[k].size, 1])
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
	
	train_data = train_data.dropna(how='any', axis=0)
	test_data = test_data.dropna(how='any', axis=0)
	# print(train_data.ix[0])

	train_data[LABEL_COLUMN] = (train_data["rating"].astype(int))
	test_data[LABEL_COLUMN] = (test_data["rating"].astype(int))

	model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
	print("model directory = %s" % model_dir)

	m = build_estimator(model_dir)
	m.fit(input_fn=lambda: input_fn(train_data), steps=FLAGS.train_steps)
	results = m.evaluate(input_fn=lambda: input_fn(test_data), steps=1)
	for key in sorted(results):
		print("%S : %s" % (key, results[key]))

def main(_):
	train_and_eval()

if __name__ == "__main__":
	tf.app.run()
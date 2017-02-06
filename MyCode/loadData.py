import numpy as np

def computeAgeCategory(age):
	boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
	i = 0
	for val in boundaries:
		if age <= val:
			return i
		else:
			i += 1
	return i


def loadMovieLens(path='../ml-100k/'):
	# load movie data
	movieTitle = {}
	movieGenre = {}
	for line in open(path + 'u.item'):
		tupleData = line.split('|')
		movieId = int(tupleData[0])
		movieTitle[movieId] = tupleData[1]
		movieGenre[movieId] = list(map(int, tupleData[5:]))

	# load occupations data
	occupations = {}
	i = 0
	for line in open(path + 'u.occupation'):
		key = line.strip('\n')
		occupations[key] = i
		i += 1

	# load user data
	users = {}
	for line in open(path + 'u.user'):
		tupleData = line.split('|')
		userId = int(tupleData[0])
		age = int(tupleData[1])
		gender = tupleData[2]
		if gender == 'M':
			gender = 0
		else:
			gender = 1
		occupation = list(map(int, np.zeros(len(occupations))))
		occupation[occupations[tupleData[3]]] = 1
		users[userId] = [age, gender] + list(occupation)
	
	# load training data
	i = 0
	train_x = []
	train_y = []
	for line in open(path + 'u2.base'):
		userId, movieId, rating, ts = line.split('\t')
		userId = int(userId)
		movieId = int(movieId)
		rating = int(rating)
		train_x.append(users[userId] + movieGenre[movieId])
		train_y.append(rating)
		i += 1


	# load testing data
	i = 0
	test_x = []
	test_y = []
	for line in open(path + 'u2.test'):
		userId, movieId, rating, ts = line.split('\t')
		userId = int(userId)
		movieId = int(movieId)
		rating = int(rating)
		test_x.append(users[userId] + movieGenre[movieId])
		test_y.append(rating)
		i += 1
	return train_x, train_y, test_x, test_y
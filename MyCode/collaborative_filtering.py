from math import sqrt


def compute_rating(y_pred):
	for i in range(len(y_pred)):
		y_pred[i] = list(y_pred[i])
		intPart, fracPart = divmod(y_pred[i][0], 1)
		if fracPart >= 0.5:
			y_pred[i][0] = intPart + 1
		else:
			y_pred[i][0] = intPart
	return y_pred



# load data
def loadMovieLens(path='../ml-100k/'):
	# load movie data
	movieTitle = {}
	movieGenre = {}
	for line in open(path + 'u.item'):
		tupleData = line.split('|')
		movieId = int(tupleData[0])
		movieTitle[movieId] = movieId
		movieGenre[movieId] = map(int, tupleData[5:])

	# load training data
	trainUsers = {}
	for line in open(path + 'u1.base'):
		(user, movieId, rating, ts) = line.split('\t')
		user = int(user)
		movieId = int(movieId)
		trainUsers.setdefault(user, {})
		trainUsers[user][movieTitle[movieId]] = float(rating)

	# load test data
	testUsers = {}
	for line in open(path + 'u1.test'):
		(user, movieId, rating, ts) = line.split('\t')
		user = int(user)
		movieId = int(movieId)
		testUsers.setdefault(user, {})
		testUsers[user][movieTitle[movieId]] = float(rating)
	return trainUsers, testUsers




# Euclidean Distance Metric
def sim_distance(prefs, person1, person2):
	#Get the list of shared items
	si = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			si[item] = 1

	#if they have not ratings in common
	if len(si) == 0:
		return 0

	#Add up the squares of all the differences
	sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2)
						  for item in prefs[person1] if item in prefs[person2]])
	return 1/(1+sum_of_squares)


# Pearson Correlation Coefficient
def sim_pearson(prefs, person1, person2):
	# Get the list of mutually rated items
	si = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			si[item] = 1

	# Find number of elements
	n = len(si)

	# if there are no ratings in common
	if n == 0:
		return 0

	# Add up all preferences
	sum1 = sum([prefs[person1][item] for item in si])
	sum2 = sum([prefs[person2][item] for item in si])

	# Sum up the squares
	sum1Sq = sum([pow(prefs[person1][item], 2) for item in si])
	sum2Sq = sum([pow(prefs[person2][item], 2) for item in si])

	# Sum up the products
	pSum = sum([prefs[person1][item]*prefs[person2][item] for item in si])

	# Calculate the Pearson score
	num = pSum - (sum1*sum2/n)
	den = sqrt((sum1Sq - pow(sum1, 2)/n) * (sum2Sq - pow(sum2, 2)/n))
	
	if den == 0:
		return 0

	r = num/den
	return r



# Returns the best matches for person from the prefs dict
# Number of results and similarity function are optional params
def topMatches(prefs, person, n=5, similarity=sim_pearson):
	scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
	# Sort the list so the highest scores appear at the top
	scores.sort()
	scores.reverse()
	return scores[0:n]


# User CF
# Get recommendations for a person by using a weighted average of every user's ratings
def getRecommendations(prefs, person, similarity=sim_pearson):
	totals = {}
	simSums = {}
	for other in prefs:
		# don't comapare me to myself
		if other == person:
			continue

		sim = similarity(prefs, person, other)

		# ignore scores of zero or lower
		if sim <= 0:
			continue

		for item in prefs[other]:
			# only score movies I haven't seen yet
			if item not in prefs[person] or prefs[person][item] == 0:
				# Similarity * Score
				totals.setdefault(item, 0)
				totals[item] += prefs[other][item] * sim
				# Sum of similarities
				simSums.setdefault(item, 0)
				simSums[item] += sim

	# Create the nomalized list
	rankings = [(round(total/simSums[item], 4), item) for item,total in totals.items()]

	# Return the sorted list
	rankings.sort()
	rankings.reverse()
	return rankings


# Transform user-item rating matrix to item-user rating matrix
def transformPrefs(prefs):
	result = {}
	for person in prefs:
		for item in prefs[person]:
			result.setdefault(item, {})
			# Flip item and person
			result[item][person] = prefs[person][item]
	return result


def calculateSimilarItems(prefs, n=10):
	# Create a dict of items showing which other items they are most similar to
	result = {}

	# Invert the preference matrix to be item-centric
	itemPrefs = transformPrefs(prefs)
	c = 0
	for item in itemPrefs:
		# Status updates for large datasets
		c += 1
			# Find the most similar items to this one
		scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
		result[item] = scores
	return result

# Item CF
def getRecommendedItems(prefs, itemMatch, user):
	userRatings = prefs[user]
	scores = {}
	totalSim = {}
	# Loop over items rated by this user
	for (item, rating) in userRatings.items():
		# Loop over items similar to this one
		for (similarity, item2) in itemMatch[item]:
			# Ignore if this user has already rated this item
			if item2 in userRatings:
				continue

			# Weighted sum of rating times similarity
			scores.setdefault(item2, 0)
			scores[item2] += similarity*rating

			# Sum of all similarities
			totalSim.setdefault(item2, 0)
			totalSim[item2] += similarity
	# Divide each total score by toal weighting to get an average
	rankings = [(score/totalSim[item], item) for item, score in scores.items()]

	# Return the rankings from highest to lowest
	rankings.sort()
	rankings.reverse()
	return rankings


def computeError(y_pred, y_test):
	mae = 0.0
	rmse = 0.0
	maeItr = 0.0
	i = 0
	
	for movieId in y_test:
		for rat, mId  in y_pred:
			if movieId == mId:
				maeItr = abs(y_test[movieId] - rat)
				mae += maeItr
				rmse += maeItr*maeItr
				i += 1
	if i != 0:
		mae = mae/i
		rmse = (sqrt(rmse/i))
	else:
		mae = 0
		rmse = 0
	return (mae, rmse)



trainData, testData = loadMovieLens()



# for USER CF
totalMAE = 0.0
totalRMSE = 0.0
maeIter = 0.0
rmseIter = 0.0
for i in testData:
	recommendations = getRecommendations(trainData, i)
	recommendations = compute_rating(recommendations)
	maeIter, rmseIter = computeError(recommendations, testData[i])
	totalMAE += maeIter
	totalRMSE += rmseIter
print (totalMAE/len(testData))
print(totalRMSE/len(testData))



# for ITEM CF
itemSim = calculateSimilarItems(trainData)

totalMAE = 0.0
totalRMSE = 0.0
maeIter = 0.0
rmseIter = 0.0
itr = 0
for i in testData:
	maeIter = 0.0
	rmseIter = 0.0
	recommendations = getRecommendedItems(trainData, itemSim, i)
	recommendations = compute_rating(recommendations)
	maeIter, rmseIter = computeError(recommendations, testData[i])
	if maeIter != 0:
		totalMAE += maeIter
		totalRMSE += rmseIter
		itr +=1
print (totalMAE/itr)
print(totalRMSE/itr)

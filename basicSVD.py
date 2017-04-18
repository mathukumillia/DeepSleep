import numpy as np

# these are all the global variables used by the program

# users and movies are one indexed
numUsers = 458293
numMovies = 17770

# K is the constant representing the number of features
# lrate is the learning rate
K = 5
lrate = 0.001

# user and movie feature matrices
userValues = np.zeros([numUsers + 1, K]) + 0.1
movieValues = np.zeros([numMovies + 1, K]) + 0.1

# cache of the residuals we have
NUM_TRAIN_POINTS = 100000000
residuals = [0] * NUM_TRAIN_POINTS

# given a single movie and user index, along with the correct ratingsData
# pointNum is the index of the point in the training set
# trains the feature matrices
def train (movieID, userID, rating, pointNum):
	global userValues
	global movieValues
	global residuals

	for feature in range(K):
		# calculate the error with the current feature values
		err = lrate * (rating - predictRating(movieID, userID, pointNum))

		# train this feature with the calculated error
		mv = movieValues[movieID][feature]
		uv = userValues[userID][feature]
		userValues[userID][feature] += err * mv
		movieValues[movieID][feature] += err * uv

		# update the residual for this point with the previous change
		newVal = (userValues[userID][feature] * movieValues[movieID][feature])
		residuals[pointNum] += newVal - (uv * mv)

# predicts the rating given a particular user and movie
def predictRating (movieID, userID, pointNum):
	global residuals
	residualValue = residuals[pointNum]
	# only compute the residual if its not cached
	if residualValue == 0:
		residualValue = np.dot(movieValues[movieID], userValues[userID])
		residuals[pointNum] = residualValue
	return residualValue

def SVD ():
	# open the indices file
	# change the file paths to reflect your directory set up
	indexFile = open('../um/all.idx')
	# go through the training data and train on each point
	with open('../um/all.dta') as srcFile:
		pointNum = 0
		for line in srcFile:
			# find out if data is from training set
			index = int(indexFile.readline())
			# only train on training data
			if index != 4 and index != 5:
				print pointNum
				point = line.split()
				userID = int(point[0])
				movieID = int(point[1])
				date = int(point[2])
				rating = int(point[3])
				train(movieID, userID, rating, pointNum)
				pointNum += 1

	print(userValues)
	print(movieValues)

SVD()

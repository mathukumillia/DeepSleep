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

# given a single movie and user index, along with the correct ratingsData
# pointNum is the index of the point in the training set
# trains the feature matrices
def train (movieID, userID, rating, pointNum):
	global userValues
	global movieValues

	# calculate the error with the current feature values
	err = lrate * (rating - predictRating(movieID, userID, pointNum))

	# train vectors
	uv = userValues[userID]
	userValues[userID] += err * movieValues[movieID]
	movieValues[movieID] += err * uv

# predicts the rating given a particular user and movie
def predictRating (movieID, userID, pointNum):
	return np.dot(movieValues[movieID], userValues[userID])

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

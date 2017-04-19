import numpy as np

# users and movies are one indexed 
numUsers = 458293
numMovies = 17770

def load_data():
	
	ratingsData = []
	return ratingsData

def SVD():
	# K is the number of features/aspects
	K = 5
	lrate = 0.001

	userValues = np.add(np.zeros((numUsers + 1, K)), 0.1)
	movieValues = np.add(np.zeros((numMovies + 1, K)), 0.1)

	indexFile = open('all.idx')

	with open('all.dta') as srcFile:
		ratingsData = []
		lineNum = 0
		err = 0

		for line in srcFile:
			# find out if data is from training set
			index = int(indexFile.readline())

			# only train on training data
			if index != 4 and index != 5:
				lineNum += 1
				ratingsData.append(line.split())

				if lineNum % 100 == 0:
					for point in ratingsData:
						userID = int(point[0])
						movieID = int(point[1])
						date = int(point[2])
						rating = int(point[3])

						err = rating - np.multiply(userValues[userID], movieValues[movieID])

						uv = userValues[userID]
						userValues[userID] += lrate * err * movieValues[movieID]
						movieValues[movieID] += lrate * err * uv

					ratingsData = []


	print(userValues)
	print(movieValues)
	print(lineNum)
	print(err)


SVD()

		

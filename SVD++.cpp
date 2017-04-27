#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0.0001

using namespace std;

// these are all the global variables used by the program

// users and movies are one indexed
const double num_users = 458293;
const double num_movies = 17770;
const double num_pts = 102416306;

// K is the constant representing the number of features
// lrate is the learning rate
const double K = 5;
const double lrate = 0.001;

// though these are declared as single dimensional, I will use them as 2D arrays
// to facilitate this, I will store the sizes of the arrays as well
// we add one to the num_users because these arrays are 1-indexed
double *user_values; // this is p in the SVD++ paper
int user_values_size = (int)((num_users + 1) * K);
double *movie_values; // this is q in the SVD++ paper
int movie_values_size = (int)((num_movies + 1) * K);

// the arrays to store the ratings and indices data
// note that ratings will be used as a 2D array as well
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
double *indices;

// stores each user's neighborhoods
// functionally, this is a 2D array that stores for each user the id of the
// movies they provided feedback for
double *neighborhoods;

// y is a 2D array that holds K features for each of the movies
// the plus one in the size derives from the fact that the movies are 1 indexed
double *y;
int y_size = (int) ((num_movies + 1) * K)

/*
* Allocates memory and initializes user, movie, ratings, and indices arrays
*/
void initialize()
{
	cout << "Initializing the program.\n";

	// allocate and initialize the user_values and movie_values matrices
	// all of the +1 terms result from the fact that these arrays are 1
	// indexed in the data
	user_values = new double[user_values_size];
	for (int i = 0; i < user_values_size; i++)
	{
		user_values[i] = 0.1;
	}

	movie_values = new double[movie_values_size];
	for (int i = 0; i < movie_values_size; i++)
	{
		movie_values[i] = 0.1;
	}

	// create  the arrays that store the ratings input data and the indexes
	ratings = new double[ratings_size];
	indices = new double[((int) num_pts)];

	cout << "Done allocating memory.\n";
}

/*
* clears all used memory
*/
void clean_up()
{
	cout << "Cleaning up.\n";
	delete [] user_values;
	delete [] movie_values;
	delete [] ratings;
	delete [] indices;
}

/*
* Reads the input data into ratings and indices
*/
void read_data()
{
	cout << "Reading in training data.\n";
	// read in ratings data
	fstream ratings_file("../ratings.bin", ios::in | ios::binary);
	ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * num_pts * POINT_SIZE);
	ratings_file.close();

	// read in index data
	fstream indices_file("../indices.bin", ios::in | ios::binary);
	indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * num_pts);
	indices_file.close();
}

/*
* Predicts a rating given a user and a movie
*/
double predict_rating(double user, double movie)
{
	/* TODO */
	return 0.0;
}

/*
* Gets the total error of the SVD++ model on the set index provided
* i.e., to get validation error, pass in set = 2
*/
double error(int set)
{
	cout << "Calculating error.\n";

	double error = 0;
	double diff = 0;
	double index;
	double points_in_set = 0;

	for (int i = 0; i < num_pts; i++) {
		index = indices[i];

		if (index == set) {
			diff = ratings[i * POINT_SIZE + 3] - predict_rating(ratings[i + POINT_SIZE], ratings[i * POINT_SIZE + 1]);
			error += diff * diff;
			points_in_set += 1;
		}
	}

	return sqrt(error/points_in_set);
}

/*
* Trains the SVD++ model on one provided point
* Point must contain the user, movie, date, and rating
* utilizes SGD
*/
void train(double user, double movie, double date, double rating)
{
	/* TODO */
}

/*
* Iterates through every point in the training set and trains on each one
*/
void run_epoch()
{
	cout << "Running an epoch.\n";
	int index;
	for (int i = 0; i < num_pts; i++) {
		index = indices[i];
		// trains only on point set one; change this line if you want to train
		// on additional points
		if (index == 1) {
			train(ratings[i * POINT_SIZE], ratings[i * POINT_SIZE + 1], ratings[i * POINT_SIZE + 2], ratings[i * POINT_SIZE + 3]);
		}
	}
}

int main()
{
	initialize();
	read_data();

	double initialError = 10;
	double finalError = error(2); // gets the initial validation error
	int counter = 0;

	cout << "Initial Error is: " << initialError << "\n";
	while (initialError - finalError > STOPPING_CONDITION) {
		cout << "Starting Epoch " << counter << "\n";
		counter++;
		initialError = finalError;
		run_epoch();
		finalError = error(2); // error(2) returns the validation error
		cout << "Error after Epoch " << finalError << "\n";
	}

	clean_up();
	return 0;
}


#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include "baselinePrediction.h"
using namespace std;

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 50

// these are all the global variables used by the program

// users and movies are one indexed
const int num_users = 458293;
const int num_movies = 17770;
const int num_pts = 102416306;
const double lambda = 0.025;

// K is the constant representing the number of features
// lrate is the learning rate
const int K = 50;
const double lrate = 0.005;

// these 2D arrays are the U and V in the SVD
// they are one dimensional arrays in memory to make access quicker
double *user_values;
int user_values_size = (num_users + 1) * K;
double *movie_values;
int movie_values_size = (num_movies + 1) * K;

// 2D arrays to store ratings input
// again, it's a one dimensional array in memory to make access quicker
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
// stores the indices of each point, in the same order as the ratings
double *indices;

/*
* Allocates memory and initializes user, movie, ratings, and indices arrays
*/
inline void initialize()
{
    cout << "Initializing the program.\n";

    // allocate and initialize the user_values and movie_values matrices
    // all of the +1 terms result from the fact that these arrays are 1
    // indexed in the data
    user_values = new double[user_values_size];
    for (int i = 0; i < user_values_size; i++)
    {
        user_values[i] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
    }

    movie_values = new double[movie_values_size];
    for (int i = 0; i < movie_values_size; i++)
    {
        movie_values[i] = 0.1 * (double)(rand() % 10) + 0.01;
    }

    // create  the arrays that store the ratings input data and the indexes
    ratings = new double[ratings_size];
    indices = new double[num_pts];


    cout << "Done allocating memory.\n";
}

/*
* Clears all used memory
*/
inline void clean_up()
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
inline void read_data()
{
    cout << "Reading in training data.\n";

    // read in ratings data - currently, this is training without the baseline
    fstream ratings_file("../ratings_baseline_removed.bin", ios::in | ios::binary);
    ratings_file.read((char *)(ratings), sizeof(double) * num_pts * POINT_SIZE);
    ratings_file.close();

    // read in index data
    fstream indices_file("../indices.bin", ios::in | ios::binary);
    indices_file.read((char *)(indices), sizeof(double) * num_pts);
    indices_file.close();
}

/*
* Given a user and a movie, this function gives the predicted rating
*/
inline double predict_rating(int user, int movie)
{
  double rating = 0;
  for (int i = 0; i < K; i++)
  {
    rating += user_values[user * K + i] * movie_values[movie * K + i];
  }
  return rating;
}

/*
* Gets the total error of the SVD model on the set index provided
* i.e., to get validation error, pass in set = 2
*/
inline double error(int set)
{
    cout << "Calculating error.\n";

    double error = 0;
    double diff = 0;
    double index;
    double points_in_set = 0;

    for (int i = 0; i < num_pts; i++) {
        index = indices[i];
        if (index == set) {
            diff = (ratings[i * POINT_SIZE + 3]) - 
            	predict_rating((int)ratings[i * POINT_SIZE], (int)ratings[i * POINT_SIZE + 1]);
            error += diff * diff;
            points_in_set += 1;
        }
    }

    return sqrt(error/points_in_set);
}

/*
* Runs SGD on a single point
*/
inline void train(int user, int movie, double rating)
{

  	// calculate the error with the current feature values
	double err = rating - predict_rating(user, movie);

	// updates the movie and user vectors feature by feature
	double uv;
	for (int i = 0; i < K; i++){
		uv = user_values[user * K + i];
		user_values[user * K + i] +=  lrate * (err * movie_values[movie * K + i] - lambda * uv);
		movie_values[movie * K + i] += lrate * (err * uv - lambda * movie_values[movie * K + i]);
	}
}

/*
* Run a full epoch.
*/
inline void run_epoch ()
{
	cout << "Running Epoch." << "\n";

    int pt;
    double index;
	for (int i = 0; i < num_pts; i++) {

        // select an arbitrary point to train with 
        pt = rand() % num_pts;
		index = indices[pt];
        // make sure the selected point is in the first data set
        while (index != 1)
        {
            pt = rand() % num_pts;
            index = indices[pt];
        }
        // train with this point
		train((int)ratings[pt * POINT_SIZE], (int)ratings[pt * POINT_SIZE + 1], ratings[pt * POINT_SIZE + 3]);
	}
	cout << "Epoch complete." << "\n";
}

/*
* Predicts ratings on the qual set and writes them to a file.
*/
inline void find_qual_predictions()
{
	ofstream outputFile;
	outputFile.open("naive_SVD_output.dta");

	double index;
	double prediction;

	int user;
	int movie;
	int date;
	for(int i = 0; i < num_pts; i++)
	{
	    index = indices[i];
	    // the qual set is set 5
	    if (index == 5)
	    {
	    	user = (int)ratings[i * POINT_SIZE];
	    	movie = (int)ratings[i * POINT_SIZE + 1];
	    	date = (int)ratings[i * POINT_SIZE + 2];
	        prediction = baselinePrediction(user, movie, date) + predict_rating(user, movie);
	        if (prediction < 1)
	        {
	            prediction = 1;
	        }
	        if (prediction > 5)
	        {
	            prediction = 5;
	        }
	        outputFile << prediction << "\n";
	    }
	}
}

int main()
{
    initialize();
    read_data();

    double initialError = 10000;
    double finalError = error(2); // gets the validation error before training
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";
    while (initialError - finalError > STOPPING_CONDITION && counter <= MAX_EPOCHS) {
        cout << "Starting Epoch " << counter << "\n";
        initialError = finalError;
        run_epoch();
        finalError = error(2); // error(2) returns the validation error
        cout << "Error after Epoch " << counter << ": " << finalError << "\n";
        counter++;
        cout << "-----------------------------------\n";
    }

    // find the values on the qual set
    find_qual_predictions();

    clean_up();
    return 0;
}
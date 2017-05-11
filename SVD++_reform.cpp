#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include "baselinePrediction.h"

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 30 // the maximum number of epochs to run; 30 in the paper
#define MAX_NEIGHBOR_SIZE 300 // obtained from SVD++ paper
#define LAMBDA_7 0.015 // obtained from the SVD++ paper

using namespace std;

// these are all the global variables used by the program

// users and movies are one indexed
const double num_users = 458293;
const double num_movies = 17770;
const double num_pts = 102416306;

// K is the constant representing the number of features
// gamma_2 is the step size
const double K = 5;
double GAMMA_2 = 0.007;

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
// this is an int array because the double array is too large to work with
// movies they provided feedback for
int *neighborhoods;
int neighborhoods_size = (int) ((num_users + 1) * MAX_NEIGHBOR_SIZE);
double *neighborhood_sizes;

// y is a 2D array that holds K features for each of the movies
// the plus one in the size derives from the fact that the movies are 1 indexed
double *y;
int y_size = (int) ((num_movies + 1) * K);

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
        user_values[i] = 0.1; // arbitrary initial condition
    }

    movie_values = new double[movie_values_size];
    y = new double[y_size];
    for (int i = 0; i < movie_values_size; i++)
    {
        movie_values[i] = 0.1;
        y[i] = 0.1; // this was an arbitrary initial condition
    }

    // create  the arrays that store the ratings input data and the indexes
    ratings = new double[ratings_size];
    indices = new double[((int) num_pts)];

    neighborhoods = new int[neighborhoods_size];
    neighborhood_sizes = new double[(int)(num_users + 1)];


    cout << "Done allocating memory.\n";
}

/*
* clears all used memory
*/
inline void clean_up()
{
    cout << "Cleaning up.\n";
    delete [] user_values;
    delete [] movie_values;
    delete [] ratings;
    delete [] indices;
    delete [] neighborhoods;
    delete [] neighborhood_sizes;
    delete [] y;
}

/*
* Reads the input data into ratings and indices
*/
inline void read_data()
{
    cout << "Reading in training data.\n";
    // read in ratings data - currently, this is training without the baseline
    fstream ratings_file("../ratings_baseline_removed.bin", ios::in | ios::binary);
    ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * num_pts * POINT_SIZE);
    ratings_file.close();

    // read in index data
    fstream indices_file("../indices.bin", ios::in | ios::binary);
    indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * num_pts);
    indices_file.close();

    // read in neighborhod data
    fstream neighborhood_file("../neighborhoods.bin", ios::in | ios::binary);
    neighborhood_file.read(reinterpret_cast<char *>(neighborhoods), sizeof(int) * (num_users + 1) * MAX_NEIGHBOR_SIZE);
    neighborhood_file.close();

    // read in the neighborhood size data
    fstream nsize_file ("../neighborhood_sizes.bin", ios::in | ios::binary);
    nsize_file.read(reinterpret_cast<char *>(neighborhood_sizes), sizeof(double) * (num_users + 1));
    nsize_file.close();
}

/*
* Get the sum of the neighborhood vectors for the given user
* this is |N(u)|^(-1/2) * sum of y's in neighborhood of u
* @return: a double array of the vector sum; make sure you delete the memory
* allocated to this array when using this function
*/
inline double * get_y_sum(int user)
{
	// stores the sum of user vectors
    // (all the y's in the users neighborhood plus the user factors)
    // initializes to 0 because it accumulates the other values
    double * y_vector_sum = new double[(int)K]();
    // stores the movie for which we are obtaining y
    int neighborhood_movie;
    // loop through neighborhood and get sum of y vectors for each movie in 
    // neighborhood
    if(neighborhood_sizes[user] > 0)
    {
        for (int i = 0; i < neighborhood_sizes[user]; i++)
        {
            neighborhood_movie = neighborhoods[user * MAX_NEIGHBOR_SIZE + i];
            // add the current neighborhood movies y vector to the user vector sum
            for (int j = 0; j < K; j++)
            {
                y_vector_sum[j] += y[neighborhood_movie * (int)K + j];
            }
        }
        // loop through and divide each of the elements by the square root of 
        // the neighborhood size
		for (int i = 0; i < K; i++)
        {
            y_vector_sum[i] = y_vector_sum[i]/sqrt(neighborhood_sizes[user]);
        }
    }
    return y_vector_sum;
}

/*
* Predict the rating given a user and movie
*
* @return: the double rating
*/
inline double predict_rating(int user, int movie)
{
	// gets the sum of the neighborhood vectors 
	double * user_vector = get_y_sum(user);

	// add in the current user factors to the neighborhood sum
	for (int i = 0; i < K; i++)
	{
		user_vector[i] = user_values[user * (int)K + i] + user_vector[i];
	}

	// compute the rating as a fu
	double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector[i] * movie_values[movie * (int)K + i];
    }

    delete [] user_vector;
    return rating;
}

/*
* Predicts the rating given a user vector that has already been calculated
*/
inline double predict_rating(double * user_vector, int movie)
{
	// compute the rating as a fu
	double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector[i] * movie_values[movie * (int)K + i];
    }
    return rating;
}

/*
* Get the error on a given set of points.
* 	i.e. set 2 is the validation set
*
* @return: the double error
*/
inline double error (int set)
{
	cout << "Calculating error.\n";

    double error = 0;
    double diff = 0;
    double points_in_set = 0;
    int user; 
    int movie;
    double rating;

    for (int i = 0; i < num_pts; i++) {

        user = (int)ratings[i * POINT_SIZE];
        movie = (int)ratings[i * POINT_SIZE + 1];
        rating = ratings[i * POINT_SIZE + 3];

        if (indices[i] == set) {
            diff = rating - predict_rating(user, movie);
            error += diff * diff;
            points_in_set += 1;
        }
    }

    return sqrt(error/points_in_set);
}

/*
* function to train the SVD; takes in a user and a movie index, along with 
* a rating 
*
*/
inline void train(double user, double movie, double date, double rating)
{

}

/*
* Iterates through every point in the training set and trains on each one
*/
inline void run_epoch()
{
    cout << "Running an epoch.\n";
    double user;
    double movie;
    double date;
    double rating;
    
    for (int i = 0; i < num_pts; i++) {

        // trains only on point set one; change this line if you want to train
        // on additional points
        if (indices[i] == 1)
        {
            user = ratings[i * POINT_SIZE];
            movie = ratings[i * POINT_SIZE + 1];
            date = ratings[i * POINT_SIZE + 2];
            rating = ratings[i* POINT_SIZE + 3];
            train(user, movie, date, rating);
            if (i%1000000 == 0)
            {
                cout << "i: " << i << "\n";
            }
        }
    }
    // decrease gamma_2 by 10%, as suggested in paper
    GAMMA_2 = 0.9 * GAMMA_2;
}

/*
* get predictions on the qual set and output them to a file
* fix this to work with baseline removed predictions
*
*/
inline void find_qual_predictions()
{
    cout << "Finding and writing qual predictions.\n";
    ofstream outputFile;
    ofstream probeFile;
    outputFile.open("SVD++_output.dta");
    probeFile.open("SVD++_probe.dta");

    double prediction;
    int user;
    int movie;
    int date;
    for(int i = 0; i < num_pts; i++)
    {
        if (indices[i] == 5 || indices[i] == 4)
        {
            user = (int)ratings[i * POINT_SIZE];
            movie = (int)ratings[i * POINT_SIZE + 1];
            date = (int)ratings[i * POINT_SIZE + 2];
            // I have to add the ratings in the file because this ratings file
            // has the baselines removed
            prediction = 
                baselinePrediction(user, movie, date) + predict_rating(user, movie);
            if (prediction < 1)
            {
                prediction = 1;
            }
            if (prediction > 5)
            {
                prediction = 5;
            }
            if(indices[i] == 5)
            {
                outputFile << prediction << "\n";
            }
            else
            {  
                probeFile << prediction << "\n";
            }
        }
    }
    outputFile.close();
    probeFile.close();
}

int main()
{
    initialize();
    read_data();

    double initialError = 100000;
    double finalError = error(2); // gets the validation error before training
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";
    while (initialError - finalError > STOPPING_CONDITION && counter <= MAX_EPOCHS) {
        cout << "Starting Epoch " << counter << "\n";
        counter++;
        initialError = finalError;
        run_epoch();
        finalError = error(2); // error(2) returns the validation error
        cout << "Error after Epoch " << counter << ": " << finalError << "\n";
    }

    // find the values on the qual set
    find_qual_predictions();

    clean_up();
    return 0;
}
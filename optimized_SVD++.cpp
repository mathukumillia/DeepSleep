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
#define LAMBDA_6 0.005 // from paper
#define DECAY 0.975 // from paper

using namespace std;

// these are all the global variables used by the program

// users and movies are one indexed
const double num_users = 458293;
const double num_movies = 17770;
const double num_pts = 102416306;

// K is the constant representing the number of features
// gamma_2 is the step size
const double K = 75;
double GAMMA_2 = 0.007;
double GAMMA_1 = 0.007;
// the mean rating with the baselines removed in point set 1
const double baseline_removed_mean = 0.00199931;

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

// the user and movie bias arrays
double * user_biases;
double * movie_biases;

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
        user_values[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K); // arbitrary initial condition
    }

    movie_values = new double[movie_values_size];
    y = new double[y_size];
    for (int i = 0; i < movie_values_size; i++)
    {
        movie_values[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K);
        y[i] = 0.0; // this was an arbitrary initial condition
    }

    // create  the arrays that store the ratings input data and the indexes
    ratings = new double[ratings_size];
    indices = new double[((int) num_pts)];

    neighborhoods = new int[neighborhoods_size];
    neighborhood_sizes = new double[(int)(num_users + 1)];

    // initialize the user and movie biases
    user_biases = new double[(int)num_users + 1];
    for (int i = 0; i < num_users + 1; i++)
    {
        user_biases[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K);
    }

    movie_biases = new double[(int)num_movies + 1];
    for (int i = 0; i < num_movies + 1; i++)
    {
        movie_biases[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K); 
    }

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
    delete [] user_biases;
    delete [] movie_biases;
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
* takes in a pointer to an array of all 0s to which it can store the result
* allocated to this array when using this function
*/
inline void get_y_sum(int user, double * y_vector_sum)
{
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
}

/*
* Predict the rating given a user and movie
*
* @return: the double rating
*/
inline double predict_rating(int user, int movie)
{
	// gets the sum of the neighborhood vectors 
	double * user_vector = new double[(int)K]();
    get_y_sum(user, user_vector);

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

    // add in the user and movie biases
    rating += user_biases[user];
    rating += movie_biases[movie];

    // add in the mean
    rating += baseline_removed_mean;

    delete [] user_vector;
    return rating;
}

/*
* Predicts the rating given a user vector that has already been calculated
*/
inline double predict_rating(int user, int movie, double * y_sum)
{
    double user_vector[(int)K] = {};
    for (int i = 0; i < K; i++)
    {
        user_vector[i] = user_values[user * (int)K + i] + y_sum[i];
    }
	// compute the rating
	double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector[i] * movie_values[movie * (int)K + i];
    }
    // add in the user and movie biases
    rating += user_biases[user];
    rating += movie_biases[movie];

    // add in the mean
    rating += baseline_removed_mean;
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
* Trains the SVD++ model on one provided point
* Point must contain the user, movie, date, and rating
* also takes in y_sum, which just contains the sum of the y's in the 
* neighborhood 
* utilizes SGD
*/
inline void train()
{
    int userId, itemId, currentUser, i;
    double rating;
    int pt_index = 0;
    double * y_sum_im = new double[(int)K]();
    double * y_sum_old = new double[(int)K]();
    double user_vector[(int)K] = {};
    double point_error;
    double movie_factor;
    double user_factor;
    // neighborhood size
    double size; 

    // iterates through the users
    for (userId = 1; userId < num_users; userId++)
    {
        // get the neighborhood size for this user
        size = neighborhood_sizes[userId];
        // get the sum of neighborhood vectors for this user
        get_y_sum(userId, y_sum_im);
        // set y_sum_old equal to y_sum_im
        for (i = 0; i < K; i++)
        {
            y_sum_old[i] = y_sum_im[i];
        }

        // this goes through all the training samples associated with a user
        while (ratings[pt_index * POINT_SIZE] == userId)
        {
            // only train if this is in the training set 
            if (indices[pt_index] == 1)
            {
                itemId = (int)ratings[pt_index * POINT_SIZE + 1]; 
                rating = ratings[pt_index * POINT_SIZE + 3];
                // get the point error
                point_error = rating - predict_rating(userId, itemId, y_sum_im);
                for (i = 0; i < K; i++)
                {
                    // update the movie and user factors 
                    movie_factor = movie_values[itemId * (int)K + i];
                    user_factor = user_values[userId * (int)K + i];

                    movie_values[itemId * (int)K + i] += 
                        GAMMA_2 * (point_error * (user_factor + y_sum_im[i]) - LAMBDA_7 * movie_factor);
                    user_values[userId * (int)K + i] +=
                        GAMMA_2 * (point_error * movie_factor - LAMBDA_7 * user_factor);

                    // update y_sum_im
                    if(size != 0)
                    {
                        y_sum_im[i] += 
                            GAMMA_2 * (point_error/sqrt(size) * movie_factor - LAMBDA_7 * y_sum_im[i]);
                    }            
                }
                // update the user and movie biases
                user_biases[userId] += GAMMA_1 * (point_error - LAMBDA_6 * user_biases[userId]);
                movie_biases[itemId] += GAMMA_1 * (point_error - LAMBDA_6 * movie_biases[itemId]);
            }
            // update the pt_index to the next point
            pt_index++;
        }

        // update they y_values using the y_sum_im and y_sum_old vectors
        for (i = 0; i < size; i++)
        {
            // the movie neighbor
            itemId = neighborhoods[userId * MAX_NEIGHBOR_SIZE + i];
            for (int j = 0; j < K; j++)
            {
                y[itemId * (int)K + j] += y_sum_im[j] - y_sum_old[j];
            }
        }
    }
    delete [] y_sum_im;
    delete [] y_sum_old;
}

/*
* Iterates through every point in the training set and trains on each one
*/
inline void run_epoch()
{
    cout << "Running an epoch.\n";
    train();
    // decrease gamma_2 by 10%, as suggested in paper
    GAMMA_2 = DECAY * GAMMA_2;
    GAMMA_1 = DECAY * GAMMA_1;
}

/*
* get predictions on the qual set and output them to a file
* fix this to work with baseline removed predictions
*
*/
inline void findQualPredictions()
{
    cout << "Finding and writing qual predictions.\n";
    ofstream outputFile;
    ofstream probeFile;
    outputFile.open("SVD++_qual_output.dta");
    probeFile.open("SVD++_probe_output.dta");

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
            else if(indices[i] == 4)
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
    double finalError;
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";
    while (counter <= MAX_EPOCHS) {
        cout << "Starting Epoch " << counter << "\n";
        run_epoch();
        if (counter % 5 == 0)
        {
            finalError = error(2);
            cout << "Error after " << counter << " epochs: " << finalError << "\n";
        }
        counter++;
    }
    cout << "Final validation error: " << finalError << "\n";

    // find the values on the qual set
    findQualPredictions();

    clean_up();
    return 0;
}

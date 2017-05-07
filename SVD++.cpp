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
#define GAMMA_2 0.007 // obtained from the SVD++ paper
#define LAMBDA_7 0.015 // obtained from the SVD++ paper

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
void initialize()
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
void clean_up()
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
void read_data()
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
* Calculate the vector combining user factors and neighborhood factors
* this is p_u + |N(u)|^(-1/2) * sum of y's in neighborhood of u
* when using this, remember to delete the allocated memory for the user vector
*/
double* get_user_vector(int user, int movie)
{
    // stores the sum of user vectors
    // (all the y's in the users neighborhood plus the user factors)
    // initializes to 0 because it accumulates the other values
    double * user_vector_sum = new double[(int)K]();
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
                user_vector_sum[j] += y[neighborhood_movie * (int)K + j];
            }
        }
    }
    // add the user factors to the neighborhood sum
    // neighborhood sum must be divided by square root of neighborhood size 
    // first
    // then takes inner product of user vector and movie vector
    if (neighborhood_sizes[user] > 0)
    {
        for (int i = 0; i < K; i++)
        {
            user_vector_sum[i] = user_vector_sum[i]/sqrt(neighborhood_sizes[user]) + user_values[user * (int)K + i];
        }
    }
    else
    {
        for (int i = 0; i < K; i++)
        {
            user_vector_sum[i] = user_values[user * (int)K + i];
        }
    }
    return user_vector_sum;
}


/*
* Finds the predicted rating using the movie and the user vector sum that 
* we found earlier
*
*/
double predict_rating(int movie, double * user_vector_sum)
{
    double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector_sum[i] * movie_values[movie * (int)K + i];
    }
    return rating;
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
            double *user_vector_sum = get_user_vector((int)ratings[i * POINT_SIZE], (int)ratings[i * POINT_SIZE + 1]);
            diff = ratings[i * POINT_SIZE + 3] - predict_rating((int)ratings[i * POINT_SIZE + 1], user_vector_sum);
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
    double * user_vector_sum = get_user_vector((int)user, (int)movie);
    double point_error = rating - predict_rating((int)movie, user_vector_sum);

    // update the movie and user factors 
    double movie_factor;
    double user_factor;
    for (int i = 0; i < K; i++)
    {
        // stores the current factor that we are updating
        movie_factor = movie_values[(int)movie * (int)K + i];
        // update the movie factor in the movie values array
        movie_values[(int)movie * (int)K + i] = movie_factor + GAMMA_2 * (point_error * user_vector_sum[i] - LAMBDA_7 * movie_factor);
        // stores the current user factor
        user_factor = user_values[(int)user * (int)K + i];
        // update the user factor in the user values array
        user_values[(int)user * (int)K + i] = user_factor + GAMMA_2 * (point_error * movie_factor - LAMBDA_7 * user_factor);
    }

    // update the neighbors factors
    double y_factor;
    int movie_neighbor;
    double size = neighborhood_sizes[(int)user];
    for (int i = 0; i < size; i++)
    {
        movie_neighbor = neighborhoods[(int)user * MAX_NEIGHBOR_SIZE + i];
        for(int j = 0; j < K; j++)
        {
            y_factor = y[movie_neighbor * (int)K + j];
            movie_factor = movie_values[(int)movie * (int)K + j];
            y[movie_neighbor * (int)K + j] = y_factor + GAMMA_2 * (point_error/sqrt(size) * movie_factor - LAMBDA_7 * y_factor);
        }
    }

    // delete the user_vector_sum because it was allocated in the get_user_vector
    // function
    delete [] user_vector_sum;
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
        if (i%1000000 == 0)
        {
            cout << "i: " << i << "\n";
        }
    }
}

/*
* get predictions on the qual set and output them to a file
* fix this to work with baseline removed predictions
*
*/
void findQualPredictions()
{
    cout << "Finding and writing qual predictions.\n";
    ofstream outputFile;
    outputFile.open("SVD++_output.dta");

    int index;
    double prediction;
    double * user_vector_sum;
    for(int i = 0; i < num_pts; i++)
    {
        index = indices[i];
        if (index == 5)
        {
            user_vector_sum = get_user_vector((int)ratings[i * POINT_SIZE], (int)ratings[i * POINT_SIZE + 1]);
            // I have to add the ratings in the file because this ratings file has the baselines removed
            // this rating is negative, so I must multiply by -1 first
            prediction = baselinePrediction((int)ratings[i * POINT_SIZE], (int)ratings[i * POINT_SIZE + 1], (int)ratings[i * POINT_SIZE + 2]) + predict_rating((int)ratings[i * POINT_SIZE + 1], user_vector_sum);
            if (prediction < 1)
            {
                prediction = 1;
            }
            if (prediction > 5)
            {
                prediction = 5;
            }
            outputFile << prediction << "\n";
            delete [] user_vector_sum;
        }
    }
}

int main()
{
    initialize();
    read_data();

    double initialError = 10;
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
    findQualPredictions();

    clean_up();
    return 0;
}


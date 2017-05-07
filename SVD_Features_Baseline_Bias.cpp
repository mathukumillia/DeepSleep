#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "baselinePrediction.h"
using namespace std;

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 20

// these are all the global variables used by the program

// users and movies are one indexed
const int num_users = 458293;
const int num_movies = 17770;
const int num_pts = 102416306;
const double lambda = 0.02;

// K is the constant representing the number of features
// lrate is the learning rate
const int K = 50;
const double lrate = 0.005;
const double global_mean = 3.6033;

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

// arrays for user and movie biases
double *user_biases;
int user_biases_size = num_users + 1;
double *movie_biases;
int movie_biases_size = num_movies + 1;


inline double getRandom() {
  // generate random number from 10 to 100
   int first = rand() % 91 + 10;
   double second = 1.0;

   // get a random double between 0.1 and 0.01
   return second / first;
}

// creates arrays for storage
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

    user_biases = new double [user_biases_size];
    for (int i = 0; i < user_biases_size; ++i)
    {
        user_biases[i] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
    }

    movie_biases = new double [movie_biases_size];
    for (int i = 0; i < movie_biases_size; ++i)
    {
        movie_biases[i] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
    }

    cout << "Done allocating memory.\n";
}

/*
* Reads the input data into ratings and indices
*/
inline void read_data()
{
    cout << "Reading in training data.\n";

    // read in ratings data - currently, this is training without the baseline
    fstream ratings_file("ratings_baseline_removed.bin", ios::in | ios::binary);
    ratings_file.read((char *)(ratings), sizeof(double) * num_pts * POINT_SIZE);
    ratings_file.close();

    // read in index data
    fstream indices_file("indices.bin", ios::in | ios::binary);
    indices_file.read((char *)(indices), sizeof(double) * num_pts);
    indices_file.close();
}

// reads in and stores the ratings and indexes
inline void readRatingsIndexes()
{
    cout << "Reading in the training data." << "\n";
    cout << "Note to user: Make sure the input file paths are correct.\n";
    // read in data
    // change file paths as necessary
    fstream ratingsFile ("../../Caltech/CS156B/um/all.dta");
    fstream indexFile ("../../Caltech/CS156B/um/all.idx");
    // user, movie, date, rating
    double inputs [4] = {};
    int index = 0;
    int pt = 0;

    if (ratingsFile.is_open() && indexFile.is_open()) {
      while (ratingsFile >> inputs[0] >> inputs[1] >> inputs[2] >> inputs[3]) {
          indexFile >> index;
          indices[pt] = index;

          double user = inputs[0];
          double movie = inputs[1];
          double date = inputs[2];
          double rating = inputs[3];

          ratings[pt * POINT_SIZE] = user;
          ratings[pt * POINT_SIZE + 1] = movie;
          ratings[pt * POINT_SIZE + 2] = date;
          // subtract baseline from ratings
          ratings[pt * POINT_SIZE + 3] = rating - baselinePrediction(user, movie, date);

          // increment pt so that we update the values for a new point next time
          pt += 1;
      }
    }

    ratingsFile.close();
    indexFile.close();
} 

// given a user and a movie this function gives the predicted rating
inline double predict_rating(int user, int movie)
{
    double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_values[user * K + i] * movie_values[movie * K + i];
    }
    return rating;
}

inline double magSquared(double* vector) {
    double mag = 0;
    for (int i = 0; i < K; i++) {
        mag += vector[i] * vector[i];
    }
    return mag;
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
            int user = (int)ratings[i * POINT_SIZE];
            int movie = (int)ratings[i * POINT_SIZE + 1];
            double rating = ratings[i * POINT_SIZE + 3];
            double prediction = user_biases[user] + movie_biases[movie] + predict_rating(user, movie);
            //cout << "prediction: " << prediction << "\n";
            //cout << "rating: " << rating << "\n";

            diff = rating - prediction;
            error += diff * diff;

            //cout << "diff^2: " << diff * diff << "\n";
            //cout << "error: " << error << "\n";
            //cout << "pt: " << i << "\n";
            double inf = std::numeric_limits<double>::infinity();

            if(error >= inf){
              cout << "ERROR IS INFINITY!!" << "\n";
              return 1000000;
            }

            points_in_set += 1;
        }
    }

    return sqrt(error/points_in_set);
}

inline void train(int user, int movie, double rating, int feature)
{
    // calculate the error with the current feature values
    double err = rating - (user_biases[user] + movie_biases[movie] + predict_rating(user, movie));

    // updates the movie and user vectors for given feature
    double uv = user_values[user * K + feature];

    user_values[user * K + feature] += lrate * (err * movie_values[movie * K + feature] - lambda * uv);
    movie_values[movie * K + feature] += lrate * (err * uv - lambda * movie_values[movie * K + feature]);

    // update biases
    double oldUserBias = user_biases[user];
    user_biases[user] += lrate * (err - lambda * (oldUserBias + movie_biases[movie] - global_mean));
    movie_biases[movie] += lrate * (err - lambda * (oldUserBias + movie_biases[movie] - global_mean));
}

/*
* Run a full epoch.
*/
inline void run_epoch (int feature)
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
    int user = (int)ratings[pt * POINT_SIZE];
    int movie = (int)ratings[pt * POINT_SIZE + 1];
    double rating = ratings[pt * POINT_SIZE + 3];
    train(user, movie, rating, feature);
  }

  cout << "Epoch complete." << "\n";
}

inline void find_qual_predictions()
{
  ofstream outputFile;
  outputFile.open("SVD_bias_output.dta");

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
          prediction = baselinePrediction(user, movie, date) + predict_rating(user, movie)
              + user_biases[user] + movie_biases[movie];
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

inline void storeUserFeatures()
{
    cout << "storing user features" << "\n";

    ofstream featureFile;
    featureFile.open("userFeatures.dta");
    // file has userid followed by all K feature values

    for (int user = 0; user < num_users; user++) {
       featureFile << user << " ";
       for (int feature = 0; feature < K; feature++) {
           featureFile << user_values[user * K + feature] << " ";
       }
       featureFile << "\n";
    }
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
    delete [] user_biases;
    delete [] movie_biases;
}

int main()
{
    initialize();
    readRatingsIndexes();

    double initialError = 10000;
    double finalError = error(2); // gets the validation error before training
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";

    for (int feature = 0; feature < K; feature++) {
      int featureEpochCounter = 1;
      initialError = 10000;

      while (initialError - finalError > STOPPING_CONDITION && featureEpochCounter <= MAX_EPOCHS) {
        cout << "Feature: " << feature << "\n";
        cout << "Starting Epoch " << counter << "\n";

        initialError = finalError;
        run_epoch(feature);
        finalError = error(2); // error(2) returns the validation error

        cout << "Error after Epoch " << counter << ": " << finalError << "\n";
        counter++;
        featureEpochCounter++;
        cout << "-----------------------------------\n";
      }
    }
    
    // find the values on the qual set
    find_qual_predictions();

    clean_up();
    return 0;
}
 

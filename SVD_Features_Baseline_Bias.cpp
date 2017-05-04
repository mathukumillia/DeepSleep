#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include "baselinePrediction.h"
using namespace std;

// these are all the global variables used by the program

// users and movies are one indexed
const double numUsers = 458293;
const double numMovies = 17770;
const double numPts = 102416306;

// K is the constant representing the number of features
// lrate is the learning rate
const double K = 10;
const double lrate = 0.001;
const double lambda = 0.002;
const double global_mean = 3.6033;

// these 2D arrays are the U and V in the SVD
double **userValues;
double **movieValues;

// 2D arrays to store ratings and indexes
double **ratings;
double *indexes;

// arrays for user and movie biases
double *userBiases;
double *movieBiases;

// creates arrays for storage
void initialize()
{
      cout << "Initializing the program.\n";
      // allocate and initialize the userValues and movieValues matrices
      // all of the +1 terms result from the fact that these arrays are 1
      // indexed in the data
      userValues = new double*[((int) numUsers + 1)];
      for (int i = 0; i < numUsers + 1; i++)
      {
        userValues[i] = new double[(int) K];
        for (int j = 0; j < K; j++)
        {
          userValues[i][j] = 0.1;
        }
      }

      movieValues = new double*[((int) numMovies + 1)];
      for (int i = 0; i < numMovies + 1; i++)
      {
        movieValues[i] = new double[(int) K];
        for (int j = 0; j < K; j++)
        {
          movieValues[i][j] = 0.1;
        }
      }
      // create  the arrays that store the ratings input data and the indexes
      ratings = new double* [((int) numPts)];
      for(int i = 0; i < numPts; i++)
      {
          ratings[i] = new double [4];
      }

      indexes = new double [((int) numPts)];

      userBiases = new double [((int) numUsers)];
      for (int i = 0; i < numUsers; ++i)
      {
          userBiases[i] = 0.1;
      }

      movieBiases = new double [((int) numMovies)];
      for (int i = 0; i < numMovies; ++i)
      {
          movieBiases[i] = 0.1;
      }

      cout << "Done allocating memory.\n";
}

// reads in and stores the ratings and indexes
void readRatingsIndexes()
{
    cout << "Reading in the training data." << "\n";
    cout << "Note to user: Make sure the input file paths are correct.\n";
    // read in data
    // change file paths as necessary
    fstream ratingsFile ("../../Caltech/CS156B/um/all.dta");
    fstream indexFile ("../../Caltech/CS156B/um/all.idx");
    // user, movie, date, rating
    double inputs [4] = {};
    int index;
    int pt = 0;

    if (ratingsFile.is_open() && indexFile.is_open()) {
      while (ratingsFile >> inputs[0] >> inputs[1] >> inputs[2] >> inputs[3]) {
          indexFile >> index;
          indexes[pt] = index;

          double user = inputs[0];
          double movie = inputs[1];
          double date = inputs[2];
          double rating = inputs[3];

          ratings[pt][0] = user;
          ratings[pt][1] = movie;
          ratings[pt][2] = date;
          // subtract baseline from ratings
          ratings[pt][3] = rating - baselinePrediction(user, movie, date);

          // increment pt so that we update the values for a new point next time
          pt += 1;
      }
    }

    ratingsFile.close();
    indexFile.close();
}

// given a user and a movie this function gives the predicted rating
double predictRating(int user, int movie)
{
    double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += userValues[user][i] * movieValues[movie][i];
    }
    return rating;
}

double magSquared(double* vector) {
    double mag = 0;
    for (int i = 0; i < K; i++) {
        mag += vector[i] * vector[i];
    }
    return mag;
}

double error ()
{
  cout << "Calculating Validation error..." << "\n";

  //  counter keeps track of the number of points we've been through
  int counter = 0;
  long double error = 0;
  double diff = 0;
  double index = 0;
  double numValidationPts = 0;

  for (int i = 0; i < numPts; i++) {
      index = indexes[i];

      if (index == 2) {
          int user = ratings[i][0];
          int movie = ratings[i][1];
          double rating = ratings[i][3];
          diff = rating - (userBiases[user] + movieBiases[movie] + predictRating(user, movie));

          //cout << "diff " << diff << "\n";

          double errorpart1 = diff * diff;
          long double errorpart2 = (magSquared(userValues[user]) + magSquared(movieValues[movie]) + 
            userBiases[user] * userBiases[user] + movieBiases[movie] * movieBiases[movie]);

          error += errorpart1 + lambda * errorpart2;

          //cout << "error1 " << errorpart1 << "\n";
          //cout << "error2 " << errorpart2 << "\n";
          //cout << "error " << error << "\n";
          numValidationPts += 1;

          //cout << "pt number " << numValidationPts << "\n";
      }
      counter++;
  }

  return sqrt(error/numValidationPts);
}

void train(int user, int movie, int rating, int feature)
{
    // calculate the error with the current feature values
    double err = (double) rating - (userBiases[user] + movieBiases[movie] + predictRating(user, movie));

    // updates the movie and user vectors for given feature
    double *uv = userValues[user];

    userValues[user][feature] += lrate * (err * movieValues[movie][feature] - lambda * userValues[user][feature]);
    movieValues[movie][feature] += lrate * (err * uv[feature] - lambda * movieValues[movie][feature]);

    // update biases
    double oldUserBias = userBiases[user];
    userBiases[user] += lrate * (err - lambda * (oldUserBias + movieBiases[movie] - global_mean));
    movieBiases[movie] += lrate * (err - lambda * (oldUserBias + movieBiases[movie] - global_mean));
}

void runEpoch (int feature)
{
    cout << "Running Epoch..." << "\n";
    for (int i = 0; i < numPts; i++) {
        int index = indexes[i];

        if (index == 1) {
            train(ratings[i][0], ratings[i][1], ratings[i][3], feature);
        }
    }
    cout << "Epoch complete." << "\n";
}

void findQualPredictions()
{
    cout << "Finding Qual Predictions" << "\n";

    ofstream outputFile;
    fstream qualFile ("../../Caltech/CS156B/um/qual.dta");
    outputFile.open("output.dta");
    double inputs [3] = {};

    if (qualFile.is_open()) {
      while (qualFile >> inputs[0] >> inputs[1] >> inputs[2]) {
         int user = inputs[0];
         int movie = inputs[1];
         int date = inputs[2];

         // add back baseline into predictions
         double prediction = userBiases[user] + movieBiases[movie] + 
            predictRating(user, movie) + baselinePrediction(user, movie, date);

         // clip predictions within range of 1 and 5
         if(prediction < 1){
            prediction = 1;
         }
         if(prediction > 5){
            prediction = 5;
         }
         outputFile << prediction << "\n";
      }
    }
}

void storeUserFeatures()
{
    cout << "storing user features" << "\n";

    ofstream featureFile;
    featureFile.open("userFeatures.dta");
    // file has userid followed by all K feature values

    for (int user = 0; user < numUsers; user++) {
       featureFile << user << " ";
       for (int feature = 0; feature < K; feature++) {
           featureFile << userValues[user][feature] << " ";
       }
       featureFile << "\n";
    }
}

void cleanUp()
{
      cout << "Cleaning up\n" << "\n";
      // de-allocate user values and movie values to prevent memory leak
      for (int i = 0; i < numUsers + 1; i++)
      {
        delete [] userValues[i];
      }
      delete [] userValues;
      for (int i = 0; i < numMovies + 1; i++)
      {
        delete [] movieValues[i];
      }
      delete [] movieValues;

      for (int i = 0; i < numPts; i++) {
        delete[] ratings[i];
      }
      delete[] ratings;

      delete[] indexes;
}

// Opens the file and runs the SVD
int main()
{
    initialize();
    readRatingsIndexes();
    // gets the initial validation error
    // NOTE: the initial error should calculate the error in the final program
    double initialError = 10;
    double finalError = error();
    int epochCounter = 0;
    int featureEpochCounter = 0;

    cout << "Initial Error is: " << initialError << "\n";
    cout << "Final Error is:" << finalError << "\n";

    // train one feature at a time
    for(int i = 0; i < K; i++) {
      initialError = 10;
      featureEpochCounter = 0;
        // while error is decreasing
        while (initialError - finalError > 0) {
          cout << "Feature " << i << "\n";
          cout << "Starting Epoch " << epochCounter << "\n";

          epochCounter++;
          featureEpochCounter++;

          initialError = finalError;
          runEpoch(i);
          finalError = error();

          cout << "Error after Epoch " << finalError << "\n";
       }
       // didn't train on feature, because initialError - finalError < 0.0001 already
       // just train for n more epochs
       int n = 10;
       if (featureEpochCounter <= 1) {
          for (int j = 0; j < n; j++) {
            cout << "Feature " << i << "\n";
            cout << "Starting Epoch " << epochCounter << "\n";

            epochCounter++;
            featureEpochCounter++;

            initialError = finalError;
            runEpoch(i);
            finalError = error();

            cout << "Error after Epoch " << finalError << "\n";
          }
        }
  }

  findQualPredictions();
  storeUserFeatures();
  cleanUp();
  return 0;
}

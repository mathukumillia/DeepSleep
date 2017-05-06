#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "baselinePrediction.h"
using namespace std;

#define STOPPING_CONDITION 0
#define MAX_EPOCHS 50

// these are all the global variables used by the program

// users and movies are one indexed
const double numUsers = 458293;
const double numMovies = 17770;
const int numPts = 102416306;

// K is the constant representing the number of features
// lrate is the learning rate
const double K = 50;
const double lrate = 0.01;
const double lambda = 0.05;
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

double getRandom() {
  // generate random number from 10 to 100
   int first = rand() % 91 + 10;
   double second = 1.0;

   // get a random double between 0.1 and 0.01
   return second / first;
}

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
          userValues[i][j] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
        }
      }

      movieValues = new double*[((int) numMovies + 1)];
      for (int i = 0; i < numMovies + 1; i++)
      {
        movieValues[i] = new double[(int) K];
        for (int j = 0; j < K; j++)
        {
          movieValues[i][j] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
        }
      }
      // create  the arrays that store the ratings input data and the indexes
      ratings = new double* [((int) numPts)];
      for(int i = 0; i < numPts; i++)
      {
          ratings[i] = new double [4];
      }

      indexes = new double [((int) numPts)];

      userBiases = new double [((int) numUsers + 1)];
      for (int i = 0; i < numUsers + 1; ++i)
      {
          userBiases[i] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
      }

      movieBiases = new double [((int) numMovies + 1)];
      for (int i = 0; i < numMovies + 1; ++i)
      {
          movieBiases[i] = 0.1 * (double)(rand() % 10) + 0.01; // arbitrary initial condition
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
    int index = 0;
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

/*
* Gets the total error of the SVD model on the set index provided
* i.e., to get validation error, pass in set = 2
*/
double error (int set)
{
  cout << "Calculating Validation error..." << "\n";

  //  counter keeps track of the number of points we've been through
  int counter = 0;
  double error = 0;
  double diff = 0;
  int index = 0;
  double numValidationPts = 0;

  for (int i = 0; i < numPts; i++) {
      index = indexes[i];

      if (index == set) {
          int user = ratings[i][0];
          int movie = ratings[i][1];
          double rating = ratings[i][3];
          diff = rating - (userBiases[user] + movieBiases[movie] + predictRating(user, movie));
          //cout << diff << "\n";

          double errorpart1 = diff * diff;
          //double errorpart2 = (magSquared(userValues[user]) + magSquared(movieValues[movie]) + 
            //userBiases[user] * userBiases[user] + movieBiases[movie] * movieBiases[movie]);
          //double errorpart2 = magSquared(userValues[user]) + magSquared(movieValues[movie]);

          error += errorpart1;
          numValidationPts += 1;
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
    int pt;
    int index;

    for (int i = 0; i < numPts; i++) {
        // select a random pt to train with
        pt = rand() % numPts;
        index = indexes[pt];

        // make sure the selected point is in the first data set
        while (index != 1)
        {
            pt = rand() % numPts;
            index = indexes[pt];
        }

        // train on this point
        train(ratings[pt][0], ratings[pt][1], ratings[pt][3], feature);
    }
    cout << "Epoch complete." << "\n";
}

void findQualPredictions()
{
    cout << "Finding Qual Predictions" << "\n";

    ofstream outputFile;
    outputFile.open("output.dta");
    int index;
    
    for (int i = 0; i < numPts; i++) {
        index = indexes[i];

        if(index == 5){
          int user = ratings[i][0];
          int movie = ratings[i][1];
          double date = ratings[i][2];

           // add back baseline into predictions
          double prediction = (userBiases[user] + movieBiases[movie] + 
              predictRating(user, movie)) + baselinePrediction(user, movie, date);

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

      delete[] userBiases;

      delete[] movieBiases;
}

// Opens the file and runs the SVD
int main()
{
    initialize();
    readRatingsIndexes();
    // gets the initial validation error
    // NOTE: the initial error should calculate the error in the final program
    double initialError = 10000;
    double finalError = error(2);
    int epochCounter = 0;
    int featureEpochCounter = 0;

    cout << "Initial Error is: " << finalError << "\n";

    for(int i = 0; i < K; i++) {
      initialError = 10000;
      featureEpochCounter = 0;
      while (initialError - finalError > STOPPING_CONDITION && featureEpochCounter <= MAX_EPOCHS) {
          cout << "Feature " << i << "\n";
          cout << "Starting Epoch " << epochCounter  << "\n";
          initialError = finalError;
          runEpoch(i);
          finalError = error(2); // error(2) returns the validation error
          cout << "Error after Epoch " << epochCounter  << ": " << finalError << "\n";
          featureEpochCounter++;
          epochCounter++;
          cout << "-----------------------------------\n";
      }
  }

    // train one feature at a time
   //for(int i = 0; i < K; i++) {
      //initialError = 10;
      //featureEpochCounter = 1;
        // while error is decreasing by threshold
        /*while (initialError - finalError > threshold) {
          cout << "Feature " << i << "\n";
          cout << "Starting Epoch " << epochCounter << "\n";

          epochCounter++;
          featureEpochCounter++;

          initialError = finalError;
          runEpoch(i);
          finalError = error();

          cout << "Error after Epoch " << finalError << "\n";
       }
       int unforcedEpochs = featureEpochCounter;
       // didn't train on feature, because initialError - finalError < 0.0001 already
       // just train for n more epochs
       int minEpochs = 10;
       if (unforcedEpochs < minEpochs) {
          for (int j = 0; j < minEpochs - unforcedEpochs; j++) {
            cout << "FORCE TRAINING" << "\n";
            cout << "Feature " << i << "\n";
            cout << "Starting Epoch " << epochCounter << "\n";

            epochCounter++;
            featureEpochCounter++;

            initialError = finalError;
            runEpoch(i);
            finalError = error();

            cout << "Error after Epoch " << finalError << "\n";
          }
        }*/
      /*  for(int j = 0; j < 30; j++){
          cout << "Feature " << i << "\n";
          cout << "Starting Epoch " << j << "\n";

          initialError = finalError;
          runEpoch(i);
          finalError = error();

          cout << "Error after Epoch " << finalError << "\n";

          featureEpochCounter++;
        }*/
      findQualPredictions();
      storeUserFeatures();
      cleanUp();
      return 0;
  }

//}

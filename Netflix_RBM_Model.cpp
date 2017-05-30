#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include <map>
#include <set>
#include "RBM.h"

using namespace std;

int RBM::numHiddenUnits = 20;
int RBM::numRatingValues = 6;
double RBM::learningRate = 0.01;

double*** RBM::weights = new double**[RBM::numHiddenUnits];
double*** RBM::deltaWeights = new double**[RBM::numHiddenUnits];

map<int, set<int>* > RBM::userMovies;
// these are all the global variables used by the program

// users and movies are one indexed
const int numUsers = 458293;
const int numMovies = 17770;
const int numPts = 102416306;

int RBM::numMovies = numMovies;

// 2D arrays to store ratings and indexes
double **ratings;
double *indexes;

RBM* userRBMs[numUsers + 1];

void initialize()
{
  cout << "creating ratings array..." << "\n";
  // create  the arrays that store the ratings input data and the indexes
  ratings = new double* [numPts];
  for(int i = 0; i < numPts; i++)
  {
      ratings[i] = new double [4];
  }

  cout << "creating indexes array..." << "\n";
  indexes = new double [numPts];

  //userMovies = new map<int, unordered_set<int> > ();
}

void initializeRBMs() {
  cout << "initializing weights..." << "\n";
  for (int hiddenUnit = 0; hiddenUnit < RBM::numHiddenUnits; hiddenUnit++) {
    RBM::weights[hiddenUnit] = new double*[RBM::numRatingValues];

    for (int ratingValue = 0; ratingValue < RBM::numRatingValues; ratingValue++) {
        RBM::weights[hiddenUnit][ratingValue] = new double[numMovies + 1];

        for (int visibleUnit = 0; visibleUnit < numMovies + 1; visibleUnit++) {
            // set weight to some small positive value
            RBM::weights[hiddenUnit][ratingValue][visibleUnit] = 0.1;
        }
    }
  }

  for (int hiddenUnit = 0; hiddenUnit < RBM::numHiddenUnits; hiddenUnit++) {
    RBM::deltaWeights[hiddenUnit] = new double*[RBM::numRatingValues];

    for (int ratingValue = 0; ratingValue < RBM::numRatingValues; ratingValue++) {
        RBM::deltaWeights[hiddenUnit][ratingValue] = new double[numMovies + 1];

        for (int visibleUnit = 0; visibleUnit < numMovies + 1; visibleUnit++) {
            // set delta weight to some small positive value
            RBM::deltaWeights[hiddenUnit][ratingValue][visibleUnit] = 0.1;
        }
    }
  }

  cout << "creating an rbm for each user..." << "\n";
  // create rbms for each user, with number of visible units == number of movies
  for(int userID = 0; userID < numUsers + 1; userID++) {
      /*int size = 0;
      if(RBM::userMovies[userID]!= NULL) {
          size = (RBM::userMovies[userID])->size();
      }*/
      // cout << "num movies rated: " << size << endl;
      userRBMs[userID] = new RBM(userID);

  }

  cout << "finished creating rbms..." << "\n";
  // set visible units
  for(int i = 0; i < numPts; i++) {
    int user = ratings[i][0];
    int movie = ratings[i][1];
    int rating = ratings[i][3];
    cout << "read rating details" << endl;
    RBM* rbm = userRBMs[user];
    cout << "got rbm for user" << endl;
    if(rbm == NULL) {
      cout << "rbm is null!" << endl;
    }
    else{
      cout << "rbm is not null" << endl;
    }
    if(rbm->visibleUnits == NULL) {
      cout << "visible units array is null!" << endl;
    }
    else{
      cout << "rating: " << rating << endl;
      cout << "movie: " << movie << endl;
      cout << "user: " << user << endl;
      cout << rbm->userID << endl;
      rbm->visibleUnits[rating][movie] = 1;
      cout << "set visible unit " << endl;
    }
  }

  cout << "finished setting visible units.." << "\n";
  // go to user rbm, update probability of movie and rating
}

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
        ratings[pt][3] = rating;

        // increment pt so that we update the values for a new point next time
        pt += 1;
        //cout << "stored rating" << endl;
        // add user, movie to map
        //unordered_set<int>* movieSet = userMovies[(int) user];
        
        if (RBM::userMovies[(int) user] == NULL){
          //cout << "movie set is null" << endl;
          RBM::userMovies[(int) user] = new set<int> ();
          //cout << "movie set made new" << endl;
          if(RBM::userMovies[(int) user] == NULL) {
           // cout << "movie set is still null" << endl;
          }
          RBM::userMovies[(int) user]->insert((int) movie);
          //cout << "inserted movie into movie set" << endl;
        }
        else {
          RBM::userMovies[(int) user]->insert((int) movie);
        }

        // set visible units of RBMs
        // activate unit based on rating
        /*double** visibleUnits = userRBMs[(int) user]->visibleUnits;
        cout << "got visible units" << endl;
        visibleUnits[(int)rating][(int)user] = 1;
        cout << "set visible units" << endl;*/
    }
  }
}

double predictRating(RBM* userRBM, int movie) {
    // take weighted average of rating for movie for the given user
    double** visibleUnitsProbabilities = userRBM->visibleUnitsProbabilities;
    double prediction = 0;
    for (int ratingValue = 1; ratingValue <= RBM::numRatingValues; ratingValue++) {
        prediction += visibleUnitsProbabilities[ratingValue][movie] * ratingValue;
    }
    return prediction;
}

void findQualPredictions()
{
  ofstream outputFile;
  fstream qualFile ("../../Caltech/CS156B/um/qual.dta");
  outputFile.open("output.dta");
  double inputs [3] = {};

  if (qualFile.is_open()) {
    while (qualFile >> inputs[0] >> inputs[1] >> inputs[2]) {
       double prediction = predictRating(userRBMs[(int)inputs[0]], (int)inputs[1]);
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

int main() {
    initialize();
    readRatingsIndexes();
    initializeRBMs();
    int maxEpochs = 20;

    for (int epoch = 0; epoch < maxEpochs; epoch++) {
        // run epoch on all RBMS
        for(int userID = 0; userID < numUsers + 1; userID++) {
            userRBMs[userID]->trainEpoch();
        }

        // divide weights by number of users, to get average weights
        for (int featureNum = 0; featureNum < RBM::numHiddenUnits; featureNum++) {
            for (int ratingValue = 0; ratingValue < RBM::numRatingValues; ratingValue++) {
                for (int movieNum = 0; movieNum < numMovies; movieNum++) {
                    RBM::weights[featureNum][ratingValue][movieNum] /= numUsers;
                }
            }
        }
    }

    findQualPredictions();

    return 0;
}
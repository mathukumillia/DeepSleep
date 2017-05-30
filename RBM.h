#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include <map>
#include <set>

using namespace std;

// each user will have an RBM
class RBM {
public:
    // number of hidden units is same for each RBM
    // similar to features in SVD
    static int numHiddenUnits;
    static int numRatingValues;
    static double learningRate;
    // there is a weight corresponding to each hidden unit, rating, movie triple
    // all users share the same array of weights
    static double*** weights;
    static double*** deltaWeights;
    // keep track of how many users rated each movie
    // key: movieID, value: numUsers
    static map<int, int> numMovieRatings;
    // set to hold user, num movies rated
    static map<int, set<int>* > userMovies;
    static int numMovies;

    // number of visible units is number of movies a user has rated
    int numVisibleUnits;

    // id of user that this RBM is for
    int userID;

    double* hiddenUnits;

    double* hiddenUnitsProbabilities;

    // visible units is a K x M matrix, K is the number of different ratings
    // M is the number of movies, only the columns for movies rated by the user will be used
    double** visibleUnits;

    double** visibleUnitsProbabilities;

    // bias of rating k for movie i
    // K x M matrix, K is the number of different ratings,
    // M is the number of movies rated by the user
    double** movieRatingBiases;

    // bias of feature
    double* featureBiases;

    // array of movieIDs user has rated
    int* movies;

    // constructor
    RBM (int userID);
    RBM();
    void updateVisibleUnit(int ratingValue, int movie);
    void updateHiddenUnit(int feature);
    double sigmoid(int val);
    double deltaW(double dataExpectation, double reconstructionExpectation);
    void trainEpoch();
};
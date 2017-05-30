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


RBM::RBM (int userID) {
    this->userID = userID;
    //this->numVisibleUnits = numVisibleUnits;

    hiddenUnits = new double[numHiddenUnits];
    hiddenUnitsProbabilities = new double[numHiddenUnits];

    visibleUnits = new double*[numRatingValues];
    for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
        visibleUnits[ratingValue] = new double[numMovies + 1];

        for(int visibleUnit = 0; visibleUnit < numMovies + 1; visibleUnit++) {
            visibleUnits[ratingValue][visibleUnit] = 0;
        }
    }

    visibleUnitsProbabilities = new double*[numRatingValues];
    for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
        visibleUnitsProbabilities[ratingValue] = new double[numMovies + 1];

        for(int visibleUnit = 0; visibleUnit < numMovies + 1; visibleUnit++) {
            visibleUnitsProbabilities[ratingValue][visibleUnit] = 0;
        }
    }

    /*positiveAssociations = new double**[numHiddenUnits];
    for (int hiddenUnit = 0; hiddenUnit < numHiddenUnits; hiddenUnit++) {
        positiveAssociations[hiddenUnit] = new double*[numRatingValues];

        for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
            positiveAssociations[hiddenUnit][ratingValue] = new double[numVisibleUnits];

            for (int visibleUnit = 0; visibleUnit < numVisibleUnits; visibleUnit++) {
                // set weight to some small positive value
                positiveAssociations[hiddenUnit][ratingValue][visibleUnit] = 0.1;
            }
        }
    }

    negativeAssociations = new double**[numHiddenUnits];
    for (int hiddenUnit = 0; hiddenUnit < numHiddenUnits; hiddenUnit++) {
        negativeAssociations[hiddenUnit] = new double*[numRatingValues];

        for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
            negativeAssociations[hiddenUnit][ratingValue] = new double[numVisibleUnits];

            for (int visibleUnit = 0; visibleUnit < numVisibleUnits; visibleUnit++) {
                // set weight to some small positive value
                negativeAssociations[hiddenUnit][ratingValue][visibleUnit] = 0.1;
            }
        }
    }*/

    movieRatingBiases = new double*[numRatingValues];
    for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
        movieRatingBiases[ratingValue] = new double[numMovies + 1];
    }
    
    featureBiases = new double[numHiddenUnits];
}

void RBM::updateVisibleUnit(int ratingValue, int movie) {
    double sum_hiddenWeights = 0;

    for (int hiddenUnitNum = 0; hiddenUnitNum < numHiddenUnits; hiddenUnitNum++) {
        sum_hiddenWeights += hiddenUnits[hiddenUnitNum] * weights[hiddenUnitNum][ratingValue][movie];
    }

    double numerator = exp(movieRatingBiases[ratingValue][movie] + sum_hiddenWeights);

    double denominator = 0;
    for (int ratingValueNum = 0; ratingValueNum < numRatingValues; ratingValueNum++) {
        double innerDenom = movieRatingBiases[ratingValueNum][movie];

        for (int hiddenUnitNum = 0; hiddenUnitNum < numHiddenUnits; hiddenUnitNum++) {
            innerDenom += hiddenUnits[hiddenUnitNum] * weights[hiddenUnitNum][ratingValueNum][movie];
        }

        denominator += exp(innerDenom);
    }

    // update visible unit
    double probability = numerator / denominator;

    double randNum = ((double) rand() / (RAND_MAX));
    if (randNum <= probability) {
        visibleUnits[ratingValue][movie] = 1;
    }
    else {
        visibleUnits[ratingValue][movie] = 0;
    }
}

void RBM::updateHiddenUnit(int feature) {
    double feature_bias = featureBiases[feature];
    double innerProb = feature_bias;
    for(set<int>::iterator it = userMovies[userID]->begin(); it != userMovies[userID]->end(); it++) {
        int movieNum = *it;
        for(int ratingNum = 0; ratingNum < numRatingValues; ratingNum++) {
            innerProb += visibleUnits[ratingNum][movieNum] * weights[feature][ratingNum][movieNum];
        }
    }
    // update hidden unit
    double probability = sigmoid(innerProb);
    hiddenUnitsProbabilities[feature] = probability;

    double randNum = ((double) rand() / (RAND_MAX));
    if (randNum <= probability) {
        hiddenUnits[feature] = 1;
    }
    else {
        hiddenUnits[feature] = 0;
    }
}

double RBM::sigmoid(int val) {
    return 1.0 / (1 + exp(-1 * val));
}

// contrastive divergence function
double RBM::deltaW(double dataExpectation, double reconstructionExpectation) {
    return learningRate * (dataExpectation - reconstructionExpectation);
}

void RBM::trainEpoch() {
    // compute binary states of hidden units
    for (int featureNum = 0; featureNum < numHiddenUnits; featureNum++) {
        updateHiddenUnit(featureNum);
    }

    // calculate positive associations
    // data visible unit vectors * hidden probabilities
    for (int featureNum = 0; featureNum < numHiddenUnits; featureNum++) {
        for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
            for (set<int>::iterator it = userMovies[userID]->begin(); it != userMovies[userID]->end(); it++) {
                int movieNum = *it;
                deltaWeights[featureNum][ratingValue][movieNum] = 
                    visibleUnits[ratingValue][movieNum] * hiddenUnitsProbabilities[featureNum];
            }
        }
    }

    // reconstruction: calculate the binary states of the visible units
    for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
        for (set<int>::iterator it = userMovies[userID]->begin(); it != userMovies[userID]->end(); it++) {
            int movieNum = *it;
            updateVisibleUnit(ratingValue, movieNum);
        }
    }

    // calculate negative associations
    // visible probabilities * hidden probabilities
    // and update weights
    for (int featureNum = 0; featureNum < numHiddenUnits; featureNum++) {
        for (int ratingValue = 0; ratingValue < numRatingValues; ratingValue++) {
            for (set<int>::iterator it = userMovies[userID]->begin(); it != userMovies[userID]->end(); it++) {
                int movieNum = *it;
                deltaWeights[featureNum][ratingValue][movieNum] = 
                    learningRate * 
                    (deltaWeights[featureNum][ratingValue][movieNum] - 
                    visibleUnitsProbabilities[ratingValue][movieNum] * hiddenUnitsProbabilities[featureNum]);

                weights[featureNum][ratingValue][movieNum] += deltaWeights[featureNum][ratingValue][movieNum];
            }
        }
    }

}

/*void RBM::findNumMovies() {
    return 0;
}*/






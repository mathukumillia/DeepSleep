// -*- C++ -*-

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

int main() {

// THE USER ONLY NEEDS TO CUSTOMIZE THE NUMBER OF PREDICTIONS AND THE FILE NAMES

////////////////////////////////////////////////////////////////////////////////

  // specify the number of predictions to blend
  const int numberOfPredictionsToBlend = 3;  

  // enter the names of the probe prediction files
  vector<const char*> probePredictionFiles;
  probePredictionFiles.push_back("probePrediction_eq10.dta");
  probePredictionFiles.push_back("probePrediction_eq11.dta");
  probePredictionFiles.push_back("probePrediction_eq4.dta");

  // enter the names of the qual prediction files
  vector<const char*> qualPredictionFiles;
  qualPredictionFiles.push_back("qualPrediction_eq10.dta");
  qualPredictionFiles.push_back("qualPrediction_eq11.dta");
  qualPredictionFiles.push_back("qualPrediction_eq4.dta");

/////////////////////////////////////////////////////////////////////////////////

  // import the probe ratings and insert into "Y" for linear regression
  int user;
  int movie;
  int time;
  int rating;
  int numberOfProbeRatings = 1374739;
  VectorXd Y(numberOfProbeRatings);
  ifstream probeDataFile("probe.dta");
  for (int probeIndex=0; probeIndex<numberOfProbeRatings; probeIndex++){
    probeDataFile >> user >> movie >> time >> rating;
    Y(probeIndex) = double(rating);
  }
  probeDataFile.close();
 
  // import the probe predictions and insert into "X" for linear regression;
  double prediction;
  MatrixXd X(numberOfProbeRatings,numberOfPredictionsToBlend);
  for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
    ifstream probePredictionFile(probePredictionFiles[predictionIndex]);
    for (int probeIndex=0; probeIndex<numberOfProbeRatings; probeIndex++){
      probePredictionFile >> prediction;
      X(probeIndex,predictionIndex) = prediction;
    }
    probePredictionFile.close();
  }
  probeDataFile.close();

  // perform linear regression to solve for weights, "w"
  VectorXd w;
  MatrixXd pseudoInverse = X.transpose()*X;
  MatrixXd pseudoOutput = X.transpose()*Y;
  w = (pseudoInverse).lu().solve(pseudoOutput);

  // print weights to terminal
  cout << "w = " << endl << w << endl;

  // make qual predictions and write to file
  int numberOfQualRatings = 2749898;
  ofstream qualPredictionBlendFile;
  qualPredictionBlendFile.open("qualPredictionBlend.dta");
  double predictionIndividual;
  vector<vector<double> > qualPredictionsIndividual;
  for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
    ifstream qualPredictionFile(qualPredictionFiles[predictionIndex]);
    vector<double> predictionVector;
    for (int qualIndex=0; qualIndex<numberOfQualRatings; qualIndex++){
      qualPredictionFile >> predictionIndividual;
      predictionVector.push_back(predictionIndividual);
    }
    qualPredictionsIndividual.push_back(predictionVector);
    qualPredictionFile.close();
  }
  for (int qualIndex=0; qualIndex<numberOfQualRatings; qualIndex++){
    prediction = 0;
    for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
      prediction = prediction + w(predictionIndex)*qualPredictionsIndividual[predictionIndex][qualIndex];    
    }
    qualPredictionBlendFile << prediction << endl;
  }


  qualPredictionBlendFile.close(); 

  // as a check, calculate the rmse of the blend on the probe set and compare
  // to the indidiaul rmse's to make sure the blend is better
  double sErrorBlend = 0;
  vector<double> sErrorIndividual;
  for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
    sErrorIndividual.push_back(0.);
  }
  for (int probeIndex=0; probeIndex<numberOfProbeRatings; probeIndex++){
    prediction = 0.;
    for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
      prediction = prediction + w(predictionIndex)*X(probeIndex,predictionIndex);
      sErrorIndividual[predictionIndex] = sErrorIndividual[predictionIndex] + pow(X(probeIndex,predictionIndex)-Y(probeIndex),2);
    }
    sErrorBlend = sErrorBlend + pow(prediction-Y(probeIndex),2);
  }
  double rmsErrorBlend = pow(sErrorBlend/double(numberOfProbeRatings),0.5);

  vector<double> rmsErrorIndividual;
  for (int predictionIndex=0; predictionIndex<numberOfPredictionsToBlend; predictionIndex++){
    rmsErrorIndividual.push_back(pow(sErrorIndividual[predictionIndex]/double(numberOfProbeRatings),0.5));
    cout << "probe rmsErrorIndividual[" << predictionIndex << "] = " << endl << rmsErrorIndividual[predictionIndex] << endl;
  }
  
  cout << "probe rmsErrorBlend = " << rmsErrorBlend << endl;


  return 0;
}











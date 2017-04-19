// -*- C++ -*-

// this is only included to make "cout" work
#include <iostream>

// make sure to include baselinePrediction.h header file
#include "baselinePrediction.h"

using namespace std;

int main() {

  // choose random user, movie, and time 
  int user = 18767;
  int movie = 9869;
  int time = 1876;

  // calculate baseline prediction from user, movie, and time
  double prediction = baselinePrediction(user,movie,time);

  cout << "prediction = " << prediction << endl;
  
  return 0;
}











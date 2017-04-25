// -*- C++ -*-

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

int main() {

  // stream all necessary .dta files
  ifstream dataFile("base.dta");

  // these global values are taken from the base set
  int numberOfMovies = 17770;
  int numberOfUsers = 458293;
  double globalAverage = 3.60861;
  int numberOfGlobalRatings = 94362233;
  int maxTime = 2243;

  // this value is taken from BellKor paper
  int numberOfTimeBins = 30;

  // initialize variables for reading from base set
  int user;
  int movie;
  int time;
  int rating;
  int timeBin;

  // initialize vectors to store data from base set
  vector<int> users;
  vector<int> movies;
  vector<int> times;
  vector<int> ratings;
  vector<int> timeBins;

  // eventually, create ordered list with length of numberOfGlobalRatings for use in stochastic gradient descent
  vector<int> orderedList;

  // initialize vectors of variables that will be used for learning
  vector<double> userNumbersOfRatings(numberOfUsers);
  vector<double> movieNumbersOfRatings(numberOfMovies);
  vector<vector<int> > userTimes(numberOfUsers);
  vector<int> userNumbersOfTimes(numberOfUsers);
  vector<vector<int> > userNumbersOfRatingsPerTime(numberOfUsers);
  vector<double> userTimeSums(numberOfUsers);

  // initialize variables to help with data processing
  int userIndex;
  int movieIndex;
  int timeIndex;
  
  // read and process data point by point 
  for (int dataIndex=0; dataIndex<numberOfGlobalRatings; dataIndex++){
    dataFile >> user >> movie >> time >> rating; // read one line of data

    // fill in vectors to store data from base set
    users.push_back(user);
    movies.push_back(movie);
    times.push_back(time);
    ratings.push_back(rating);
    timeBins.push_back(int(ceil(double(time)*double(numberOfTimeBins)/double(maxTime))));
    orderedList.push_back(dataIndex);

    // calculate variables that will be used for learning
    userIndex = user-1; // specify userIndex (starts at zero)
    movieIndex = movie-1; // specify movieIndex (starts at zero)
    userNumbersOfRatings[userIndex] = userNumbersOfRatings[userIndex]+1; // increment number of ratings for user
    movieNumbersOfRatings[movieIndex] = movieNumbersOfRatings[movieIndex]+1; // increment number of ratings for movie
    if (find(userTimes[userIndex].begin(),userTimes[userIndex].end(),time) == userTimes[userIndex].end()){ // if the user has not yet rated a movie at this time (day)...
      userTimes[userIndex].push_back(time); // add time (day) to user's list of times (days)
      userNumbersOfTimes[userIndex] = userNumbersOfTimes[userIndex]+1; // increment the number of times (days) at which user has rated at least one movie
      userNumbersOfRatingsPerTime[userIndex].push_back(1); // initialize number of movies that user has rated at current time (day) to one.
    } else { // if the user has already rated a movie at this time (day)
      timeIndex = distance(userTimes[userIndex].begin(),find(userTimes[userIndex].begin(),userTimes[userIndex].end(),time)); // calculate timeIndex corresponding to the current time's place in userTimes
      userNumbersOfRatingsPerTime[userIndex][timeIndex] = userNumbersOfRatingsPerTime[userIndex][timeIndex]+1; // increment the number of movies that user has rated at current time (day)
    }
    userTimeSums[userIndex] = userTimeSums[userIndex]+double(time);
  }
  dataFile.close(); 

  // initialize average user time vector, which I'll calculate in next loop
  vector<double> userAverageTimes(numberOfUsers);

  // intialize vectors of user variables that will eventually be learned
  vector<double> userOffsets(numberOfUsers);
  vector<double> userMovieScalingsStable(numberOfUsers);
  vector<double> userTimeMultipliers(numberOfUsers);
  vector<vector<double> > userTimeOffsets(numberOfUsers);
  vector<vector<double> > userMovieScalingsTimed(numberOfUsers);


  // for every user...
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){

    // initialize userOffsets, usermovieScalingsStable, and userTimeMultipliers
    userOffsets[userIndex] = 0.;
    userMovieScalingsStable[userIndex] = 1.;
    userTimeMultipliers[userIndex] = 0.;

    // calculate average user time
    userAverageTimes[userIndex] = userTimeSums[userIndex]/double(userNumbersOfRatings[userIndex]);

    for (int timeIndex=0; timeIndex<userNumbersOfTimes[userIndex]; timeIndex++){
      // initialize UserTimeOffsets and userMovieScalingsTimed
      userTimeOffsets[userIndex].push_back(0.);
      userMovieScalingsTimed[userIndex].push_back(0.);
    }
  }

  // create an initial timeBinOffsets vector that will be learned for each movie
  vector<double> timeBinOffsets(numberOfTimeBins);
  for (int timeBinIndex=0; timeBinIndex<numberOfTimeBins; timeBinIndex++){
    timeBinOffsets[timeBinIndex] = 0.;    
  }

  // intialize vectors of movie variables that will eventually be learned
  vector<double> movieOffsets(numberOfMovies);
  vector<vector<double> > movieTimeBinOffsets;
  vector<vector<double> > movieFrequencyOffsets(numberOfMovies);


  // for each movie...
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){

    // initialize movieOffsets, movieTimeBinOffsets, and movieFrequenyOffsets
    movieOffsets[movieIndex] = 0.;
    movieTimeBinOffsets.push_back(timeBinOffsets);
    movieFrequencyOffsets[movieIndex].push_back(0.);
    movieFrequencyOffsets[movieIndex].push_back(0.);
    movieFrequencyOffsets[movieIndex].push_back(0.);
    movieFrequencyOffsets[movieIndex].push_back(0.);

  }

  // set learning rates and regularizers for all variables to be learned
  // these values are taken from BellKor paper
  double learningRateBu = 0.00267;
  double learningRateBi = 0.00048;
  double learningRateBiBin = 0.000115;
  double learningRateAu = 0.00000311;
  double learningRateBut = 0.00257;
  double learningRateCu = 0.00564;
  double learningRateCut = 0.00103;
  double learningRateBifui = 0.00236; 
  double regularizerBu = 0.0255;
  double regularizerBi = 0.0255;
  double regularizerBiBin = 0.0929;
  double regularizerAu = 3.95;
  double regularizerBut = 0.00231;
  double regularizerCu = 0.0476;
  double regularizerCut = 0.0190;
  double regularizerBifui = 0.000000011;

  // initialize variables for learning process
  int userNumberOfRatings;
  int movieNumberOfRatings;
  int userAverageTime;
  int timeBinNumberOfRatings;
  vector<int> userTimeList;
  int userNumberOfRatingsAtTime;  
  double userOffset;
  double movieOffset;
  double movieTimeBinOffset;
  double userTimeMultiplier;
  double userTimeOffset;
  double userMovieScalingStable;
  double userMovieScalingTimed;
  int frequency;
  double movieFrequencyOffset;
  double devU;

  int numberOfIterations = 10; // set number of iterations
  vector<int> shuffledList = orderedList; // initialized "shuffled list" of all the data points

  // for every iteration...
  for (int iteration=0; iteration<numberOfIterations; iteration++){

    cout << "iteration # = " << iteration << endl; // print out iteration # for monitoring
    random_shuffle(shuffledList.begin(), shuffledList.end()); // create new shuffled list of data points

    // for every data point in the shuffled list...
    for (int shuffleIndex=0; shuffleIndex<numberOfGlobalRatings; shuffleIndex++){
    //for (int shuffleIndex=0; shuffleIndex<1; shuffleIndex++){
     // cout << "checkpoint 1" << endl;
      int dataIndex = shuffleIndex;  
      //int dataIndex = shuffledList[shuffleIndex]; // extract "random" data point
      userIndex = users[dataIndex]-1; // specify userIndex (starts at zero)
      movieIndex = movies[dataIndex]-1; // specify movieIndex (starts at zero)
      time = times[dataIndex]; // extract rating time (day)
      rating = ratings[dataIndex]; // extract rating

     // cout << "checkpoint 2" << endl;  
      userNumberOfRatings = userNumbersOfRatings[userIndex]; // extract the number of ratings the user has made
      movieNumberOfRatings = movieNumbersOfRatings[movieIndex]; // extract the number of ratings the movie has receieved
      timeBinNumberOfRatings = int(ceil(double(movieNumberOfRatings)/double(numberOfTimeBins))); // calculate average number of ratings per time bin for the movie
      timeBin = timeBins[dataIndex]-1; // extract time bin
      userAverageTime = userAverageTimes[userIndex]; // extract the user's average time (day)
      userTimeList = userTimes[userIndex]; // extract list of times (days) on which the user has rated at least one movie
      timeIndex = distance(userTimeList.begin(),find(userTimeList.begin(),userTimeList.end(),time)); // extract timeIndex from userTimeList corresponding to current time
      userNumberOfRatingsAtTime = userNumbersOfRatingsPerTime[userIndex][timeIndex]; // extract the number of ratings the user has made at the given time

     // cout << "checkpoint 3" << endl;  
      userOffset = userOffsets[userIndex]; // get current userOffset
      movieOffset = movieOffsets[movieIndex]; // get current movieOffset
      movieTimeBinOffset = movieTimeBinOffsets[movieIndex][timeBin]; // get current movieTimeBinOffset
      userTimeMultiplier = userTimeMultipliers[userIndex]; // get current userTimeMultiplier
      userTimeOffset = userTimeOffsets[userIndex][timeIndex]; // get current userTimeOffset
      userMovieScalingStable = userMovieScalingsStable[userIndex]; // get current userMovieScalingStable
      userMovieScalingTimed = userMovieScalingsTimed[userIndex][timeIndex]; // get current userMovieScalingTimed

      // calculate "frequency" variable (see page 3 in the BellKor paper)
      if (userNumberOfRatingsAtTime < 7.){
        frequency = 0;
      } else{
        if (userNumberOfRatingsAtTime < 46.){
          frequency = 1;
        } else{
          if (userNumberOfRatingsAtTime < 309.){
            frequency = 2;
          } else{
              frequency = 3;
          }
        }
      }
      movieFrequencyOffset = movieFrequencyOffsets[movieIndex][frequency]; // get current movieFrequencyOffset

      // calculate "devU" variable (see Eq between Eq. 6 and Eq. 7 in the BellKor paper)
      if (double(time) < userAverageTime){
        devU = -1.*pow(fabs(double(time)-userAverageTime),0.4);
      } else {    
        devU = pow(fabs(double(time)-userAverageTime),0.4);
      }

      // use stochastic gradient descent to calculate new...

      //cout << "checkpoint 4" << endl;  
      // userOffset
      userOffset = userOffset-learningRateBu*2*((regularizerBu/double(userNumberOfRatings)+1.)*userOffset+globalAverage
                   +(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                   +userTimeMultiplier*devU+userTimeOffset+movieFrequencyOffset-double(rating));
      
      // movieOffset
      movieOffset = movieOffset-learningRateBi*2*(userMovieScalingStable+userMovieScalingTimed)*
                    ((regularizerBi/double(movieNumberOfRatings)+userMovieScalingStable+userMovieScalingTimed)*movieOffset+globalAverage+userOffset
                    +movieTimeBinOffset*(userMovieScalingStable+userMovieScalingTimed)
                    +userTimeMultiplier*devU+userTimeOffset+movieFrequencyOffset-double(rating));
     
      // movieTimeBinOffset
      movieTimeBinOffset = movieTimeBinOffset-learningRateBiBin*2*(userMovieScalingStable+userMovieScalingTimed)*
                      ((regularizerBiBin/double(timeBinNumberOfRatings)+userMovieScalingStable+userMovieScalingTimed)*movieTimeBinOffset+globalAverage+userOffset
                      +movieOffset*(userMovieScalingStable+userMovieScalingTimed)
                      +userTimeMultiplier*devU+userTimeOffset+movieFrequencyOffset-double(rating));
     
      // userTimeMultiplier
      userTimeMultiplier = userTimeMultiplier-learningRateAu*2*devU*(regularizerAu/double(userNumberOfRatingsAtTime)*userTimeMultiplier
                           +userTimeMultiplier*devU+globalAverage
                           +userOffset+(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                           +userTimeOffset+movieFrequencyOffset-double(rating));
     
      // userTimeOffset
      userTimeOffset = userTimeOffset-learningRateBut*2*((regularizerBut/double(40*numberOfUsers)+1.)*userTimeOffset+globalAverage+userOffset
                       +(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                       +userTimeMultiplier*devU+movieFrequencyOffset-double(rating));
      
      // userMovieScalingStable
      userMovieScalingStable = userMovieScalingStable-learningRateCu*2*(movieOffset+movieTimeBinOffset)*((regularizerCu/double(userNumberOfRatings))
                              *(userMovieScalingStable-1.)
                              +globalAverage+userOffset+(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                              +userTimeMultiplier*devU+userTimeOffset+movieFrequencyOffset-double(rating));
     
      // userMovieScalingTimed
      userMovieScalingTimed = userMovieScalingTimed-learningRateCut*2*(movieOffset+movieTimeBinOffset)*((regularizerCut/double(userNumberOfRatingsAtTime))
                              *userMovieScalingTimed
                              +globalAverage+userOffset+(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                              +userTimeMultiplier*devU+userTimeOffset+movieFrequencyOffset-double(rating));
     
      // movieFrequencyOffset
      movieFrequencyOffset = movieFrequencyOffset-learningRateBifui*2*((regularizerBifui/double(4*movieNumberOfRatings)+1.)*movieFrequencyOffset
                             +globalAverage+userOffset+(movieOffset+movieTimeBinOffset)*(userMovieScalingStable+userMovieScalingTimed)
                             +userTimeMultiplier*devU+userTimeOffset-double(rating));
      
      // insert newly-calculated learned variables back into vectors for storage
      userOffsets[userIndex] = userOffset;
      movieOffsets[movieIndex] = movieOffset;
      movieTimeBinOffsets[movieIndex][timeBin] = movieTimeBinOffset;
      userTimeMultipliers[userIndex] = userTimeMultiplier;
      userTimeOffsets[userIndex][timeIndex] = userTimeOffset;
      userMovieScalingsStable[userIndex] = userMovieScalingStable;
      userMovieScalingsTimed[userIndex][timeIndex] = userMovieScalingTimed;
      movieFrequencyOffsets[movieIndex][frequency] = movieFrequencyOffset;

    }
  }


  // write all of the learned variables to their respective .dta files

  ofstream movieOffsetsFile;
  movieOffsetsFile.open("movieOffsets.dta");
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){
    movieOffsetsFile << movieOffsets[movieIndex] << endl;
  }
  movieOffsetsFile.close();

  ofstream userOffsetsFile;
  userOffsetsFile.open("userOffsets.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userOffsetsFile << userOffsets[userIndex] << endl;
  }
  userOffsetsFile.close();

  ofstream movieTimeBinOffsetsFile;
  movieTimeBinOffsetsFile.open("movieTimeBinOffsets.dta");
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){
    for (int timeBinIndex=0; timeBinIndex<numberOfTimeBins; timeBinIndex++){
      movieTimeBinOffsetsFile << movieTimeBinOffsets[movieIndex][timeBinIndex] << " ";
    }
    movieTimeBinOffsetsFile << endl;
  }
  movieTimeBinOffsetsFile.close();

  ofstream userTimeMultipliersFile;
  userTimeMultipliersFile.open("userTimeMultipliers.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userTimeMultipliersFile << userTimeMultipliers[userIndex] << endl;
  }
  userTimeMultipliersFile.close();

  ofstream userTimeOffsetsFile;
  userTimeOffsetsFile.open("userTimeOffsets.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    for (int timeIndex=0; timeIndex<userNumbersOfTimes[userIndex]; timeIndex++){
      userTimeOffsetsFile << userTimeOffsets[userIndex][timeIndex] << " ";
    }
    userTimeOffsetsFile << endl;
  }
  userTimeOffsetsFile.close();

  ofstream userMovieScalingsStableFile;
  userMovieScalingsStableFile.open("userMovieScalingsStable.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userMovieScalingsStableFile << userMovieScalingsStable[userIndex] << endl;
  }
  userMovieScalingsStableFile.close();

  ofstream userMovieScalingsTimedFile;
  userMovieScalingsTimedFile.open("userMovieScalingsTimed.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    for (int timeIndex=0; timeIndex<userNumbersOfTimes[userIndex]; timeIndex++){
      userMovieScalingsTimedFile << userMovieScalingsTimed[userIndex][timeIndex] << " ";
    }
    userMovieScalingsTimedFile << endl;
  }

  ofstream movieFrequencyOffsetsFile;
  movieFrequencyOffsetsFile.open("movieFrequencyOffsets.dta");
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){
    movieFrequencyOffsetsFile << movieFrequencyOffsets[movieIndex][0] << " "; 
    movieFrequencyOffsetsFile << movieFrequencyOffsets[movieIndex][1] << " "; 
    movieFrequencyOffsetsFile << movieFrequencyOffsets[movieIndex][2] << " "; 
    movieFrequencyOffsetsFile << movieFrequencyOffsets[movieIndex][3] << " "; 
    movieFrequencyOffsetsFile << endl;
  }
  movieOffsetsFile.close();

  // also write two other useful .dta files
  ofstream userNumbersOfTimesFile;
  ofstream userTimesFile;
  userNumbersOfTimesFile.open("userNumberOfTimes.dta");
  userTimesFile.open("userTimes.dta");
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){ 
    userNumbersOfTimesFile << userNumbersOfTimes[userIndex] << endl;
    for (int timeIndex=0; timeIndex<userNumbersOfTimes[userIndex]; timeIndex++){
      userTimesFile << userTimes[userIndex][timeIndex] << " ";
    }
    userTimesFile << endl;
  }
  userNumbersOfTimesFile.close();
  userTimesFile.close();  


  return 0;
}











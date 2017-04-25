// "fstream" allows for streaming of data files
#include<fstream>

// "vector" provides functionality for vector<> container
#include<vector>

#include<algorithm>
#include<math.h>

using namespace std;

// First, there's a list of functions that uploads all the necessary
// .dta files in order to make the baseline prediction

// get userOffsets
vector<double>
getUserOffsets(){

  int numberOfUsers = 458293;
  ifstream userOffsetFile("userOffsets.dta");
  vector<double> userOffsets(numberOfUsers);

  double userOffset;
  int user = 0;
  while (userOffsetFile >> userOffset){
    userOffsets[user] = userOffset;
    user = user+1;
  }  

  userOffsetFile.close();
  return userOffsets;

}

// get movieOffsets
vector<double>
getMovieOffsets(){

  int numberOfMovies = 17770;
  ifstream movieOffsetFile("movieOffsets.dta");
  vector<double> movieOffsets(numberOfMovies);

  double movieOffset;
  int movie = 0;
  while (movieOffsetFile >> movieOffset){
    movieOffsets[movie] = movieOffset;
    movie = movie+1;
  }
  
  movieOffsetFile.close();
  return movieOffsets;

}

// get movieTimeBinOffsets
vector<vector<double> >
getMovieTimeBinOffsets(){
  
  int numberOfMovies = 17770;
  int numberOfTimeBins = 30;
  ifstream movieTimeBinOffsetFile("movieTimeBinOffsets.dta");
  vector<double> movieTimeBinOffset(numberOfTimeBins);
  vector<vector<double> > movieTimeBinOffsets;

  // move average movie ratings from data file to movieRatings vector
  double movieOffset;
  int movie = 0;
  double movieTimeOffset;
  string dataLine;
  int stringIndex = 0;
  int stringLength = 0;
  int timeBinIndex=0;
  string entry;
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){
    for (int timeBinIndex=0; timeBinIndex<numberOfTimeBins; timeBinIndex++){
      movieTimeBinOffsetFile >> movieTimeOffset;
      movieTimeBinOffset[timeBinIndex] = movieTimeOffset;
    }
    movieTimeBinOffsets.push_back(movieTimeBinOffset);
  }

  movieTimeBinOffsetFile.close();
  return movieTimeBinOffsets;

}

// get userAverageTimes
vector<double>
getUserAverageTimes(){

  int numberOfUsers = 458293;
  ifstream userAverageTimeFile("userAverageTimes.dta");
  vector<double> userAverageTimes(numberOfUsers);
  double userAverageTime;
  int user = 0;
  while (userAverageTimeFile >> userAverageTime){
    userAverageTimes[user] = userAverageTime;
    user = user+1;
  }  
  
  userAverageTimeFile.close();
  return userAverageTimes;

}

// get userTimeMultipliers
vector<double>
getUserTimeMultipliers(){

  int numberOfUsers = 458293;
  ifstream userTimeMultipliersFile("userTimeMultipliers.dta");
  vector<double> userTimeMultipliers(numberOfUsers);

  // move user time multipliers from data file to userTimeMultipliers vector
  double userTimeMultiplier;
  int user = 0;
  while (userTimeMultipliersFile >> userTimeMultiplier){
    userTimeMultipliers[user] = userTimeMultiplier;
    user = user+1;
  }  

  userTimeMultipliersFile.close();
  return userTimeMultipliers;

}

// get userTimeOffsets
vector<vector<double> >
getUserTimeOffsets(){

  int numberOfUsers = 458293;
  ifstream userTimeOffsetsFile("userTimeOffsets.dta");
  ifstream userNumberOfTimesFile("userNumberOfTimes.dta");

  vector<vector<double> > userTimeOffsets(numberOfUsers);

  int userNumberOfTime;
  double userTimeOffset;
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userNumberOfTimesFile >> userNumberOfTime;
    for (int timeIndex=0; timeIndex<userNumberOfTime; timeIndex++){
      userTimeOffsetsFile >> userTimeOffset;
      userTimeOffsets[userIndex].push_back(userTimeOffset);
    }
  }

  return userTimeOffsets;

}

// get userTimes
vector<vector<int> >
getUserTimes(){

  int numberOfUsers = 458293;
  ifstream userTimesFile("userTimes.dta");
  ifstream userNumberOfTimesFile("userNumberOfTimes.dta");

  vector<vector<int> > userTimes(numberOfUsers);

  int userNumberOfTime;
  int userTime;
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userNumberOfTimesFile >> userNumberOfTime;
    for (int timeIndex=0; timeIndex<userNumberOfTime; timeIndex++){
      userTimesFile >> userTime;
      userTimes[userIndex].push_back(userTime);
    }
  }

  return userTimes;

}

// get userMovieScalingsStable
vector<double>
getUserMovieScalingsStable(){

  int numberOfUsers = 458293;
  ifstream userMovieScalingsStableFile("userMovieScalingsStable.dta");
  vector<double> userMovieScalingsStable(numberOfUsers);

  // move user offsets from data file to userOffsets vector
  double userMovieScalingStable;
  int user = 0;
  while (userMovieScalingsStableFile >> userMovieScalingStable){
    userMovieScalingsStable[user] = userMovieScalingStable;
    user = user+1;
  }  

  userMovieScalingsStableFile.close();
  return userMovieScalingsStable;

}

// get userMovieScalingsTimed
vector<vector<double> >
getUserMovieScalingsTimed(){

  int numberOfUsers = 458293;
  ifstream userMovieScalingsTimedFile("userMovieScalingsTimed.dta");
  ifstream userNumberOfTimesFile("userNumberOfTimes.dta");

  vector<vector<double> > userMovieScalingsTimed(numberOfUsers);

  int userNumberOfTime;
  double userMovieScalingTimed;
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userNumberOfTimesFile >> userNumberOfTime;
    for (int timeIndex=0; timeIndex<userNumberOfTime; timeIndex++){
      userMovieScalingsTimedFile >> userMovieScalingTimed;
      userMovieScalingsTimed[userIndex].push_back(userMovieScalingTimed);
    }
  }

  return userMovieScalingsTimed;

}

// get movieFrequencyOffsets
vector<vector<double> >
getMovieFrequencyOffsets(){

  int numberOfMovies = 17770;
  int numberOfFrequencies = 4;
  ifstream movieFrequencyOffsetsFile("movieFrequencyOffsets.dta");

  double movieFrequencyOffset;

  vector<vector<double> > movieFrequencyOffsets(numberOfMovies);
  for (int movieIndex=0; movieIndex<numberOfMovies; movieIndex++){
    for (int frequencyIndex=0; frequencyIndex<numberOfFrequencies; frequencyIndex++){
      movieFrequencyOffsetsFile >> movieFrequencyOffset;
      movieFrequencyOffsets[movieIndex].push_back(movieFrequencyOffset);
    }
  }
  movieFrequencyOffsetsFile.close();
  return movieFrequencyOffsets;
}


// get userNumberOfRatingsPerTime
vector<vector<int> >
getUserNumberOfRatingsPerTime(){

  int numberOfUsers = 458293;
  ifstream userNumberOfRatingsPerTimeFile("userNumberOfRatingsPerTime.dta");
  ifstream userNumberOfTimesFile("userNumberOfTimes.dta");

  vector<vector<int> > userNumberOfRatingsPerTime(numberOfUsers);

  int userNumberOfTime;
  int userNumberOfRatings;
  for (int userIndex=0; userIndex<numberOfUsers; userIndex++){
    userNumberOfTimesFile >> userNumberOfTime;
    for (int timeIndex=0; timeIndex<userNumberOfTime; timeIndex++){
      userNumberOfRatingsPerTimeFile >> userNumberOfRatings;
      userNumberOfRatingsPerTime[userIndex].push_back(userNumberOfRatings);
    }
  }

  return userNumberOfRatingsPerTime;

}


// get all the above functions to get and store the necessary data
vector<double> userOffsets = getUserOffsets();
vector<double> movieOffsets = getMovieOffsets();
vector<vector<double> > movieTimeBinOffsets = getMovieTimeBinOffsets();
vector<double> userAverageTimes = getUserAverageTimes();
vector<double> userTimeMultipliers = getUserTimeMultipliers();
vector<vector<double> > userTimeOffsets = getUserTimeOffsets();
vector<vector<int> > userTimes = getUserTimes();
vector<double> userMovieScalingsStable = getUserMovieScalingsStable();
vector<vector<double> > userMovieScalingsTimed = getUserMovieScalingsTimed();
vector<vector<double> > movieFrequencyOffsets = getMovieFrequencyOffsets();
vector<vector<int> > userNumberOfRatingsPerTime = getUserNumberOfRatingsPerTime();

// instantiate globalAverage as global variable
double globalAverage = 3.60861;

// baselinePrediction(user,movie,time) returns the baseline prediction
double
baselinePrediction(int user,int movie,int time){
  
  int maxTime = 2243;

  // get the number of time bins
  int numberOfTimeBins = movieTimeBinOffsets[0].size();

  // calculate the time bin
  int timeBin = int(ceil(double(time)*double(numberOfTimeBins)/double(maxTime)));

  // find time index for userTimeOffset
  vector<int> userTime = userTimes[user-1];
  int timeIndex = distance(userTime.begin(),find(userTime.begin(),userTime.end(),time)); 

  // calculate "frequency" variable
  int frequency;
  int userNumberOfRatingsAtTime = userNumberOfRatingsPerTime[user-1][timeIndex];
  if (userNumberOfRatingsAtTime < 7){
    frequency = 0;
  } else{
    if (userNumberOfRatingsAtTime < 46){
      frequency = 1;
    } else{
      if (userNumberOfRatingsAtTime < 309){
        frequency = 2;
      } else{
          frequency = 3;
      }
    }
  }


  // extract all the relevant learned offsets/multipliers based on input
  double userOffset = userOffsets[user-1];
  double movieOffset = movieOffsets[movie-1];
  double movieTimeBinOffset = movieTimeBinOffsets[movie-1][timeBin-1];
  double userAverageTime = userAverageTimes[user-1];
  double userTimeMultiplier = userTimeMultipliers[user-1];
  double userTimeOffset = userTimeOffsets[user-1][timeIndex];
  double userMovieScalingStable = userMovieScalingsStable[user-1];
  double userMovieScalingTimed = userMovieScalingsTimed[user-1][timeIndex];
  double movieFrequencyOffset = movieFrequencyOffsets[movie-1][frequency];

  // set userTimeOffset = 0 and userMovieScalingTimed =0 if input time doesn't exist in training
  if (userTime[timeIndex] != time){
    userTimeOffset = 0;
    userMovieScalingTimed = 0;
  }

  // calculate devU
  double devU;
  if (double(time) < userAverageTime){
    devU = -1.*pow(fabs(double(time)-userAverageTime),0.4);
  } else {
    devU = pow(fabs(double(time)-userAverageTime),0.4);
  }

  // calculate  and return prediction
  double prediction = globalAverage+userOffset
                      +(movieOffset+movieTimeBinOffset)
                      *(userMovieScalingStable+userMovieScalingTimed)
                      +userTimeMultiplier*devU+userTimeOffset
                      +movieFrequencyOffset;
  return prediction;


}



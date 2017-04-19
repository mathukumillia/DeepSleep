// "fstream" allows for streaming of data files
#include<fstream>

// "vector" provides functionality for vector<> container
#include<vector>

using namespace std;

// getBaselineData() loads previously-computed userAverageOffsets and movieAverageRatings
// and stores into global variable baselineData
vector<vector<double> >
getBaselineData(){

  vector<vector<double> > baselineData;

  // hard-code number of users and movies
  int numberOfUsers = 458293;
  int numberOfMovies = 17770;

  // stream data files
  ifstream userOffsetFile("userAverageOffsets.dta");
  ifstream movieRatingFile("movieAverageRatings.dta");
  
  // intialize vectors that will contain user offsets and movie ratings
  vector<double> userOffsets(numberOfUsers);
  vector<double> movieRatings(numberOfMovies);

  // move user offsets from data file to userOffsets vector
  double offset;
  int user = 0;
  while (userOffsetFile >> offset){
    userOffsets[user] = offset;
    user = user+1;
  }  
  
  // move average movie ratings from data file to movieRatings vector
  double averageRating;
  int movie = 0;
  while (movieRatingFile >> averageRating){
    movieRatings[movie] = averageRating;
    movie = movie+1;
  }
  
  // close data files
  userOffsetFile.close();
  movieRatingFile.close();

  // store userOffsets and movieRatings vector into a single variable baselineData
  baselineData.push_back(userOffsets);
  baselineData.push_back(movieRatings);

  return baselineData;

}

// instantiate baselineData as global variable
vector<vector<double> > baselineData = getBaselineData();

// baselinePrediction(user,movie,time) returns the baseline prediction
double
baselinePrediction(int user,int movie,int time){
  
  // for now, ignore time
  (void)time;

  // extract userOffset and movieRating from baselineData
  double userOffset = baselineData[0][user-1];
  double movieRating = baselineData[1][movie-1];

  // calculate  and return prediction
  double prediction = movieRating + userOffset;
  return prediction;

}



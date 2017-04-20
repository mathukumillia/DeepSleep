#include<iostream>
#include<fstream>
#include<string>
#include<numeric>
using namespace std;

// these are all the global variables used by the program

// users and movies are one indexed
const double numUsers = 458293;
const double numMovies = 17770;

// K is the constant representing the number of features
// lrate is the learning rate
const double K = 5;
const double lrate = 0.001;

double **userValues;
double **movieValues;


double predictRating(int user, int movie)
{
  return inner_product(userValues[user], userValues[user] + (int)K, movieValues[movie], 0.0);
}


double error ()
{

  fstream inputFile ("../um/all.dta");
  fstream indexFile ("../um/all.idx");
  double inputs [4] = {};

  //  counter keeps track of the number of points we've been through
  int counter = 0;
  double error = 0;
  double diff;
  string index;
  if (inputFile.is_open() && indexFile.is_open())
  {
    while (inputFile >> inputs[0] >> inputs[1] >> inputs[2] >> inputs[3])
    {
      getline(indexFile, index);
      if (atoi(index.c_str()) == 1)
      {
        cout << "Getting error for point " << counter << "\n";
        diff = inputs[2] - predictRating(inputs[0], inputs[1]);
        error += diff * diff;
      }
      counter++;
    }
  }
  inputFile.close();
  indexFile.close();
  return error;
}

void train(int user, int movie, int rating)
{
  // calculate the error with the current feature values
	double err = lrate * ((double) rating - predictRating(user, movie));

  double *uv = userValues[user];
  for (int i = 0; i < K; i++){
    userValues[user][i] += err * movieValues[movie][i];
    movieValues[movie][i] += err * uv[i];
  }
}

void initialize()
{
  // allocate and initialize the userValues and movieValues matrices
  userValues = new double*[numUsers];
  for (int i = 0; i < numUsers; i++)
  {
    userValues[i] = new double[K];
    fill(&userValues[0][0], &userValues[0][0] + sizeof(userValues), 0.1);
  }

  movieValues = new double*[numMovies];
  for (int i = 0; i < numMovies; i++)
  {
    movieValues[i] = new double[K];
    fill(&movieValues[0][0], &movieValues[0][0] + sizeof(movieValues), 0.1);
  }
}

void cleanUp()
{
  // de-allocate user values and movie values to prevent memory leak
  for (int i = 0; i < numUsers; i++)
  {
    delete [] userValues[i];
  }
  delete [] userValues;
  for (int i = 0; i < numMovies; i++)
  {
    delete [] movieValues[i];
  }
  delete [] movieValues;
}

void runEpoch ()
{
  // read in the training data and train on each input
  fstream inputFile ("../um/all.dta");
  fstream indexFile ("../um/all.idx");
  double inputs [4] = {};

  //  counter keeps track of the number of points we've been through
  int counter = 0;
  string index;
  if (inputFile.is_open() && indexFile.is_open())
  {
    while (inputFile >> inputs[0] >> inputs[1] >> inputs[2] >> inputs[3])
    {
      getline(indexFile, index);
      if (atoi(index.c_str()) == 1)
      {
        cout << "Processing point " << counter << "\n";
        train(inputs[0], inputs[1], inputs[3]);
      }
      counter++;
    }
  }
  inputFile.close();
  indexFile.close();
}

// Opens the file and runs the SVD
int main()
{
  initialize();
  // gets the initial validation error
  // NOTE: the initial error should calculate the error in the final program
  double initialError = 1;
  double finalError = 10;
  cout << "Initial Error is: " << initialError << "\n";
  cout << "Final Error is: " << finalError << "\n";
  // NOTE: Uncomment the while loop once we've optimized the single epoch stuff
  // while (initialError/finalError > 0.01) {
    initialError = finalError;
    cout << "Running Epoch..." << "\n";
    runEpoch();
    finalError = error();
    cout << "Initial Error is: " << initialError << "\n";
    cout << "Final Error is: " << finalError << "\n";
  // }
  cleanUp();
  return 0;
}

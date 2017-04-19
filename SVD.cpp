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
  // read in the training data and train on each input
  fstream inputFile ("../um/all.dta");
  double inputs [4] = {};
  double temp;
  int num = 0;
  int counter = 0;
  double error = 0;
  double diff;
  if (inputFile.is_open())
  {
    while (inputFile >> temp)
    {
      // the while loop reads in number by number
      // so we store each point as it passes through the program
      // in inputs
      inputs[num] = temp;
      if (num % 3 == 0 && num != 0)
      {
        // if we have a full point in inputs and the point is part of the
        // validation set, validate the SVD with it
        if (inputs[3] == 2)
        {
          cout << "Getting error for point " << counter << "\n";
          diff = inputs[2] - predictRating(inputs[0], inputs[1]);
          error += diff * diff;
        }
        num = 0;
      }
      else
      {
        num++;
      }
      counter++;
    }
  }
  inputFile.close();
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
  double inputs [4] = {};

  //  counter keeps track of the number of points we've been through
  int counter = 0;
  double temp;
  int num = 0;
  if (inputFile.is_open())
  {
    while (inputFile >> temp)
    {
      // the while loop reads in number by number
      // so we store each point as it passes through the program
      // in inputs
      inputs[num] = temp;
      if (num % 3 == 0 && num != 0)
      {
        // if we have a full point in inputs and the point is part of the
        // training set, train the SVD with it
        if (inputs[3] == 1)
        {
         train(inputs[0], inputs[1], inputs[2]);
         cout << "Processing point " << counter << "\n";
        }
        num = 0;
      }
      else
      {
        num++;
      }
      counter++;
    }
  }
  inputFile.close();
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

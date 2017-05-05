#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
using namespace std;

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 100

// these are all the global variables used by the program

// users and movies are one indexed
const int num_users = 458293;
const int num_movies = 17770;
const int num_pts = 102416306;

// K is the constant representing the number of features
// lrate is the learning rate
const int K = 20;
const double lrate = 0.001;

// these 2D arrays are the U and V in the SVD
// they are one dimensional arrays in memory to make access quicker
double *user_values;
int user_values_size = (num_users + 1) * K;
double *movie_values;
int movie_values_size = (num_movies + 1) * K;

// 2D arrays to store ratings input
// again, it's a one dimensional array in memory to make access quicker
int *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
// stores the indices of each point, in the same order as the ratings
int *indices;

/*
* Allocates memory and initializes user, movie, ratings, and indices arrays
*/
void initialize()
{
    cout << "Initializing the program.\n";

    // allocate and initialize the user_values and movie_values matrices
    // all of the +1 terms result from the fact that these arrays are 1
    // indexed in the data
    user_values = new double[user_values_size];
    for (int i = 0; i < user_values_size; i++)
    {
        user_values[i] = 0.1; // arbitrary initial condition
    }

    movie_values = new double[movie_values_size];
    for (int i = 0; i < movie_values_size; i++)
    {
        movie_values[i] = 0.1;
    }

    // create  the arrays that store the ratings input data and the indexes
    ratings = new int[ratings_size];
    indices = new int[num_pts];


    cout << "Done allocating memory.\n";
}

/*
* Clears all used memory
*/
void clean_up()
{
    cout << "Cleaning up.\n";
    delete [] user_values;
    delete [] movie_values;
    delete [] ratings;
    delete [] indices;
}

/*
* Reads the input data into ratings and indices
*/
void read_data()
{
    cout << "Reading in training data.\n";

    // read in ratings data - currently, this is training without the baseline
    fstream ratings_file("../ratings_baseline_removed.bin", ios::in | ios::binary);
    ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(int) * num_pts * POINT_SIZE);
    ratings_file.close();

    // read in index data
    fstream indices_file("../indices.bin", ios::in | ios::binary);
    indices_file.read(reinterpret_cast<char *>(indices), sizeof(int) * num_pts);
    indices_file.close();
}

/*
* Given a user and a movie, this function gives the predicted rating
*/
double predict_rating(int user, int movie)
{
  double rating = 0;
  for (int i = 0; i < K; i++)
  {
    rating += user_values[user * K + i] * movie_values[movie * K + i];
  }
  return rating;
}

/*
* Gets the total error of the SVD model on the set index provided
* i.e., to get validation error, pass in set = 2
*/
double error(int set)
{
    cout << "Calculating error.\n";

    double error = 0;
    double diff = 0;
    int index;
    double points_in_set = 0;

    for (int i = 0; i < num_pts; i++) {
        index = indices[i];

        if (index == set) {
            diff = (double)(ratings[i * POINT_SIZE + 3]) - 
            	predict_rating(ratings[i * POINT_SIZE], ratings[i * POINT_SIZE + 1]);
            error += diff * diff;
            points_in_set += 1;
        }
    }

    return sqrt(error/points_in_set);
}

/*
* Runs SGD on a single point
*/
void train(int user, int movie, int rating)
{

  	// calculate the error with the current feature values
	double err = lrate * ((double) rating - predict_rating(user, movie));

	// updates the movie and user vectors feature by feature
	double uv;
	for (int i = 0; i < K; i++){
		uv = user_values[user * K + i];
		user_values[user * K + i] += err * movie_values[movie * K + i];
		movie_values[movie * K + i] += err * uv;
	}
}

/*
* Run a full epoch.
*/
void run_epoch ()
{
	cout << "Running Epoch." << "\n";
	for (int i = 0; i < num_pts; i++) {
		int index = indices[i];

		// uses point set 1 to train
		if (index == 1) {
			train(ratings[i + POINT_SIZE], ratings[i * POINT_SIZE + 1], ratings[i * POINT_SIZE + 3]);
		}
	}
	cout << "Epoch complete." << "\n";
}

/*
* Predicts ratings on the qual set and writes them to a file.
*/
void find_qual_predictions()
{
	ofstream outputFile;
	outputFile.open("naive_SVD_output.dta");

	int index;
	double prediction;

	for(int i = 0; i < num_pts; i++)
	{
	    index = indices[i];
	    // the qual set is set 5
	    if (index == 5)
	    {
	        prediction = (double)(-1 * ratings[i * POINT_SIZE + 3]) + 
	        	predict_rating(ratings[i * POINT_SIZE], ratings[i * POINT_SIZE + 1]);
	        if (prediction < 1)
	        {
	            prediction = 1;
	        }
	        if (prediction > 5)
	        {
	            prediction = 5;
	        }
	        outputFile << prediction << "\n";
	    }
	}
}

int main()
{
    initialize();
    read_data();

    double initialError = 10;
    double finalError = error(2); // gets the validation error before training
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";
    while (initialError - finalError > STOPPING_CONDITION && counter <= MAX_EPOCHS) {
        cout << "Starting Epoch " << counter << "\n";
        counter++;
        initialError = finalError;
        run_epoch();
        finalError = error(2); // error(2) returns the validation error
        cout << "Error after Epoch " << counter << ": " << finalError << "\n";
    }

    // find the values on the qual set
    find_qual_predictions();

    clean_up();
    return 0;
}
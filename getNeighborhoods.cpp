/*
* gets the neighborhood of each user, stores it in binary file
*
*/
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
#include <algorithm>

using namespace std;

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define MAX_NEIGHBOR_SIZE 300 // the largest any possible neighborhood can be

// users and movies are one indexed
const double num_users = 458293;
const double num_pts = 102416306;

// 2D array that stores the neighborhood of each user 
// the neighborhood is just all of the movies for which the user has given
// a rating
// I use a 1D array to make it more time efficient
int *neighborhoods;
int neighborhoods_size = (int) ((num_users + 1) * MAX_NEIGHBOR_SIZE);
// array stores how many movies are in the neighborhood of each user
// essentially store |N(u)|
double *neighborhood_sizes;

// the arrays to store the ratings and indices data
// note that ratings will be used as a 2D array as well
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
double *indices;

void initialize ()
{
	cout << "Initializing.\n";
	// create the arrays that store the ratings input data and the indexes
	ratings = new double[ratings_size];
	indices = new double[((int) num_pts)];
	neighborhoods = new int[neighborhoods_size];
	neighborhood_sizes = new double[(int)(num_users + 1)];
	// initializes all of the sizes to 0
	for (int i = 0; i < num_users + 1; i++)
	{
		neighborhood_sizes[i] = 0;
	}
}

/*
* Reads the input data into ratings and indices
*/
void read_data()
{
	cout << "Reading in training data.\n";
	// read in ratings data
	fstream ratings_file("../ratings.bin", ios::in | ios::binary);
	ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * num_pts * POINT_SIZE);
	ratings_file.close();

	// read in index data
	fstream indices_file("../indices.bin", ios::in | ios::binary);
	indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * num_pts);
	indices_file.close();
}

/*
*
* Read through all of the points and assemble the neighborhoods of each
* user
* limits neighborhood size to 300, as dictated by paper
*/
void assemble_neighborhoods()
{
	cout << "Assembling neighborhoods.\n";
	double index;
	double user;
	double movie;
	for (int i = 0; i < num_pts; i++)
	{
		index = indices[i];
		// currently only assembles neighborhoods from a provided set
		// limit the size of the neighborhood to 300, just like the paper does
		if (index == 1 || index == 2 || index == 3 || index == 4 || index == 5)
		{
			user = ratings[i * POINT_SIZE];
			movie = ratings[i * POINT_SIZE + 1];
			if (neighborhood_sizes[(int)user] < MAX_NEIGHBOR_SIZE)
			{
				neighborhoods[(int)user * MAX_NEIGHBOR_SIZE + (int)neighborhood_sizes[(int)user]] = (int)movie;
				neighborhood_sizes[(int)user] += 1;
			}	
		}
	}
}

/*
* Writes the neighborhoods and neighborhood sizes to binary files
*
*/
void write_neighborhoods()
{
	cout << "Writing to binary.\n";
	fstream neighborhoods_file ("neighborhoods_12345.bin", ios::out | ios::binary);
	fstream neighborhood_sizes_file ("neighborhood_sizes_12345.bin", ios::out | ios::binary);
	neighborhoods_file.write(reinterpret_cast<char *>(neighborhoods), sizeof(double) * (num_users + 1) * MAX_NEIGHBOR_SIZE);
	neighborhood_sizes_file.write(reinterpret_cast<char *>(neighborhood_sizes), sizeof(double) * (num_users + 1));
	neighborhoods_file.close();
	neighborhood_sizes_file.close();
}

/*
* Deallocate allocated memory
*/
void clean_up()
{
	delete [] ratings;
	delete [] indices;
	delete [] neighborhoods;
	delete [] neighborhood_sizes;
}

int main()
{
	initialize();
	read_data();
	assemble_neighborhoods();
	write_neighborhoods();
	clean_up();
	return 0;
}
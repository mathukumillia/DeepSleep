#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <math.h>

using namespace std;

#define POINT_SIZE 4 // the size of a single input point in the training data

// users and movies are one indexed
const double num_users = 458293;
const double num_movies = 17770;
const double num_pts = 102416306;

// the arrays to store the ratings and indices data
// note that ratings will be used as a 2D array as well
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
double *indices;

// the array that stores the mean time of rating for each user
double * t_u;

void initialize()
{
	ratings = new double[ratings_size];
	indices = new double[((int) num_pts)];
	t_u = new double[(int)num_users + 1];
}

void clean_up()
{
	delete [] ratings;
	delete [] indices;
	delete [] t_u;
}

/*
* Reads the input data into ratings and indices
*/
inline void read_data()
{
    cout << "Reading in data.\n";
    // read in ratings data - currently, this is training without the baseline
    fstream ratings_file("../ratings_baseline_removed.bin", ios::in | ios::binary);
    ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * num_pts * POINT_SIZE);
    ratings_file.close();

    // read in index data
    fstream indices_file("../indices.bin", ios::in | ios::binary);
    indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * num_pts);
    indices_file.close();
}


int main()
{
	initialize();
	read_data();

	double prev_user = 1;
	double date_sum = 0; 
	double num_dates = 0;
	for(int i = 0; i < num_pts; i++)
	{
		if (indices[i] == 1)
		{
			if (ratings[i * POINT_SIZE] != prev_user)
			{
				t_u[(int)prev_user] = date_sum/num_dates;
				cout << "t_u: " << t_u[(int)prev_user] << "\n";
				prev_user = ratings[i * POINT_SIZE];
				date_sum = 0;
				num_dates = 0;
			}
			date_sum += ratings[i * POINT_SIZE + 2];
			num_dates++;
		}
	}

	// write the found average dates to a binary file
	fstream outputFile ("average_rating_time.bin", ios::out | ios::binary);

	outputFile.write(reinterpret_cast<char *>(t_u), sizeof(double) * (num_users + 1));

	outputFile.close();

	return 0;
}
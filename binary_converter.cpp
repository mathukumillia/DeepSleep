/*
* binary_converter
*	takes the .dta files and outputs a binary file containing array data
* to use this: 
* 1) download .bin files
* 2) allocate memory for the ratings and indices data (note that ratings 
* has to be a 1D array, not a 2D array)
* 3) use the following lines: 
*
*	 	fstream ratings_file("ratings.bin", ios::in | ios::binary);
*		ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * numPts * POINT_SIZE);
*		fstream indices_file("indices.bin", ios::in | ios::binary);
*		indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * numPts);
*	keep in mind that you may need to change the filepath to the filename in these lines
*/

#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <math.h>
using namespace std;

#define POINT_SIZE 4

// these are all the global variables used by the program
const double numPts = 102416306;

double * ratings;
double * indices;


void initialize()
{
	cout << "Initializing the program.\n";
	ratings = new double [((int) numPts) * POINT_SIZE];
	indices = new double [((int) numPts)];
}

void clean_up()
{
	cout << "Cleaning up\n";
	delete [] ratings;
	delete [] indices;
}

void read_data()
{
	cout << "Reading in the training data." << "\n";
	cout << "Note to user: Make sure the input file paths are correct.\n";

	// read in data
	// change file paths as necessary
	fstream ratingsFile ("../um/all.dta");
	fstream indexFile ("../um/all.idx");
	// user, movie, date, rating
	double inputs [POINT_SIZE] = {};
	int index;
	int pt = 0;

	if (ratingsFile.is_open() && indexFile.is_open()) {
		while (ratingsFile >> inputs[0] >> inputs[1] >> inputs[2] >> inputs[3]) {
		    indexFile >> index;
		    indices[pt] = index;

		    // user
		    ratings[pt * POINT_SIZE] = inputs[0];
		    // movie
		    ratings[pt * POINT_SIZE + 1] = inputs[1];
		    // date
		    ratings[pt * POINT_SIZE + 2] = inputs[2];
		    // rating
		    ratings[pt * POINT_SIZE + 3] = inputs[3];

		    // increment pt so that we update the values for a new point next time
		    pt += 1;
		}
	}
	ratingsFile.close();
	indexFile.close();
}	

void write_binary()
{
	cout << "Writing binary file\n";
	fstream ratings_file ("ratings.bin", ios::out | ios::binary);
	fstream indices_file ("indices.bin", ios::out | ios::binary);

	ratings_file.write(reinterpret_cast<char *>(ratings), sizeof(double) * numPts * POINT_SIZE);
	indices_file.write(reinterpret_cast<char *>(indices), sizeof(double) * numPts);

	ratings_file.close();
	indices_file.close();
}

int main()
{
	initialize();
	read_data();
	cout << ratings[0] << ratings[1] << ratings[2] << ratings[3] << "\n";
	cout << indices[0] << "\n";
	write_binary();
	clean_up();

	initialize();
	fstream ratings_file("ratings.bin", ios::in | ios::binary);
	ratings_file.read(reinterpret_cast<char *>(ratings), sizeof(double) * numPts * POINT_SIZE);
	fstream indices_file("indices.bin", ios::in | ios::binary);
	indices_file.read(reinterpret_cast<char *>(indices), sizeof(double) * numPts);

	cout << ratings[0] << ratings[1] << ratings[2] << ratings[3] << "\n";
	cout << indices[0];

	indices_file.close();
	ratings_file.close();
	clean_up();


	return 0;
}

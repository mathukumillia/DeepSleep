#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <cmath>
#include <map>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>

using namespace std;

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 30 // the maximum number of epochs to run; 30 in the paper
#define MAX_NEIGHBOR_SIZE 300 // obtained from SVD++ paper
#define LAMBDA_7 0.0205 // tuned
#define LAMBDA_A 50 // from belkor paper 
#define LABMDA_pukt 0.01 // from bellkor paper
#define BETA 0.4 // from timeSVD++ paper
#define DECAY 0.9 // from belkor paper 
#define VAL_SET 4 // the point set being used for validation

// set learning rates and regularizers for all variables to be learned
// these values are taken from BellKor paper
#define learningRateBu 0.00267
#define learningRateBi 0.00048
#define learningRateBiBin 0.000115
#define learningRateAu 0.00000311
#define learningRateBut 0.00257
#define learningRateCu 0.00564
#define learningRateCut 0.00103
#define learningRateBifui 0.00236
#define regularizerBu 0.0255
#define regularizerBi 0.0255
#define regularizerBiBin 0.0929
#define regularizerAu 3.95
#define regularizerBut 0.00231
#define regularizerCu 0.0476
#define regularizerCut 0.0190
#define regularizerBifui 0.000000011
#define K 100.0

/*
* These are all the global variables used by the program
*/

// users and movies are one indexed
double num_users = 458293;
double num_movies = 17770;
double num_pts = 102416306;

// GPU version of number of points
double *d_num_pts;

// gamma_2 is the step size         
double GAMMA_2 = 0.008;
double *d_GAMMA_2; // GPU version of GAMMA 2

double GAMMA_pukt = 0.004;
double *d_GAMMA_pukt; // GPU version of GAMMA_pukt
// alpha step size; got default from timeSVD++ repo online
double GAMMA_A = 0.00001;
double *d_GAMMA_A; // GPU version of GAMMA_A

// though these are declared as single dimensional, I will use them as 2D arrays
// to facilitate this, I will store the sizes of the arrays as well
// we add one to the num_users because these arrays are 1-indexed
// these will be on the GPU because I will never need to access them on the host
double *user_values; // this is p in the SVD++ paper
int user_values_size = (int)((num_users + 1) * K);

double *movie_values; // this is q in the SVD++ paper
int movie_values_size = (int)((num_movies + 1) * K);

// the arrays to store the ratings and indices data
// note that ratings will be used as a 2D array as well
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
double *d_ratings; // this is the GPU version of the ratings data

double *indices;
double *d_indices; // this is the GPU version of the indices data

// stores each user's neighborhoods
// functionally, this is a 2D array that stores for each user the id of the
// this is an int array because the double array is too large to work with
// movies they provided feedback for
int *neighborhoods;
int neighborhoods_size = (int) ((num_users + 1) * MAX_NEIGHBOR_SIZE);
int *d_neighborhoods; // this is the GPU version of the neighborhoods data

double *neighborhood_sizes;
double *d_neighborhood_sizes; // this is the GPU version of the neighborhood sizes data

// y is a 2D array that holds K features for each of the movies
// the plus one in the size derives from the fact that the movies are 1 indexed
double *y; // will be on the GPU
int y_size = (int) ((num_movies + 1) * K); 

// 1D array that stores the mean date of rating for each user
// is one indexed
double * t_u;
double *d_t_u; // the GPU version of this mean date of rating vector

// 2D array that stores the alpha value for each user FACTOR for SVD
// will be stored on the GPU
double * alphas;
int alphas_size = (int) ((num_users + 1) * K);


// store the day specific time dependent user SVD terms
// the first vector is indexed by user(1 indexed)
// the second layer is indexed by factor number (0 indexed)
// each element of the map maps a date to a bias term
// vector<vector<map<int, double> > > p_ukt;

/*
* Global variables for baseline prediction
*/

// the mean rating in point set 1
double mean_rating = 3.60861;
double *d_mean_rating;

// this value is taken from BellKor paper
int num_time_bins = 30;
int *d_num_time_bins;

// the maximum time value in the data
int max_time = 2243;
int *d_max_time;

// the size of a single bin
// add 1 to make sure we round up and not down
int binsize = max_time/num_time_bins + 1;
int *d_binsize;

// vector to store data from base set
vector<int> timeBins;

// stores the naive user bias term - will be on GPU 
double * user_biases;
// stores the naive movie biase term  - will be on GPU 
double * movie_biases;

// stores the alpha values that influence the user bias - will be on GPU 
double * bias_alphas;

// store the single day variables for user bias
// index in vector represents user (one indexed as usual)
// each int in the map is a date
// the double is the actual bias term 
vector<map<int, double> > Bu_t;

// store the bin bias terms  - will be on GPU 
double * Bi_bin;
int Bi_bin_size = (int)((num_movies + 1) * num_time_bins);

// store the stable c_u terms - will be on GPU 
double * c_u;

// store the time dependent c_ut terms
// set up very similarly to Bu_t
vector<map<int, double> > c_ut;


/*
* Initialize states for random number generation
*/
__global__ void init_states(unsigned int seed, curandState_t* states)
{
	/* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}


/*
* Takes in an array of states and array of doubles and puts a random double into each
*/
__global__ void randoms(curandState_t* states, double* numbers)
{
	numbers[blockIdx.x] = 0.01 * (double)(curand(&states[blockIdx.x]) % 10)/sqrt((float)K);
}

/*
* Initializes all elements of the numbers array to 1.0
*/
__global__ void init_to_1(double *numbers)
{
	numbers[blockIdx.x] = 1.0;
}

/*
* Allocates memory for the device side
*/
inline void initialize_device()
{

	cout << "Initializing the device.\n";

	// allocate memory for all constants
	cudaMalloc((void **)&d_GAMMA_2, sizeof(double));
	cudaMalloc((void **)&d_GAMMA_pukt, sizeof(double));
	cudaMalloc((void **)&d_GAMMA_A, sizeof(double));
	cudaMalloc((void **)&d_num_pts, sizeof(double));

	cudaMalloc((void **)&d_mean_rating, sizeof(double));
	cudaMalloc((void **)&d_num_time_bins, sizeof(int));
	cudaMalloc((void **)&d_max_time, sizeof(int));
	cudaMalloc((void **)&d_binsize, sizeof(int));


	// initialize all constants on the GPU by copying the memory from host to 
	// device
	cudaMemcpy(d_GAMMA_2, &GAMMA_2, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GAMMA_pukt, &GAMMA_pukt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GAMMA_A, &GAMMA_A, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_pts, &num_pts, sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_mean_rating, &mean_rating, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_time_bins, &num_time_bins, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_max_time, &max_time, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_binsize, &binsize, sizeof(int), cudaMemcpyHostToDevice);

	// now, allocate memory for the user factors, movie factors, user alpha factors, 
	// neighborhoods, neighborhood sizes, bias alpha factors, user bias factors, and movie bias
	// factors

	// these will later be intialized to stochastic values
	cudaMalloc((void **)&user_values, user_values_size * sizeof(double));
	/* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
	curandState_t* user_states;
	/* allocate space on the GPU for the random states */
	cudaMalloc((void**) &user_states, user_values_size * sizeof(curandState_t));
	/* invoke the GPU to initialize all of the random states */
	init_states<<<user_values_size, 1>>>(time(0), user_states);
	/* Assign a random value to each user factor. */
	randoms<<<user_values_size, 1>>>(user_states, user_values);
	cudaFree(user_states);


	cudaMalloc((void **)&movie_values, movie_values_size * sizeof(double));
	/* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
	curandState_t* movie_states;
	/* allocate space on the GPU for the random states */
	cudaMalloc((void**) &movie_states, movie_values_size * sizeof(curandState_t));
	/* invoke the GPU to initialize all of the random states */
	init_states<<<movie_values_size, 1>>>(time(0), movie_states);
	/* Assign a random value to each movie factor. */
	randoms<<<movie_values_size, 1>>>(movie_states, movie_values);
	cudaFree(movie_states);
	

	/* These 5 arrays are populated host side and then copied over */
	cudaMalloc((void **)&d_ratings, ratings_size * sizeof(double));
	cudaMalloc((void **)&d_indices, num_pts * sizeof(int));
	cudaMalloc((void **)&d_neighborhoods, neighborhoods_size * sizeof(int));
	cudaMalloc((void **)&d_neighborhood_sizes, (num_users + 1) * sizeof(double));
	cudaMalloc((void **)&d_t_u, (num_users + 1) * sizeof(double));
	
	cudaMalloc((void **)&y, y_size * sizeof(double));
	cudaMemset((void **)&y, 0, y_size * sizeof(double));
	
	cudaMalloc((void **)&alphas, alphas_size * sizeof(double));
	cudaMemset((void **)&alphas, 0, alphas_size * sizeof(double));


	cudaMalloc((void **)&user_biases, (num_users + 1) * sizeof(double));
	cudaMemset((void **)&user_biases, 0, (num_users + 1) * sizeof(double));

	cudaMalloc((void **)&movie_biases, (num_movies + 1) * sizeof(double));
	cudaMemset((void **)&movie_biases, 0, (num_movies + 1) * sizeof(double));


	cudaMalloc((void **)&bias_alphas, (num_users + 1) * sizeof(double));
	cudaMemset((void **)&bias_alphas, 0, (num_users + 1) * sizeof(double));
	

	cudaMalloc((void **)&c_u, (num_users + 1) * sizeof(double));
	init_to_1<<<(num_users + 1), 1>>>(c_u);

	cudaMalloc((void **)&Bi_bin, (Bi_bin_size) * sizeof(double));
	cudaMemset((void **)&Bi_bin, 0, (Bi_bin_size) * sizeof(double));
}

/*
* Allocates memory for the host side
*
*/
inline void initialize_host()
{
	cout << "Initializing the host.\n";

	// create  the arrays that store the ratings input data and the indexes
    ratings = new double[ratings_size];
    indices = new double[((int) num_pts)];

    neighborhoods = new int[neighborhoods_size];
    neighborhood_sizes = new double[(int)(num_users + 1)];

    t_u = new double[(int)num_users + 1];
}

/*
* Reads the input data into ratings, indices, neighborhoods, and time ratings. 
* Then, this function copies all of these into the GPU memory.
*/
inline void read_data()
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

    // read in neighborhod data
    fstream neighborhood_file("../neighborhoods_12345.bin", ios::in | ios::binary);
    neighborhood_file.read(reinterpret_cast<char *>(neighborhoods), sizeof(int) * (num_users + 1) * MAX_NEIGHBOR_SIZE);
    neighborhood_file.close();

    // read in the neighborhood size data
    fstream nsize_file ("../neighborhood_sizes_12345.bin", ios::in | ios::binary);
    nsize_file.read(reinterpret_cast<char *>(neighborhood_sizes), sizeof(double) * (num_users + 1));
    nsize_file.close();

    // read in the average rating time data
    fstream t_file("../average_time_rating.bin", ios::in | ios::binary);
    t_file.read(reinterpret_cast<char *>(t_u), sizeof(double) * (num_users + 1));
    t_file.close();

    // copy all of these host side arrays to the GPU
    cudaMemcpy(d_ratings, &ratings, ratings_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, &indices, num_pts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborhoods, &neighborhoods, neighborhoods_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborhood_sizes, &neighborhood_sizes, (num_users + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_u, &t_u, (num_users + 1) * sizeof(double), cudaMemcpyHostToDevice);
}

/*
* Clears all used memory, on device and host.
*/
inline void clean_up()
{
	delete [] ratings;
	delete [] indices;
	delete [] neighborhoods;
	delete [] neighborhood_sizes;
	delete [] t_u;

	cudaFree(d_GAMMA_2);
	cudaFree(d_GAMMA_pukt);
	cudaFree(d_GAMMA_A);
	cudaFree(d_num_pts);

	cudaFree(d_mean_rating);
	cudaFree(d_num_time_bins);
	cudaFree(d_max_time);
	cudaFree(d_binsize);

	cudaFree(d_ratings);
	cudaFree(d_indices);
	cudaFree(d_neighborhoods);
	cudaFree(d_neighborhood_sizes);
	cudaFree(d_t_u);

	cudaFree(y);
	cudaFree(alphas);
	cudaFree(user_biases);
	cudaFree(movie_biases);
	cudaFree(bias_alphas);
	cudaFree(c_u);
	cudaFree(Bi_bin);	
	cudaFree(user_values);
	cudaFree(movie_values);	
}

int main()
{
	initialize_host();
	initialize_device();
	read_data();

	

	clean_up();
}


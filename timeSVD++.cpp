#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <cmath>
#include <map>
#include <vector>
#include <stdlib.h>
#include <algorithm>

// a point looks like (user, movie, date, rating)
#define POINT_SIZE 4 // the size of a single input point in the training data
#define STOPPING_CONDITION 0
#define MAX_EPOCHS 40 // the maximum number of epochs to run; 30 in the paper
#define MAX_NEIGHBOR_SIZE 300 // obtained from SVD++ paper
#define LAMBDA_7 0.0205 // tuned
#define LAMBDA_A 50 // from belkor paper 
#define LABMDA_pukt 0.01 // from bellkor paper
#define BETA 0.4 // from timeSVD++ paper
#define DECAY 0.9 // from belkor paper 
#define VAL_SET 4 // the point set being used for validation

using namespace std;

/*
* These are all the global variables used by the program
*/

// users and movies are one indexed
const double num_users = 458293;
const double num_movies = 17770;
const double num_pts = 102416306;

// K is the constant representing the number of features
// gamma_2 is the step size         
const double K = 100;
double GAMMA_2 = 0.008;
// 
double GAMMA_pukt = 0.004;
// alpha step size; got default from timeSVD++ repo online
double GAMMA_A = 0.00001;

// though these are declared as single dimensional, I will use them as 2D arrays
// to facilitate this, I will store the sizes of the arrays as well
// we add one to the num_users because these arrays are 1-indexed
double *user_values; // this is p in the SVD++ paper
int user_values_size = (int)((num_users + 1) * K);
double *movie_values; // this is q in the SVD++ paper
int movie_values_size = (int)((num_movies + 1) * K);

// the arrays to store the ratings and indices data
// note that ratings will be used as a 2D array as well
double *ratings;
int ratings_size = (int) (num_pts * POINT_SIZE);
double *indices;

// stores each user's neighborhoods
// functionally, this is a 2D array that stores for each user the id of the
// this is an int array because the double array is too large to work with
// movies they provided feedback for
int *neighborhoods;
int neighborhoods_size = (int) ((num_users + 1) * MAX_NEIGHBOR_SIZE);
double *neighborhood_sizes;

// y is a 2D array that holds K features for each of the movies
// the plus one in the size derives from the fact that the movies are 1 indexed
double *y;
int y_size = (int) ((num_movies + 1) * K);

// 1D array that stores the mean date of rating for each user
// is one indexed
double * t_u;

// 2D array that stores the alpha value for each user FACTOR for SVD
double * alphas;
int alphas_size = (int) ((num_users + 1) * K);

// list of maps that help store the dev results to make the function run faster
vector< map<int, double> > dev_results;

// boolean to enable debugging messages 
bool DEBUG = false;

// store the day specific time dependent user SVD terms
// the first vector is indexed by user(1 indexed)
// the second layer is indexed by factor number (0 indexed)
// each element of the map maps a date to a bias term
// vector<vector<map<int, double> > > p_ukt;

/*
* Global variables for baseline prediction
*/

// the mean rating in point set 1
const double mean_rating = 3.60861;

// this value is taken from BellKor paper
const int num_time_bins = 30;

// the maximum time value in the data
const int max_time = 2243;

// the size of a single bin
// add 1 to make sure we round up and not down
int binsize = max_time/num_time_bins + 1;

// vector to store data from base set
vector<int> timeBins;

// stores the naive user bias term
double * user_biases;
// stores the naive movie biase term 
double * movie_biases;

// stores the alpha values that influence the user bias
double * bias_alphas;

// store the single day variables for user bias
// index in vector represents user (one indexed as usual)
// each int in the map is a date
// the double is the actual bias term 
vector<map<int, double> > Bu_t;

// store the bin bias terms 
// stored in a 2D array because this is way too wide to store in one chunk
double ** Bi_bin;

// store the stable c_u terms
double * c_u;

// store the time dependent c_ut terms
// set up very similarly to Bu_t
vector<map<int, double> > c_ut;

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

/*
* Allocates memory and initializes user, movie, ratings, and indices arrays
*/
inline void initialize()
{
    cout << "Initializing the program.\n";

    // allocate and initialize the user_values and movie_values matrices
    // all of the +1 terms result from the fact that these arrays are 1
    // indexed in the data
    user_values = new double[user_values_size];
    for (int i = 0; i < user_values_size; i++)
    {
        user_values[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K); // arbitrary initial condition
    }

    movie_values = new double[movie_values_size];
    y = new double[y_size];
    for (int i = 0; i < movie_values_size; i++)
    {
        movie_values[i] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K);
        y[i] = 0; // this was an arbitrary initial condition
    }

    // create  the arrays that store the ratings input data and the indexes
    ratings = new double[ratings_size];
    indices = new double[((int) num_pts)];

    neighborhoods = new int[neighborhoods_size];
    neighborhood_sizes = new double[(int)(num_users + 1)];

    t_u = new double[(int)num_users + 1];
    alphas = new double[alphas_size]();

    for (int i = 0; i < alphas_size; i++)
    {
        alphas[i] = 0;
    }

    // initialize the naive user biases
    user_biases = new double[(int)num_users + 1];
    // initialize the user bias alpha values
    bias_alphas = new double[(int)num_users + 1];
    // initialize the c_u values
    c_u = new double[(int)num_users + 1];
    // allocate the dev results and initialize all of the values
    for (int i = 0; i < num_users + 1; i++)
    {
        map<int,double> tmp;
        dev_results.push_back(tmp); 
        // initialize the naive user biases 
        user_biases[i] = 0.0;
        bias_alphas[i] = 0.0;
        c_u[i] = 1.0;
    }

    // initialize the naive movie biases
    movie_biases = new double[(int)num_movies + 1];
    // initialize the bins of time for the movies; again, this is 1 indexed,
    // hence the +1
    Bi_bin = new double* [(int)num_movies + 1];
    for (int i = 0; i < num_movies + 1; i++)
    {
        // initialize the naive movie biases
        movie_biases[i] = 0;
        Bi_bin[i] = new double[num_time_bins];

        // initialize the bin biases to 0
        for (int j = 0; j < num_time_bins; j++)
        {
            Bi_bin[i][j] = 0.0;
        }
    }

    cout << "Done allocating memory.\n";
}

/*
* clears all used memory
*/
inline void clean_up()
{
    cout << "Cleaning up.\n";
    delete [] user_values;
    delete [] movie_values;
    delete [] ratings;
    delete [] indices;
    delete [] neighborhoods;
    delete [] neighborhood_sizes;
    delete [] y;
    delete [] t_u;
    delete [] alphas;
    delete [] user_biases;
    delete [] movie_biases;
    for(int i = 0; i < num_movies + 1; i++)
    {
        delete [] Bi_bin[i];
    }
    delete [] Bi_bin;
    delete [] bias_alphas;
    delete [] c_u;
}

/*
* Reads the input data into ratings and indices
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
    fstream neighborhood_file("../neighborhoods_123.bin", ios::in | ios::binary);
    neighborhood_file.read(reinterpret_cast<char *>(neighborhoods), sizeof(int) * (num_users + 1) * MAX_NEIGHBOR_SIZE);
    neighborhood_file.close();

    // read in the neighborhood size data
    fstream nsize_file ("../neighborhood_sizes_123.bin", ios::in | ios::binary);
    nsize_file.read(reinterpret_cast<char *>(neighborhood_sizes), sizeof(double) * (num_users + 1));
    nsize_file.close();

    // read in the average rating time data
    fstream t_file("../average_time_rating.bin", ios::in | ios::binary);
    t_file.read(reinterpret_cast<char *>(t_u), sizeof(double) * (num_users + 1));
    t_file.close();
}

/*
* Initializes the Bu_t bias vector and c_ut scaling vector
*
*/
inline void init_time_bias()
{
    cout << "Initializing Bu_t, c_ut, and p_ukt.\n";
    // create the map for each user
    for (int i = 0; i < num_users + 1; i++)
    {
        // insert maps into Bu_t
        map<int, double> tmp;
        Bu_t.push_back(tmp);
       
        // insert maps into c_ut
        map<int, double> tmp2;
        c_ut.push_back(tmp2);

        // initialize p_ukt by inserting vectors of maps into it into it
        // each vector signifies a single user's factor profile
        // vector<map<int, double> > tmp3;
        // // insert the date maps for each factor into each vector
        // for (int j = 0; j < K; j++)
        // {
        //     map<int, double> tmp4;
        //     tmp3.push_back(tmp4);
        // }
        // p_ukt.push_back(tmp3);
    }
    int user;
    int date; 
    // loop through training (set 1, 2, 3, and 4) data and insert into the map as needed
    for (int i = 0; i < num_pts; i++)
    {
        if(indices[i] == 1 || indices[i] == 2 || indices[i] == 3 || indices[i] == 4)
        {
            user = (int)ratings[i * POINT_SIZE];
            date = (int)ratings[i * POINT_SIZE + 2];
            // if the current user and date haven't been initialized, 
            // initialize the pair
            if (Bu_t[user].count(date) == 0)
            {
                Bu_t[user][date] = 0;
            }
            if (c_ut[user].count(date) == 0)
            {
                c_ut[user][date] = 0;
            }
            // loop through each K and check if the corresponding user, K 
            // pair has a certain bias value for this date
            // for (int j = 0; j < K; j++)
            // {
            //     if(p_ukt[user][j].count(date) == 0)
            //     {
            //         p_ukt[user][j][date] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(K);
            //     }
            // }
        }
    }
}

/*
* Get the sum of the neighborhood vectors for the given user
* this is |N(u)|^(-1/2) * sum of y's in neighborhood of u
* takes in a pointer to an array of all 0s to which it can store the result
* allocated to this array when using this function
*/
inline void get_y_sum(int user, double * y_vector_sum)
{
    // stores the movie for which we are obtaining y
    int neighborhood_movie;
    // loop through neighborhood and get sum of y vectors for each movie in 
    // neighborhood
    if(neighborhood_sizes[user] > 0)
    {
        for (int i = 0; i < neighborhood_sizes[user]; i++)
        {
            neighborhood_movie = neighborhoods[user * MAX_NEIGHBOR_SIZE + i];
            // add the current neighborhood movies y vector to the user vector sum
            for (int j = 0; j < K; j++)
            {
                y_vector_sum[j] += y[neighborhood_movie * (int)K + j];
            }
        }
        // loop through and divide each of the elements by the square root of 
        // the neighborhood size
		for (int i = 0; i < K; i++)
        {
            y_vector_sum[i] = y_vector_sum[i]/sqrt(neighborhood_sizes[user]);
        }
    }
}

/*
* sign function to help dev
*/
inline double sign(double x)
{
    if (x < 0)
        return -1;
    else if(x > 0)
        return 1;
    return 0;
}

/*
* The function dev_u(t) given in the timeSVD++ paper
*
*/
inline double dev(int user, double time)
{
    if(dev_results[user].count(time)!=0){
        return dev_results[user][time];
    }
    double tmp = sign(time - t_u[user]) * pow(double(abs(time - t_u[user])), BETA);
    dev_results[user][time] = tmp; 
    return tmp;
}

/*
* Predict the rating given a user and movie and date
*
* @return: the double rating
*/
inline double predict_rating(int user, int movie, double date)
{
	// gets the sum of the neighborhood vectors 
	double * user_vector = new double[(int)K]();
    get_y_sum(user, user_vector);

	// add in the current user factors to the neighborhood sum
	for (int i = 0; i < K; i++)
	{
        // add the fixed user values and the time values to the user_vector 
		user_vector[i] = 
            user_values[user * (int)K + i] + alphas[user * (int)K + i] * dev(user, date) + user_vector[i];
	}

	// compute the rating as a fu
	double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector[i] * movie_values[movie * (int)K + i];
    }

    // add in the user biases
    // if Bu_t has not been initialized for this date, we set it to zero
    double bu_value;
    if (Bu_t[user].count(date) == 0)
    {
        bu_value = 0;
    }
    else
    {
        bu_value = Bu_t[user][(int)date];
    }
    rating += user_biases[user] + bias_alphas[user] * dev(user, date) + bu_value;

    // add in the movie biases multiplied by the scaling factors
    // if c_ut has not been initialized for this date, we set it to zero
    double cu_value;
    if (c_ut[user].count(date) == 0)
    {
        cu_value = 0;
    }
    else
    {
        cu_value = c_ut[user][(int)date];
    }
    rating += (movie_biases[movie] + Bi_bin[movie][(int)date/binsize]) * (c_u[user] + cu_value);

    // add in the overall mean
    rating += mean_rating;

    delete [] user_vector;
    return rating;
}

/*
* Predicts the rating given a y_sum that has already been calculated
*/
inline double predict_rating(int user, int movie, double date, double * y_sum)
{
    double user_vector[(int)K] = {};
    for (int i = 0; i < K; i++)
    {
        user_vector[i] = 
            user_values[user * (int)K + i] + alphas[user * (int)K + i] * dev(user, date) + y_sum[i];
    }
	// compute the rating
	double rating = 0;
    for (int i = 0; i < K; i++)
    {
        rating += user_vector[i] * movie_values[movie * (int)K + i];
    }

    // add in the user biases
    // if Bu_t has not been initialized for this date, we set it to zero
    double bu_value;
    if (Bu_t[user].count(date) == 0)
    {
        bu_value = 0;
    }
    else
    {
        bu_value = Bu_t[user][(int)date];
    }
    rating += user_biases[user] + bias_alphas[user] * dev(user, date) + bu_value;

    // add in the movie biases multiplied by the scaling factors
    // if c_ut has not been initialized for this date, we set it to zero
    double cu_value;
    if (c_ut[user].count(date) == 0)
    {
        cu_value = 0;
    }
    else
    {
        cu_value = c_ut[user][(int)date];
    }
    rating += (movie_biases[movie] + Bi_bin[movie][(int)date/binsize]) * (c_u[user] + cu_value);

    // add in the overall mean
    rating += mean_rating;

    return rating;
}

/*
* Get the error on a given set of points.
* 	i.e. set 2 is the validation set
*
* @return: the double error
*/
inline double error (int set)
{
	cout << "Calculating error.\n";

    double error = 0;
    double diff = 0;
    double points_in_set = 0;
    int user; 
    int movie;
    double date;
    double rating;

    for (int i = 0; i < num_pts; i++) {

        user = (int)ratings[i * POINT_SIZE];
        movie = (int)ratings[i * POINT_SIZE + 1];
        date = ratings[i * POINT_SIZE + 2];
        rating = ratings[i * POINT_SIZE + 3];

        if (indices[i] == set) {
            diff = rating - predict_rating(user, movie, date);
            error += diff * diff;
            points_in_set += 1;
        }
    }

    return sqrt(error/points_in_set);
}

/*
* Trains the SVD++ model on one provided point
* Point must contain the user, movie, date, and rating
* also takes in y_sum, which just contains the sum of the y's in the 
* neighborhood. Also takes in a train_probe boolean value that tells 
* the function if we are training on probe data or not
* utilizes SGD
*/
inline void train(int train_probe)
{
    int userId, itemId, currentUser, i;
    double date;
    double rating;
    int pt_index = 0;
    double * y_sum_im = new double[(int)K]();
    double * y_sum_old = new double[(int)K]();
    double user_vector[(int)K] = {};
    double point_error;
    double movie_factor;
    double user_factor;
    double alpha_factor;
    // neighborhood size
    double size; 
    double dev_val;
    double old_bin_val;
    double old_movie_val;

    // iterates through the users
    for (userId = 1; userId <= num_users; userId++)
    {
        // get the neighborhood size for this user
        size = neighborhood_sizes[userId];
        // get the sum of neighborhood vectors for this user
        get_y_sum(userId, y_sum_im);
        // set y_sum_old equal to y_sum_im
        for (i = 0; i < K; i++)
        {
            y_sum_old[i] = y_sum_im[i];
        }

        // this goes through all the training samples associated with a user
        while (ratings[pt_index * POINT_SIZE] == userId)
        {
            // only train if this is in the set we want to train on
            if ((train_probe == 0 && indices[pt_index] < 4) || (train_probe == 1 && indices[pt_index] <= 4))
            {
                // store some things to make referring to them easier
                itemId = (int)ratings[pt_index * POINT_SIZE + 1]; 
                date = ratings[pt_index * POINT_SIZE + 2];
                rating = ratings[pt_index * POINT_SIZE + 3];
                dev_val = dev(userId, date);
                // get the point error
                point_error = rating - predict_rating(userId, itemId, date, y_sum_im);
                // update all of the SVD factors including, p, q, y, and alpha
                for (i = 0; i < K; i++)
                {
                    // update the movie, user, and alpha factors 
                    movie_factor = movie_values[itemId * (int)K + i];
                    user_factor = user_values[userId * (int)K + i];
                    alpha_factor = alphas[userId * (int)K + i];

                    movie_values[itemId * (int)K + i] += 
                        GAMMA_2 * (point_error * (user_factor + alpha_factor * dev_val + y_sum_im[i]) - LAMBDA_7 * movie_factor);
                    user_values[userId * (int)K + i] +=
                        GAMMA_2 * (point_error * movie_factor - LAMBDA_7 * user_factor);

                    // update y_sum_im
                    if(size != 0)
                    {
                        y_sum_im[i] += 
                            GAMMA_2 * (point_error * movie_factor - LAMBDA_7 * y_sum_im[i]);
                    }   
                    alphas[userId * (int)K + i] += 
                        GAMMA_A * (point_error * (movie_factor * dev_val) - LAMBDA_A * alpha_factor); 

                    // update the p_ukt factors - these are the daily effects factors for SVD
                    // p_ukt[userId][i][(int)date] += 
                    //     GAMMA_pukt * (point_error * movie_factor - LABMDA_pukt * p_ukt[userId][i][(int)date]);
                }

                /*
                * These next few lines update the baseline metrics
                *
                */
                // update the naive user biases
                user_biases[userId] += learningRateBu * (point_error - regularizerBu * user_biases[userId]);

                // update the naive movie biases
                old_movie_val = movie_biases[itemId];
                movie_biases[itemId] += learningRateBi * (point_error - regularizerBi * old_movie_val);

                // update the movie time bin bias term
                old_bin_val =  Bi_bin[itemId][(int)date/binsize];
                Bi_bin[itemId][(int)date/binsize] += 
                    learningRateBiBin * (point_error - regularizerBiBin * old_bin_val);
                
                // update the user bias alphas
                bias_alphas[userId] += 
                    learningRateAu * (point_error * dev_val - regularizerAu * bias_alphas[userId]);
                
                // update the day specific user bias value
                Bu_t[userId][(int)date] += 
                    learningRateBut * (point_error - regularizerBut * Bu_t[userId][(int)date]);

                // update the c_u value
                c_u[userId] += 
                    learningRateCu * (point_error * (old_movie_val + old_bin_val) - regularizerCu * (c_u[userId] - 1));

                // update the time sensitive c value
                c_ut[userId][(int)date] +=
                     learningRateCut * (point_error * (old_movie_val + old_bin_val) - regularizerCut * c_ut[userId][(int)date]);
                
            }
            // update the pt_index to the next point
            pt_index++;
        }

        // update they y_values using the y_sum_im and y_sum_old vectors
        for (i = 0; i < size; i++)
        {
            // the movie neighbor
            itemId = neighborhoods[userId * MAX_NEIGHBOR_SIZE + i];
            for (int j = 0; j < K; j++)
            {
                y[itemId * (int)K + j] += (y_sum_im[j] - y_sum_old[j])/sqrt(size);
            }
        }
    }
    delete [] y_sum_im;
    delete [] y_sum_old;
}

/*
* Iterates through every point in the training set and trains on each one
*/
inline void run_epoch()
{
    cout << "Running an epoch.\n";
    // train not including the probe set
    train(0);
    // decrease gammas by 10%, as suggested in paper
    GAMMA_2 = DECAY * GAMMA_2;
    GAMMA_pukt = DECAY * GAMMA_pukt;
    GAMMA_A = DECAY * GAMMA_A;
}

/*
* get predictions on the qual set and output them to a file
* fix this to work with baseline removed predictions
*
*/
inline void findQualPredictions()
{
    cout << "Finding and writing qual predictions.\n";
    ofstream outputFile;
    ofstream probeFile;
    outputFile.open("timeSVD++_qual_output.dta");
    probeFile.open("timeSVD++_probe_output.dta");

    double prediction;
    int user;
    int movie;
    double date;
    for(int i = 0; i < num_pts; i++)
    {
        if (indices[i] == 5 || indices[i] == 4)
        {
        	user = (int)ratings[i * POINT_SIZE];
        	movie = (int)ratings[i * POINT_SIZE + 1];
        	date = ratings[i * POINT_SIZE + 2];
            // I have to add the ratings in the file because this ratings file
            // has the baselines removed
            prediction = predict_rating(user, movie, date);
            if (prediction < 1)
            {
                prediction = 1;
            }
            if (prediction > 5)
            {
                prediction = 5;
            }
            if(indices[i] == 5)
            {
                outputFile << prediction << "\n";
            }
            else if(indices[i] == 4)
            {
                probeFile << prediction << "\n";
            }
        }
    }
    outputFile.close();
    probeFile.close();
}

void writeMatrices()
{
    cout << "Writing matrices.\n";
    // write the user matrix to a file
    ofstream user_file ("../user_matrix.txt");
    // loop through each user 
    if (user_file.is_open())
    {
        cout << "Writing user matrix.\n";
        for (int user = 1; user <= num_users; user++)
        {
            // loop through and write each of the users factors
            for (int factor = 0; factor < K; factor++)
            {
                user_file << user_values[user * (int)K + factor] << ", ";
            }
            // print a new line
            user_file << "\n";
        }
        user_file.close();
    }

    // write the movie matrix to a file
    ofstream movie_file ("../movie_matrix.txt");
    // loop through each movie 
    if (movie_file.is_open())
    {
        cout << "Writing movie matrix.\n";
        for (int movie = 1; movie <= num_movies; movie++)
        {
            // loop through and write each of the movie factors
            for (int factor = 0; factor < K; factor++)
            {
                movie_file << movie_values[movie * (int)K + factor] << ", ";
            }
            // print a new line
            movie_file << "\n";
        }
        movie_file.close();
    }
}

int main()
{
    initialize();
    read_data();
    init_time_bias();
    double initialError = 10000;
    double finalError = error(VAL_SET);
    int counter = 1;

    cout << "The starting error is: " << finalError << "\n";
    while (initialError - finalError > STOPPING_CONDITION && counter <= MAX_EPOCHS) {
        cout << "Starting Epoch " << counter << "\n";
        run_epoch();
        if (counter % 1 == 0)
        {
            initialError = finalError;
            finalError = error(VAL_SET);
            cout << "Error after " << counter << " epochs: " << finalError << "\n";
        }
        counter++;
    }
    cout << "Final validation error before probe training: " << finalError << "\n";

    // read in neighborhod data with probe 
    fstream neighborhood_file("../neighborhoods_1234.bin", ios::in | ios::binary);
    neighborhood_file.read(reinterpret_cast<char *>(neighborhoods), sizeof(int) * (num_users + 1) * MAX_NEIGHBOR_SIZE);
    neighborhood_file.close();

    // read in the neighborhood size data with probe
    fstream nsize_file ("../neighborhood_sizes_1234.bin", ios::in | ios::binary);
    nsize_file.read(reinterpret_cast<char *>(neighborhood_sizes), sizeof(double) * (num_users + 1));
    nsize_file.close();

    // reset learning rates
    GAMMA_2 = 0.008;
    GAMMA_pukt = 0.004;
    // alpha step size; got default from timeSVD++ repo online
    GAMMA_A = 0.00001;

    cout << "Beginning probe training: \n";
    // run 15 epochs of probe data training
    for (int i = 0; i < counter; i++)
    {
        cout << "Probe epoch " << i << "\n";
        // train with the probe set
        train(1);
        // decrease gammas by 10%, as suggested in paper
        GAMMA_2 = DECAY * GAMMA_2;
        GAMMA_pukt = DECAY * GAMMA_pukt;
        GAMMA_A = DECAY * GAMMA_A;
    }

    // write the resulting matrices to files
    writeMatrices();

    // find the values on the qual set
    findQualPredictions();


    clean_up();
    return 0;
}

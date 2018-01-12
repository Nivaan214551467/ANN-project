/******************************************************************************************************
				NIVAAN KRISHUNDUTT
				214551467
				ENEL4AI
				Classification using Neural Network
	
				DataPrep.cpp
				- Definitions of methods expressed in DataPrep

*******************************************************************************************************/

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <iomanip>
#include "DataPrep.h"

using namespace std;
using namespace cv;

DataPrep::DataPrep()
{
}

DataPrep::~DataPrep()
{
}

Mat DataPrep::GLCM_calc(Mat image, int a, int d){

	Mat GLCM = Mat::zeros(256, 256, CV_32FC1);		//Creates an empty Matrix of 256x256. Stores the GLCM data
		
	int neighbour_i = 0;				//Stores the row value of the neighboring pixel.
	int neighbour_j = 0;				//Stores the column value of the neighboring pixel.
	int i_in = 0;						//Stores the row value of where to start traversing the image
	int j_in = 0;						//Stores the column value of where to start traversing the image
	int i_lim, j_lim = 0;				//Stores the row and column values of where to stop traversing the image

	float energy = 0;					//Stores total energy of the image
	float homo = 0;						//Stores total homogeneity of the image
	float contrast = 0;					//Stores total contrast of the image
	float correlation = 0;				//Stores total correlation of the image
	float entropy = 0.0;				//Stores total entropy of the image

	if (a == 0){						//If orientation is 0:
		neighbour_i = 0;				//- neighbour pixel is in the same row
		neighbour_j = d;				//- neighbour pixel is at a distance d columns away
		i_lim = image.rows;				//- stop traversing the image at the last row
		j_lim = image.cols - d;			//	and d columns away from the last column
	}
	else if (a == 45){					//If orientation is 45:
		neighbour_i = -1 * d;			//- neighbour pixel is -d rows away
		neighbour_j = d;				//- neighbour pixel is at a distance d columns away
		i_in = d;						//- start traversing the image from row d
		i_lim = image.rows;				//- stop traversing the image at the last row
		j_lim = image.cols - d;			//	and d columns away from the last column
	}
	else if (a == 90){					//If orientation is 90:
		neighbour_i = -1 * d;			//- neighbour pixel is -d rows away
		neighbour_j = 0;				//- neighbour pixel is at the same column
		i_in = d;						//- start traversing the image from row d
		i_lim = image.rows;				//- stop traversing the image at the last row
		j_lim = image.cols;				//	and last column
	}
	else{								//If orientation is 135:
		neighbour_i = -1 * d;			//- neighbour pixel is -d rows away
		neighbour_j = -1 * d;			//- neighbour pixel is -d columns away
		i_in = d;						//- start traversing the image from row d
		j_in = d;						//	and column d
		i_lim = image.rows;				//- stop traversing the image at the last row
		j_lim = image.cols;				//	and last column
	}

	//Nested loop that traverses every element in the image matrix. Increments the GLCM at position (x,y) when 
	//the Gray level value of the reference pixel is x and the Gray level value of the neighbour pixel is y

	for (int i = i_in; i < i_lim; i++)
		for (int j = j_in; j < j_lim; j++)
			GLCM.at<int>(image.at<uchar>(i, j), image.at<uchar>(i + neighbour_i, j + neighbour_j))++;
	
//--------------------Normalizing the GLCM--------------------------------------------------

	float GLCM_sum = 0;

	for (int i = 0; i < 256; i++)					//Getting sum of the GLCM
		for (int j = 0; j < 256; j++)
			GLCM_sum += GLCM.at<int>(i, j);

	for (int i = 0; i < 256; i++)					//Dividing all elements by the sum. Converting from Int to Float
		for (int j = 0; j < 256; j++)
			GLCM.at<float>(i, j) = GLCM.at<int>(i, j) / GLCM_sum;


//-------------------------Calculating means and standard deviations-----------------------------------------------------

	float m_i[256];				//Array that stores the means of all 256 rows in the GLCM
	float m_j[256];				//Array that stores the means of all 256 columns in the GLCM
	float sd_i[256];			//Array that stores the standard deviations of all 256 rows in the GLCM
	float sd_j[256];			//Array that stores the standard deviations of all 256 columns in the GLCM

	for (int k = 0; k < 256; k++){
		for (int m = 0; m < 256; m++){
			m_i[k] += GLCM.at<float>(k, m);		//Calculating the sums of each column and row independantly
			m_j[m] += GLCM.at<float>(k, k);		//Storing at respective positions in the arrays
		}
	}

	for (int k = 0; k < 256; k++){
		m_i[k] /= 256;							//Dividing all sums by 256 thus producing the means 
		m_j[k] /= 256;
	}
	
	for (int k = 0; k < 256; k++){				//Calculating standard deviations
		for (int m = 0; m < 256; m++){
			sd_i[k] += pow((GLCM.at<float>(k, m) - m_i[k]), 2);		
			sd_j[m] += pow((GLCM.at<float>(k, m) - m_j[m]), 2);
		}
	}

	for (int k = 0; k < 256; k++){				//Completing the standard deviation calculations
		sd_i[k] = sqrt(sd_i[k] / (255));
		sd_j[k] = sqrt(sd_j[k] / (255));
	}
	

//--------------------------Calculating the Haralick features------------------------------------------
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			energy += GLCM.at<float>(i, j) * GLCM.at<float>(i, j);
			homo += GLCM.at<float>(i, j) / (1 + abs(i - j));
			contrast += GLCM.at<float>(i, j) * pow(i - j, 2);
			correlation += (GLCM.at<float>(i, j)*(i - m_i[i])*(j - m_j[j])) / (sd_i[i] * sd_j[j]); //uses respective means and standard deviations that were calculated previously
			if (GLCM.at<float>(i, j) != 0)												//if normalized GLCM is 0, skip this as log(0) will result in error
				entropy += GLCM.at<float>(i, j) * (float)(-log(GLCM.at<float>(i, j)));
		}
	}

	Mat data_vec(1, 5, CV_32FC1);					//Vector to store the haralick features
	data_vec.at<float>(0, 0) = energy;				//Stores energy in 1st element
	data_vec.at<float>(0, 1) = homo;				//Stores homogeneity in 2nd element
	data_vec.at<float>(0, 2) = contrast;			//Stores contrast in 3rd element
	data_vec.at<float>(0, 3) = correlation;			//Stores correlation in 4th element
	data_vec.at<float>(0, 4) = entropy;				//Stores entropy in 5th element
	
	return data_vec;
}

void DataPrep::print_to_file(ofstream& outfile, Mat data_vec){
	//Writes data to specified text file. Each value is 16 digits long with a decimal precision of 8 digits, padded with zeros.
	//A whitespace seperates each feature
	outfile << fixed << setprecision(30) << setfill('0') << internal << data_vec.at<float>(0, 0) << " ";
	outfile << fixed << setprecision(30) << setfill('0') << internal << data_vec.at<float>(0, 1) << " ";
	outfile << fixed << setprecision(30) << setfill('0') << internal << data_vec.at<float>(0, 2) << " ";
	outfile << fixed << setprecision(30) << setfill('0') << internal << data_vec.at<float>(0, 3) << " ";
	outfile << fixed << setprecision(30) << setfill('0') << internal << data_vec.at<float>(0, 4) << " ";
	}

void DataPrep::training_data(cv::String folder, cv::String textname){

	vector<cv::String> filenames;			//Vector that stores filenames of images

	glob(folder, filenames);				//FUnction adds filenames of images from the specified folder to the filenames vector
	ofstream datafile;
	datafile.open(textname);				//Write to specified text file

	for (int i = 0; i < filenames.size(); i++)	//Loop traverses through every file name stored in the filenames vector
	{
		Mat image_load = imread(filenames[i]);	//Loads image of the ith file name in the filenames vector
		cout << "Processing image : " << filenames[i].substr(filenames[i].find_last_of("/\\") + 1) << endl;	//Displays the folder of the image being processed

		if (!image_load.data)
			cout << "Problem loading image!!!" << endl;

		print_to_file(datafile, GLCM_calc(image_load, 0, 1));		//Calculates GLCM for 0 degrees orientation and distance 1. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 45, 1));		//Calculates GLCM for 45 degrees orientation and distance 1. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 90, 1));		//Calculates GLCM for 90 degrees orientation and distance 1. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 135, 1));		//Calculates GLCM for 135 degrees orientation and distance 1. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 0, 2));		//Calculates GLCM for 0 degrees orientation and distance 2. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 45, 2));		//Calculates GLCM for 45 degrees orientation and distance 2. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 90, 2));		//Calculates GLCM for 90 degrees orientation and distance 2. Writes data to text file.
		print_to_file(datafile, GLCM_calc(image_load, 135, 2));		//Calculates GLCM for 135 degrees orientation and distance 2. Writes data to text file.

		datafile << endl;											//Moves to next line in text file
	}
	cout << textname << " text file is closed.\n\n" << endl;
	datafile.close();
}

void DataPrep::prep_data(){		//Function called to prepare the training data set. Uses training_data function for all 3 classes
	training_data("training-and-test-files/good_train", "good_train.txt");
	training_data("training-and-test-files/empty_train", "empty_train.txt");
	training_data("training-and-test-files/bad_ train", "bad_train.txt");
}

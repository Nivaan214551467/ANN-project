/******************************************************************************************************
					NIVAAN KRISHUNDUTT
					214551467
					ENEL4AI
					Classification using Neural Network

					Main.cpp
					- Uses "DataPrep" class to calculate GLCM and prepare training data

*******************************************************************************************************/

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\core.hpp>
#include <opencv\ml.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include "DataPrep.h"				//This class contains methods for calculating GLCM and preparing the training data

using namespace std;
using namespace cv;

Mat data_to_matrix(){						//Function used to move the training data from text files to a Matrix object

	Mat training_data(108, 40, CV_32FC1);	//Create a matrix with 108 rows (number of samples) and 40 columns (number of features)
	cv:String line = "";
	
	ifstream reader("good_train.txt");		//Reading data from the "Good" training set
	int row = 0;							//Used to control which row the data is stored in the training_data Matrix 
	
	while (!reader.eof())					//Stop when end of file reached 
	{
		getline(reader, line);				//Reads one line at a time, stores in string variable called line
		int pos = 0;
		int column = 0;

		//Data in the textfiles are seperated by whitespaces. The loop below seperated the data by using a whitespace as a delimiter
		while ((pos = line.find(" ")) != std::string::npos) {					//Stops when end of string reached. 
			training_data.at<float>(row, column) = stof(line.substr(0, pos));	//Converts data from string to float and then stores in training_data matrix
			line.erase(0, pos + 1);												//Removes the data that has been stored from the string
			column++;															//Move to next column in the training_data matrix
		}
		row++;																	//Move to next row in the training_data matrix
	}
	row--;																		//Goes back to the previous row as this row is skipped before exiting the above loop
																				//Ensures that the "Empty" training data is stored directly after the "Good"training data
	ifstream reader1("empty_train.txt");										//Uses similar approach that was done for good_train.txt
	while (!reader1.eof())
	{
		getline(reader1, line);
		int pos = 0;
		int column = 0;
		while ((pos = line.find(" ")) != std::string::npos) {
			training_data.at<float>(row, column) = stof(line.substr(0, pos));
			line.erase(0, pos + 1);
			column++;
		}
		row++;
	}
	row--;
	ifstream reader2("bad_train.txt");										//Uses similar approach that was done for good_train.txt
	while (!reader2.eof())
	{
		getline(reader2, line);
		int pos = 0;
		int column = 0;
		while ((pos = line.find(" ")) != std::string::npos) {
			training_data.at<float>(row, column) = stof(line.substr(0, pos));
			line.erase(0, pos + 1);
			column++;
		}
		row++;
	}
	
	reader.close();
	reader1.close();
	reader2.close();

	return training_data;									//Returns the training data in a Matrix
}

void load_test_data(Mat test_data, Mat input, int count){				//Function that stores data from the input matrix to the test_data matrix
	test_data.at<float>(0, 0 + (5 * count)) = input.at<float>(0, 0);	//Control variable called count is used to control which rows the data is stored into in the test_data matrix
	test_data.at<float>(0, 1 + (5 * count)) = input.at<float>(0, 1);	//When count = 1, the data is stored in the first 5 rows.
	test_data.at<float>(0, 2 + (5 * count)) = input.at<float>(0, 2);	//When count = 2, the data is stored in the second 5 rows (row 6 to 10), and so on
	test_data.at<float>(0, 3 + (5 * count)) = input.at<float>(0, 3);
	test_data.at<float>(0, 4 + (5 * count)) = input.at<float>(0, 4);
}

void make_prediction(DataPrep getGLCM, Mat test_image, CvANN_MLP* mlp){	//Function that makes a predicition on the test_image
	Mat test_data(1, 40, CV_32FC1);						//Matrix used to store the test data features

	Mat tmp = getGLCM.GLCM_calc(test_image, 0, 1);		//Uses the DataPrep object that was passed as a parameter and calls the GLCM_calc function
	load_test_data(test_data, tmp, 0);					//to calculate the Haralick features of the test image.
	tmp = getGLCM.GLCM_calc(test_image, 45, 1);			//The Haralick features data is moved to the test_data matrix via the load_test_data function. 
	load_test_data(test_data, tmp, 1);
	tmp = getGLCM.GLCM_calc(test_image, 90, 1);
	load_test_data(test_data, tmp, 2);
	tmp = getGLCM.GLCM_calc(test_image, 135, 1);
	load_test_data(test_data, tmp, 3);
	tmp = getGLCM.GLCM_calc(test_image, 0, 2);
	load_test_data(test_data, tmp, 4);
	tmp = getGLCM.GLCM_calc(test_image, 45, 2);
	load_test_data(test_data, tmp, 5);
	tmp = getGLCM.GLCM_calc(test_image, 90, 2);
	load_test_data(test_data, tmp, 6);
	tmp = getGLCM.GLCM_calc(test_image, 135, 2);
	load_test_data(test_data, tmp, 7);

	Mat output(1, 3, CV_32FC1);					//Creates an output matrix to store the prediction probabilities.
	mlp->predict(test_data, output);			//Makes a prediction using the OpenCV CvANN_MLP object. Stores result in output matrix

	float max = output.at<float>(0, 0);			//Stores the maximum value from the output matrix.
	if (max < output.at<float>(0, 1))			//Checks all 3 values in the Matrix and stores the highest value in max.
		max = output.at<float>(0, 1);
	if (max < output.at<float>(0, 2))
		max = output.at<float>(0, 2);

	//cout << output.at<float>(0, 0) << " " << output.at<float>(0, 1) << " " << output.at<float>(0, 2) << "\n";
	if (max == output.at<float>(0, 0))			//If first value in the output matrix is the highest, the prediction produced a "Good" classification
		cout << "GOOD" << endl;
	else if (max == output.at<float>(0, 1))		//If second value in the output matrix is the highest, the prediction produced an "Empty" classification
		cout << "EMPTY" << endl;
	else
		cout << "BAD" << endl;					//If third value in the output matrix is the highest, the prediction produced a "Bad" classification
}

int main(){
	DataPrep getGLCM;							//Creates a DataPrep object in order to use functions of that class
	getGLCM.prep_data();						//Call to generate training data text files
		
	Mat layerSizes(1, 3, CV_32SC1);				//Matrix that determines no. of layers in the ANN as well as number of neurons in that layer
	layerSizes.at<int>(0) = 40;					//Input layer has 40 neurons since 40 features
	layerSizes.at<int>(1) = 43;					//10 hidden neurons
	layerSizes.at<int>(2) = 3;					//3 classes to identify to so 3 neurons in output layer
	
	CvANN_MLP mlp;												//Object for the ANN
	mlp.create(layerSizes, CvANN_MLP::SIGMOID_SYM , 0, 0);		//Creating the ANN with specified layers and SIGMOID activation function

	Mat training_data = data_to_matrix();		//Sends training data from text files to the training_data matrix

	Mat trainClasses;											//Matrix for storing the training data labels
	trainClasses.create(training_data.rows, 3, CV_32FC1);		//Rows represent the training data sample and the colums represent their respective classifications.
																//Row 0 is "Good". Row 1 is "Empty". Row 2 is "Bad"
	
	for (int i = 0; i < 35; i++)
	{
		trainClasses.at<float>(i, 0) = 1;			//First 35 samples are "Good" classifications
		trainClasses.at<float>(i, 1) = 0;
		trainClasses.at<float>(i, 2) = 0;
	}
	for (int i = 35; i < 72; i++)
	{
		trainClasses.at<float>(i, 0) = 0;
		trainClasses.at<float>(i, 1) = 1;			//The next 37 samples are "Empty" classifications
		trainClasses.at<float>(i, 2) = 0;
	}
	for (int i = 72; i < 108; i++)
	{
		trainClasses.at<float>(i, 0) = 0;
		trainClasses.at<float>(i, 1) = 0;
		trainClasses.at<float>(i, 2) = 1;			//The last 36 samples are "Bad" classifications
	}


	Mat weights(1, training_data.rows, CV_32FC1);
	for (int i = 0; i < weights.cols; i++)
		weights.at<float>(0, i) = (float)1.0;
	
	mlp.train(training_data, trainClasses, weights);	//train the ANN with training data, their respective labels and weights
	
//---------------------------------------------------TESTING---------------------------------------------------------------------------
	
	cv::String folder = "training-and-test-files/good_test";
	cout << "\nGOOD test: " << endl;
	vector<cv::String> good_filenames;					//Vector stores multiple filenames as strings
	glob(folder, good_filenames);						//Inputs all the filenames from the folder specified, into the good_filenames vector  
	
	for (int i = 0; i < good_filenames.size(); i++)
	{
		cout << good_filenames[i].substr(good_filenames[i].find_last_of("/\\") + 1) << " Result : ";
		Mat image_load = imread(good_filenames[i]);		//Create a matrix of the image from the filename stored in the vector
		make_prediction(getGLCM, image_load, &mlp);		//Calls make_prediction function to predict the image's classification
	}

	folder = "training-and-test-files/empty_test";
	cout << "\nEMPTY test: " << endl;
	vector<cv::String> empty_filenames;					//Similar functionality but done for the "Empty" classification
	glob(folder, empty_filenames);

	for (int i = 0; i < empty_filenames.size(); i++)
	{
		cout << empty_filenames[i].substr(empty_filenames[i].find_last_of("/\\") + 1) << " Result : ";
		Mat image_load = imread(empty_filenames[i]);
		make_prediction(getGLCM, image_load, &mlp);
	}

	folder = "training-and-test-files/bad_test";
	cout << "\nBAD test: " << endl;
	vector<cv::String> bad_filenames;					//Similar functionality but done for the "Bad" classification
	glob(folder, bad_filenames);

	for (int i = 0; i < bad_filenames.size(); i++)
	{
		cout << bad_filenames[i].substr(bad_filenames[i].find_last_of("/\\") + 1) << " Result : ";
		Mat image_load = imread(bad_filenames[i]);
		make_prediction(getGLCM, image_load, &mlp);
	}

	system("pause");
	return 0;
}
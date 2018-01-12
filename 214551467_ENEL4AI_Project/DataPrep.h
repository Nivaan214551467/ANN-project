/******************************************************************************************************
			NIVAAN KRISHUNDUTT
			214551467
			ENEL4AI
			Classification using Neural Network

			DataPrep.h
			- Contains functions to calculate GLCM and prepare training data

*******************************************************************************************************/

#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\core.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <iomanip>

using namespace std;
using namespace cv;

class DataPrep{
public:
	DataPrep();
	~DataPrep();
	Mat GLCM_calc(cv::Mat image, int a, int d);
	void print_to_file(ofstream& outfile, Mat data_vec);
	void training_data(cv::String folder, cv::String textname);
	void prep_data();
};


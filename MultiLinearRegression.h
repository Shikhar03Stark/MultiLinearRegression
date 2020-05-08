#pragma once
#include <armadillo>
#include <string>
#include <plplot/plstream.h>
using namespace arma;
class MultiLinearRegression {
private:
	Mat<double> dataset;
	Mat<double> X;
	Mat<double> Y;

	Mat<double> Xtrain;
	Mat<double> Ytrain;

	Mat<double> Xtest;
	Mat<double> Ytest;

	Mat<double> weight;

	bool isDataLoaded;
	bool isDataSplit;
	bool isDataFit;

	int n_data;
	int n_feature;

	double lbound;
	double ubound;

	double error(Mat<double> &X, Mat<double> &Y, Mat<double> &weight, int col);

public:

	//load data in dataset;
	bool loadData(std::string path);
	bool loadData(Mat<double> &D);

	//split into Training and Test data with fraction of training data to total data, default 0.7
	bool splitData(double fraction = 0.7);

	//normalize X , (0-1 Default)
	bool normalize(double l = 0, double u = 1);

	//fit data in Model, type = 0 training data (default), 1 test data set
	Mat<double>& fit(int type = 0, double lr = 0.1);

	//Error checking, MSE, type = 0 train data, 1 test data
	Mat<double>& fitError(int type = 0);

	//predict test data, type = 0 test data (default), -1 training data, 1 whole data set
	Mat<double>& predict(int type = 0);
	Mat<double>& predict(Mat<double>& input, double l = 0, double r = 1);
	Mat<double>& predict(std::string path, double l = 0, double r = 1);

	//plot data, type = 0 training data (default), 1 test data, 2 whole data
	//-1 test data with Prediction, -2 training data with Prediction, -3 whole data with Prediction
	void plotModel(int col, const char* xlabel, const char* ylabel, const char* title, int type=0);

	//Plot data inputted from Matrix or Path wit l-u norm
	void plot(Mat<double>& input, const char* ylabel, const char* title, double l = 0, double u = 1);
	void plot(std::string path, const char* ylabel, const char* title, double l = 0, double u = 1);

	//cstor
	MultiLinearRegression();

	MultiLinearRegression(std::string path);

	MultiLinearRegression(Mat<double> &data);

};
/*
Mat<double> MultiLinearRegression::dataset;
Mat<double> MultiLinearRegression::X;
Mat<double> MultiLinearRegression::Y;

Mat<double> MultiLinearRegression::Xtrain;
Mat<double> MultiLinearRegression::Ytrain;

Mat<double> MultiLinearRegression::Xtest;
Mat<double> MultiLinearRegression::Ytest;

Mat<double> MultiLinearRegression::weight;*/


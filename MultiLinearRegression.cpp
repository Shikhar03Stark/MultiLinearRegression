#include "MultiLinearRegression.h"
#include <iostream>

//loading data using built-in arma load function
bool MultiLinearRegression::loadData(std::string path) {
	std::cout << "Loading Dataset...\n";
	bool isLoaded = dataset.load(path);
	if (!isLoaded) {	//could not load data
		std::cout << "Path not Found...\n";
		std::cout << dataset << endl;
		return false;
	}
	else { //set defaults
		if (dataset.n_cols < 2) {
			std::cout << "Data Must Contain at least 2 Columns...\n";
			n_data = 0;
			n_feature = 0;

			lbound = 0;
			ubound = 1;

			isDataLoaded = false;
			isDataSplit = false;
			isDataFit = false;

			dataset.reset();
			return false;
		}
		std::cout << "Data Loaded Successfully...\n";
		Y = dataset.col(dataset.n_cols - 1);
		X = dataset;
		X.resize(dataset.n_rows, dataset.n_cols - 1);

		n_data = dataset.n_rows;
		n_feature = dataset.n_cols - 1;

		lbound = 0;
		ubound = 1;

		isDataLoaded = true;
		isDataSplit = false;
		isDataFit = false;

		return true;
	}
	
}

//loading data from Arma Matrix
bool MultiLinearRegression::loadData(Mat<double> &D) {
	if (D.n_cols < 2) {
		std::cout << "Data Must Contain at least 2 columns...\n";
		n_data = 0;
		n_feature = 0;

		lbound = 0;
		ubound = 1;

		isDataLoaded = false;
		isDataSplit = false;
		isDataFit = false;
		return false;
	}
	else {
		dataset = D;
		//std::cout << dataset << endl;
		Y = dataset.col(dataset.n_cols - 1);
		X = dataset;
		X.resize(dataset.n_rows, dataset.n_cols - 1);

		n_data = dataset.n_rows;
		n_feature = dataset.n_cols - 1;

		lbound = 0;
		ubound = 1;

		isDataLoaded = true;
		isDataSplit = false;
		isDataFit = false;
		return true;
	}
}

//normalize X (recommended)
bool MultiLinearRegression::normalize(double l, double u) {
	//std::cout << Xtrain << endl << Xtest << endl;
	lbound = l;
	ubound = u;
	if (isDataLoaded) {
		if (!isDataSplit) {
			std::cout << "Normalizing Coloumn...\n";
			double colMin = double(INT_MAX);
			double colMax = double(INT_MIN);
			for (int j = 0; j < n_feature; j++) {
				colMin = double(INT_MAX);
				colMax = double(INT_MIN);
				for (int i = 0; i < n_data; i++) {
					if (X(i, j) < colMin) {
						colMin = X(i, j);
					}
					if (X(i, j) > colMax) {
						colMax = X(i, j);
					}
				}
				if (colMax == colMin) {
					X.col(j).fill(u);
				}
				else {
					X.col(j) = l + (X.col(j) - colMin) * (u - l) / (colMax - colMin);
				}
			}
			std::cout << "Columns Normalized...\n";
			return true;
		}
		else {
			std::cout << "Normalizing Coloums...\n";
			//training data
			double colMin = double(INT_MAX);
			double colMax = double(INT_MIN);
			for (int j = 1; j <= n_feature; j++) {
				colMin = double(INT_MAX);
				colMax = double(INT_MIN);
				for (int i = 0; i < Xtrain.n_rows; i++) {
					if (Xtrain(i, j) < colMin) {
						colMin = Xtrain(i, j);
					}
					if (Xtrain(i, j) > colMax) {
						colMax = Xtrain(i, j);
					}
				}
				if (colMax == colMin) {
					Xtrain.col(j).fill(u);
				}
				else {
					Xtrain.col(j) = l + (Xtrain.col(j) - colMin) * (u - l) / (colMax - colMin);
				}
			}
			//test data
			for (int j = 1; j <= n_feature; j++) {
				colMin = double(INT_MAX);
				colMax = double(INT_MIN);
				for (int i = 0; i < Xtest.n_rows; i++) {
					if (Xtest(i, j) < colMin) {
						colMin = Xtest(i, j);
					}
					if (Xtest(i, j) > colMax) {
						colMax = Xtest(i, j);
					}
				}
				if (colMin == colMax) {
					Xtest.col(j).fill(u);
				}
				else {
					Xtest.col(j) = l + (Xtest.col(j) - colMin) * (u - l) / (colMax - colMin);
				}
			}
			std::cout << "Columns Noramalized...\n";
			return true;
		}
	}
	else {
		std::cout << "Data is not Loaded in Object...\n";
		return false;
	}
}


//spliting data into training and test data;
bool MultiLinearRegression::splitData(double fraction) {
	if (isDataLoaded) {
		if (fraction > 1 || fraction < 0) {
			std::cout << "Splitting fraction must be between 0 to 1...\n";
			return false;
		}
		int trainl = 0;
		int trainu = int(n_data * fraction);

		int testl = int(n_data * fraction);
		int testu = n_data;

		//local complete matrix, which may be used for random insertion of rows in test and train
		Mat<double> complete = Mat<double>(n_data, n_feature+1, fill::ones);
		//may implement an algo for random insertion of rows to test and train X
		//std::cout << this->X << endl;
		for (int i = 0; i < n_feature; i++) {
			//std::cout << X.col(i) << endl;
			complete.col(i+1) = X.col(i);
		}
		//std::cout << complete << endl;
		this->Xtrain = Mat<double>(trainu - trainl, n_feature+1, fill::zeros);
		this->Ytrain = Mat<double>(trainu - trainl, 1, fill::zeros);

		this->Xtest = Mat<double>(testu - testl, n_feature + 1, fill::zeros);
		this->Ytest = Mat<double>(testu - testl, 1, fill::zeros);

		for (int i = trainl; i < trainu; i++) {
			Xtrain.row(i-trainl) = complete.row(i);
			Ytrain.row(i-trainl) = Y.row(i);
		}

		for (int i = testl; i < testu; i++) {
			Xtest.row(i-testl) = complete.row(i);
			Ytest.row(i-testl) = Y.row(i);
		}
		//std::cout << "Spliting Data\n";
		//std::cout << Xtrain << Ytrain << endl;
		//std::cout << Xtest << Ytest << endl;
		isDataSplit = true;
		return true;
	}
	else {
		std::cout << "Data is not Loaded in Object...\n";
		return false;
	}
}

//private error calculating function
double MultiLinearRegression::error(Mat<double> &X, Mat<double> &Y, Mat<double> &weight, int col) {
	int size = X.n_rows;
	int feature = X.n_cols - 1;

	//std::cout << X * weight << endl;
	Mat<double> E = (X * weight) - Y;

	if (col == 0) {
		double sum = 0;
		for (double e : E) {
			sum += e;
		}
		return sum;
	}
	else {
		double sum = 0;
		for (int i = 0; i < size; i++) {
			sum += E(i, 0) * X(i, col);
		}
		return sum;
	}
}


//fitting data by Multi Variable Linear Regression using Gradien Descent
Mat<double>& MultiLinearRegression::fit(int type, double lr) {
	if (isDataLoaded) {
		if (isDataSplit) {
			weight = Mat<double>(n_feature + 1, 1, fill::zeros);

			if (type == 0) { //training set
				//cout << Xtrain << endl;
				std::cout << "Fitting Process Started...\n";
				int size = Xtrain.n_rows;
				int itr = (Xtrain.n_rows == 0 ? 0 : 10000 / Xtrain.n_rows + 1);
				for (int t = 0; t < size; t++) {
					for (int r = 0; r <= itr; r++) {
						for (int i = 0; i <= n_feature; i++) {
							if (i == 0) {
								weight(0, 0) = weight(0, 0) - lr * error(Xtrain, Ytrain, weight, 0) / size;
							}
							else {
								weight(i, 0) = weight(i, 0) - lr * error(Xtrain, Ytrain, weight, i) / size;
							}
						}
					}
				}

				std::cout << "Fitting over Training set Complete...\n";
				isDataFit = true;
				return weight;
			}

			else if (type == 1) { //testing set

				int size = Xtest.n_rows;
				int itr = (Xtest.n_rows == 0 ? 0 : 100 / Xtest.n_rows + 1);
				for (int r = 0; r < itr; r++) {
					for (int t = 0; t < size; t++) {
						for (int i = 0; i <= n_feature; i++) {
							if (i == 0) {
								weight(0, 0) = weight(0, 0) - lr * error(Xtest, Ytest, weight, 0) / size;
							}
							else {
								weight(i, 0) = weight(i, 0) - lr * error(Xtest, Ytest, weight, i) / size;
							}
						}
					}
				}

				std::cout << "Fitting over Testing set Complete...\n";
				isDataFit = true;
				return weight;
			}
			else {
				std::cout << "Invalid Fitting type...\n";
				return *(new Mat<double>(0, 0));
			}
		}
		else {
			std::cout << "Split the data into training set and Test set...\n";
			return *(new Mat<double>(0, 0));
		}
	}
	else {
		std::cout << "Data is not Loaded in Object...\n";
		return *(new Mat<double>(0, 0));
	}
}

Mat<double>& MultiLinearRegression::fitError(int type) {
	if (isDataLoaded) {
		if (isDataSplit) {
			if (isDataFit) {
				if (type == 0) { //Error over Training Set
					Mat<double>* FE = new Mat<double>(Xtrain.n_rows, 1, fill::zeros);
					*FE = (Xtrain * weight) - Ytrain;
					return (*FE);
				}
				else if (type == 1) { //Error over Testing Set
					Mat<double>* FE = new Mat<double>(Xtest.n_rows, 1, fill::zeros);
					*FE = (Xtest * weight) - Ytest;
					return (*FE);
				}
				else {
					std::cout << "FitError type is not defined...\n";
					return *(new Mat<double>(0, 0));
				}
			}
			else {
				std::cout << "Model is not Fitted on any Dataset...\n";
				return *(new Mat<double>(0, 0));
			}
		}
		else {
			std::cout << "Data is not Splited into Training Data and Test Data...\n";
			return *(new Mat<double>(0, 0));
		}
	}
	else {
		std::cout << "Data is not Loaded in Object...\n";
		return *(new Mat<double>(0, 0));
	}
}

//predict test data over fitted weights
Mat<double>& MultiLinearRegression::predict(int type) {
	if (isDataLoaded) {
		if (isDataSplit) {
			if (isDataFit) {
				if (type == 0) { //prediction on Test data
					Mat<double>* P = new Mat<double>(Xtest.n_rows, 1, fill::zeros);
					*P = Xtest * weight;
					return (*P);

				}
				else if (type == 1) { //prediction on training Data
					Mat<double>* P = new Mat<double>(Xtrain.n_rows, 1, fill::zeros);
					*P = Xtrain * weight;
					return (*P);
				}
				else{
					std::cout << "Predict type is not defined...\n";
					return *(new Mat<double>(0, 0));
				}
			}
			else {
				std::cout << "Model is not Fitted on any Dataset...\n";
				return *(new Mat<double>(0, 0));
			}
		}
		else {
			std::cout << "Data is not Splited into Training Data and Test Data...\n";
			return *(new Mat<double>(0, 0));
		}
	}
	else {
		std::cout << "Data is not Loaded in Object..\n";
		return *(new Mat<double>(0, 0));
	}
}

//ploting the model Attributes
void MultiLinearRegression::plotModel(int col, const char* xlabel, const char* ylabel, const char* title, int type) {
	if (isDataLoaded) {
		if (type == 0) { //plot point training data
			const int ticks = 1000;
			double x[ticks], y[ticks];
			double xmin, xmax, ymin, ymax;
			xmax = ymax = double(INT_MIN);
			xmin = ymin = double(INT_MAX);

			int points = (Ytrain.n_rows > 1000 ? 1000 : Ytrain.n_rows);
			for (int i = 0; i < points; i++) {
				if (Xtrain(i, col + 1) < xmin) {
					xmin = Xtrain(i, col + 1);
				}
				if (Xtrain(i, col + 1) > xmax) {
					xmax = Xtrain(i, col + 1);
				}
			}
			for (int i = 0; i < points; i++) {
				if (Ytrain(i, 0) < ymin) {
					ymin = Ytrain(i, 0);
				}
				if (Ytrain(i, 0) > ymax) {
					ymax = Ytrain(i, 0);
				}
			}

			for (int i = 0; i < points; i++) {
				x[i] = Xtrain(i, col + 1);
				y[i] = Ytrain(i, 0);
			}

			plstream* pls = new plstream;

			pls->init();
			pls->col0(3);
			pls->env(xmin, xmax, ymin, ymax, 0, 0);
			pls->lab(xlabel, ylabel, title);
			pls->col0(15);
			pls->poin(points, x, y, '+');

			delete pls;
			return;
		}
		if (isDataSplit) {
			if (type == 1) { //plot point test data
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double xmin, xmax, ymin, ymax;
				xmax = ymax = double(INT_MIN);
				xmin = ymin = double(INT_MAX);

				int points = (Ytest.n_rows > 1000 ? 1000 : Ytest.n_rows);
				for (int i = 0; i < points; i++) {
					if (Xtest(i, col + 1) < xmin) {
						xmin = Xtest(i, col + 1);
					}
					if (Xtest(i, col + 1) > xmax) {
						xmax = Xtest(i, col + 1);
					}
				}
				for (int i = 0; i < points; i++) {
					if (Ytest(i, 0) < ymin) {
						ymin = Ytest(i, 0);
					}
					if (Ytest(i, 0) > ymax) {
						ymax = Ytest(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = Xtest(i, col + 1);
					y[i] = Ytest(i, 0);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab(xlabel, ylabel, title);
				pls->col0(15);
				pls->poin(points, x, y, '+');

				delete pls;
				return;
			}
			else if (type == 2) { //plot point full data
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double xmin, xmax, ymin, ymax;
				xmax = ymax = double(INT_MIN);
				xmin = ymin = double(INT_MAX);

				int points = (Y.n_rows > 1000 ? 1000 : Y.n_rows);
				for (int i = 0; i < points; i++) {
					if (X(i, col) < xmin) {
						xmin = X(i, col);
					}
					if (X(i, col + 1) > xmax) {
						xmax = X(i, col);
					}
				}
				for (int i = 0; i < points; i++) {
					if (Y(i, 0) < ymin) {
						ymin = Y(i, 0);
					}
					if (Y(i, 0) > ymax) {
						ymax = Y(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = X(i, col);
					y[i] = Y(i, 0);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab(xlabel, ylabel, title);
				pls->col0(15);
				pls->poin(points, x, y, '+');

				delete pls;
				return;
			}
		}
		else {
			std::cout << "Data is not Splitted...\n";
			return;
		}
		if (isDataFit) {
			if (type == -2) { //plot point training data with Prediction
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double  yl[ticks];
				double xmin, xmax, ymin, ymax;
				xmax = ymax = double(INT_MIN);
				xmin = ymin = double(INT_MAX);

				int points = (Ytrain.n_rows > 1000 ? 1000 : Ytrain.n_rows);
				Mat<double> P = Xtrain * weight;

				for (int i = 0; i < points; i++) {
					if (Xtrain(i, col + 1) < xmin) {
						xmin = Xtrain(i, col + 1);
					}
					if (Xtrain(i, col + 1) > xmax) {
						xmax = Xtrain(i, col + 1);
					}
				}
				for (int i = 0; i < points; i++) {
					if (Ytrain(i, 0) < ymin) {
						ymin = Ytrain(i, 0);
					}
					if (P(i, 0) < ymin) {
						ymin = P(i, 0);
					}
					if (Ytrain(i, 0) > ymax) {
						ymax = Ytrain(i, 0);
					}
					if (P(i, 0) > ymax) {
						ymax = P(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = Xtrain(i, col + 1);
					y[i] = Ytrain(i, 0);
					yl[i] = P(i, 0);//weight(0, 0) + weight(col + 1) * Xtrain(i, col + 1);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab(xlabel, ylabel, title);

				pls->col0(15);
				pls->poin(points, x, y, '*');

				pls->col0(12);
				pls->poin(points, x, yl, 'X');

				delete pls;
				return;
			}
			else if (type == -1) { //plot point test data with Prediction
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double  yl[ticks];
				double xmin, xmax, ymin, ymax;
				xmax = ymax = double(INT_MIN);
				xmin = ymin = double(INT_MAX);

				int points = (Ytest.n_rows > 1000 ? 1000 : Ytest.n_rows);
				Mat<double> P = Xtest * weight;

				for (int i = 0; i < points; i++) {
					if (Xtest(i, col + 1) < xmin) {
						xmin = Xtest(i, col + 1);
					}
					if (Xtest(i, col + 1) > xmax) {
						xmax = Xtest(i, col + 1);
					}
				}
				for (int i = 0; i < points; i++) {
					if (Ytest(i, 0) < ymin) {
						ymin = Ytest(i, 0);
					}
					if (P(i, 0) < ymin) {
						ymin = P(i, 0);
					}
					if (Ytest(i, 0) > ymax) {
						ymax = Ytest(i, 0);
					}
					if (P(i, 0) > ymax) {
						ymax = P(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = Xtest(i, col + 1);
					y[i] = Ytest(i, 0);
					yl[i] = P(i, 0);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab(xlabel, ylabel, title);

				pls->col0(15);
				pls->poin(points, x, y, '+');

				pls->col0(12);
				pls->poin(points, x, yl, '*');

				pls->col0(1);
				for (int i = 0; i < points; i++) {
					pls->join(x[i], y[i], x[i], yl[i]);
				}

				delete pls;
				return;
			}
			else if (type == -3) { //plot point full data with Prediction
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double  yl[ticks];
				double xmin, xmax, ymin, ymax;
				xmax = ymax = double(INT_MIN);
				xmin = ymin = double(INT_MAX);

				int points = (Y.n_rows > 1000 ? 1000 : Y.n_rows);
				Mat<double> complete = Mat<double>(Y.n_rows, 1, fill::ones);
				complete.insert_cols(0, X);
				Mat<double> P = complete * weight;

				for (int i = 0; i < points; i++) {
					if (X(i, col + 1) < xmin) {
						xmin = X(i, col + 1);
					}
					if (X(i, col + 1) > xmax) {
						xmax = X(i, col + 1);
					}
				}
				for (int i = 0; i < points; i++) {
					if (Y(i, 0) < ymin) {
						ymin = Y(i, 0);
					}
					if (P(i, 0) < ymin) {
						ymin = P(i, 0);
					}
					if (Y(i, 0) > ymax) {
						ymax = Y(i, 0);
					}
					if (P(i, 0) > ymax) {
						ymax = P(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = X(i, col + 1);
					y[i] = Y(i, 0);
					yl[i] = P(i, 0);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab(xlabel, ylabel, title);

				pls->col0(15);
				pls->poin(points, x, y, '+');

				pls->col0(12);
				pls->poin(points, x, yl, '*');

				pls->col0(1);
				for (int i = 0; i < points; i++) {
					pls->join(x[i], y[i], x[i], yl[i]);
				}

				delete pls;
				return;
			}
		}
		else {
		std::cout << "Model is not Fitted over Training Set...\n";
		}
	}
	else{
		std::cout << "Data is not Loaded in Object...\n";
	}
}

//plot data inputted from Matrix
void MultiLinearRegression::plot(Mat<double>& input, const char* ylabel, const char* title, double l, double u) {
	if (isDataLoaded) {
		if (isDataFit) {
			if (input.n_cols != n_feature) {
				std::cout << "Matrix doesn't match with Number of Features...\n";
				return;
			}
			else {
				//normalize
				l = lbound;
				u = ubound;
				for (int j = 0; j < n_feature; j++) {
					double min = double(INT_MAX);
					double max = double(INT_MIN);
					for (int i = 0; i < input.n_rows; i++) {
						if (input(i, j) < min) {
							min = input(i, j);
						}
						if (input(i, j) > max) {
							max = input(i, j);
						}
					}
					if (min == max) {
						input.col(j).fill(u);
					}
					else {
						input.col(j) = l + (input.col(j) - min) * (u - l) / (max - min);
					}
				}

				//convert to X
				Mat<double> M = Mat<double>(input.n_rows, 1, fill::ones);
				M.insert_cols(0, input);

				//M is ready to be plotted;
				const int ticks = 1000;
				double x[ticks], y[ticks];
				double xmin, xmax, ymin, ymax;
				xmin = ymin = double(INT_MAX);
				xmax = ymax = double(INT_MIN);

				int points = (M.n_rows > 1000 ? 1000 : M.n_rows);

				Mat<double> H = M * weight;

				xmin = 0;
				xmax = points + 1;

				for (int i = 0; i < points; i++) {
					if (H(i, 0) < ymin) {
						ymin = H(i, 0);
					}
					if (H(i, 0) > ymax) {
						ymax = H(i, 0);
					}
				}

				for (int i = 0; i < points; i++) {
					x[i] = i + 1;
					y[i] = H(i, 0);
				}

				plstream* pls = new plstream;

				pls->init();
				pls->col0(3);
				pls->env(xmin, xmax, ymin, ymax, 0, 0);
				pls->lab("Data Points", ylabel, title);

				pls->col0(12);
				pls->poin(points, x, y, '+');

				pls->col0(2);
				for (int i = 0; i < points; i++) {
					pls->join(x[i], y[i], xmin, y[i]);
				}
				delete pls;
				return;
			}
		}
		else {
			std::cout << "Model is not Fitted with Training Set...\n";
		}
	}
	else {
		std::cout << "No Data Loaded to Train the Model...\n";
	}
}

//plot data inputted from path
void MultiLinearRegression::plot(std::string path, const char* ylabel, const char* title, double l, double u) {
	if (isDataLoaded) {
		if (isDataFit) {
			Mat<double> input;
			bool isLoaded = input.load(path);
			if (!isLoaded) {
				std::cout << "Path can't be Found...\n";
				return;
			}
			else {
				if (input.n_cols != n_feature) {
					std::cout << "Matrix doesn't match with Number of Features...\n";
					return;
				}
				else {
					//normalize
					l = lbound;
					u = ubound;
					for (int j = 0; j < n_feature; j++) {
						double min = double(INT_MAX);
						double max = double(INT_MIN);
						for (int i = 0; i < input.n_rows; i++) {
							if (input(i, j) < min) {
								min = input(i, j);
							}
							if (input(i, j) > max) {
								max = input(i, j);
							}
						}
						if (min == max) {
							input.col(j).fill(u);
						}
						else {
							input.col(j) = l + (input.col(j) - min) * (u - l) / (max - min);
						}
					}

					//convert to X
					Mat<double> M = Mat<double>(input.n_rows, 1, fill::ones);
					M.insert_cols(0, input);

					//M is ready to be plotted;
					const int ticks = 1000;
					double x[ticks], y[ticks];
					double xmin, xmax, ymin, ymax;
					xmin = ymin = double(INT_MAX);
					xmax = ymax = double(INT_MIN);

					int points = (M.n_rows > 1000 ? 1000 : M.n_rows);

					Mat<double> H = M * weight;

					xmin = 0;
					xmax = points + 1;

					for (int i = 0; i < points; i++) {
						if (H(i, 0) < ymin) {
							ymin = H(i, 0);
						}
						if (H(i, 0) > ymax) {
							ymax = H(i, 0);
						}
					}

					for (int i = 0; i < points; i++) {
						x[i] = i + 1;
						y[i] = H(i, 0);
					}

					plstream* pls = new plstream;

					pls->init();
					pls->col0(3);
					pls->env(xmin, xmax, ymin, ymax, 0, 0);
					pls->lab("Data Points", ylabel, title);

					pls->col0(12);
					pls->poin(points, x, y, '+');

					pls->col0(10);
					for (int i = 0; i < points; i++) {
						pls->join(x[i], y[i], xmin, y[i]);
					}
					delete pls;
					return;
				}
			}
		}
		else {
			std::cout << "Model is not Fitted with Training Set...\n";
		}
	}
	else {
		std::cout << "No Data Loaded to Train the Model...\n";
	}
}



//ctor
MultiLinearRegression::MultiLinearRegression() {
	isDataLoaded = false;
	isDataSplit = false;
	isDataFit = false;

	n_data = 0;
	n_feature = 0;

	lbound = 0;
	ubound = 1;
}

MultiLinearRegression::MultiLinearRegression(std::string path) {
	if (!loadData(path)) {
		std::cout << "Could not load the Dataset File\n";
		isDataLoaded = false;
		isDataSplit = false;
		isDataFit = false;

		n_data = 0;
		n_feature = 0;

		lbound = 0;
		ubound = 1;
	}
}

MultiLinearRegression::MultiLinearRegression(Mat<double> &data) {
	if (data.n_cols > 1) {
		dataset = Mat<double>(data);
		Y = dataset.col(dataset.n_cols - 1);
		X = dataset;
		X.resize(dataset.n_rows, dataset.n_cols - 1);

		n_data = dataset.n_rows;
		n_feature = dataset.n_cols - 1;

		lbound = 0;
		ubound = 1;

		isDataLoaded = true;
		isDataSplit = false;
		isDataFit = false;
	}
	else {
		isDataLoaded = false;
		isDataSplit = false;
		isDataFit = false;

		n_data = 0;
		n_feature = 0;

		lbound = 0;
		ubound = 1;
	}
}


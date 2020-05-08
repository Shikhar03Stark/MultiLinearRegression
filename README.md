# MultiLinearRegression
C++ based Multi Variable Linear Regression with Visualizations and in-depth tweeks;

## Prerequisite
1. To work Effectively with this Class use Visual Studio or any Suitable IDE.
*For Windows: use vcpkg with Visual Studio*
2. Install Armadillo (for Linear Algebra) and plplot for Visualization

*vcpkg command*

>vcpkg install armadillo

>vcpkg install plplot

>vcpkg integrate install  //*for user-wide integration*

3. Import the Files into Your Project Project and you are good to go

## How to Use this Class

The Class Name is MultiLinearRegression add the Header file MultiLinearRegression.h in your file

### 1. Constructors

*for loading data into the object through file path*

>MultiLinearRegression model(std::string);

*for loading data into the object through a arma Matrix*

>MultiLinearRegression model(arma::Mat<double>);
  
### 2. load function

*for loading data into object*
>model.load(std::string);

>model.load(arma::Mat<double>);
  
### 3. Spliting Data into Training and Test Data

*By default 70% of the data is used as training data and rest as Test data*
>model.split();

>model.split(double)

### 4. Normalize the Data (recommended)

*By delault the X data is normalized from 0 to 1*
>model.normalize();

>model.normalize(double, double);

### 5. Fitting the Model into the Trainig Data

*Train the data from trainig dataset (default), with learing ration (0.1 default)*
>model.fit();

>model(int,double);

### 6. Error Checking in Hypothesis

*Checks the error of Predicted Y to actual Y from training Data (default)*
>model.fitError();

>model.fitError(int);

### 7. Predict the Test data set

*Returns the H for Test Data sets(default)*
>model.predict();

>model.predict(int);

### 8. Visualizing the Model for Each Feature

*plots a graph between column (0-indexed) vs Y, with X label, Y label, Graph Title, Type of Graph*
>model.plotModel(int, const char\*, const char\*, const char\*, int);

### 9. Visualizing the Model for Real World Test cases

*Plots a grpah between datapoint (strating from 1) vs Y, input data, Y label, Graph Title, lower and upper limit of normalization (0-1 default)
>model.plot(std::string, const char \*, const char \*, double, double);

>model.plot(arama::Mat<double>, const char \*, const char \*, double, double);

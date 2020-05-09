#include <iostream>
#include "MultiLinearRegression.h"
using namespace std;

int main() {
	//Size, Total bill, Tips
	MultiLinearRegression model("../data/tipping/primary.csv");

	model.splitData(0.8);
	model.normalize(0, 10);
	model.fit();
	model.plot("../data/tipping/test.csv", "Tips Recieved", "Predictions 100%");
	cout << model.predict("../data/tipping/test.csv") << endl;
	cin.get();
	return 0;
}
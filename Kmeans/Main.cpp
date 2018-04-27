
#include "Kmeans.h"
#include <iostream>
using namespace std;


int main(){
	int catNum;
	cin>>catNum;
	//vector<vector<float> > dataset;
	/*int seed = (unsigned)time( NULL );
	srand(seed);
	for (int i = 0; i < 10; i++)
	{

	vector<float> temp;
	for (int j = 0; j < 3; j++)
	{

	temp.push_back(rand()%100);
	}
	dataset.push_back(temp);
	}*/
	Kmeans<float> kmeans(catNum);
	if (kmeans.readData("D:/CodeFolder/gitFolder/Kmeans/KmeansDataset.txt")) {
			kmeans.trainKmeans();
			kmeans.resultShow();
	}
	//kmeans.initClusterCenter();
	
	system("PAUSE");
	return 0;
}
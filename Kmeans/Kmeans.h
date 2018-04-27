#ifndef KMEANS_HPP
#define KMEANS_HPP


#include <iostream>  
#include <vector>  
#include <cstdlib>  
#include <fstream>  
#include <sstream> 
#include <string>  
#include <ctime>  //for srand  
using namespace std;  
 
template<typename T>
class Kmeans
{
	struct ClusterNode{
		vector< T > centroid;
		vector<int> samplesIdx;
	} *clusterCenter;						 //类别中心,以及属于该类别的所有样本下标
	vector< vector< T > > dataSet;			 //数据集
	string frileName;						 //文本地址
	int catNum;								 //类别数
public:
	Kmeans();
	Kmeans(vector<vector<T> > dataSet, int catNum);
	Kmeans(int catNum);
	bool readData(string fileName);
	void initClusterCenter();						//初始化类别中心
	void trainKmeans();								//训练

	void dataShow(vector<T> dataPoints);			//显示一个样本
	void resultShow();								//显示分类结果
	~Kmeans();
};


template<typename T>
float clacEucDist(vector<T> point1, vector<T> center){
	float dist = 0;
	for(int i = 0; i < point1.size(); i++){
		dist += (point1[i] - center[i])*(point1[i] - center[i]);
	}
	dist = sqrt(dist);
	return dist;
}

template<typename T>
Kmeans<T>::Kmeans()
{
}


template<typename T>
Kmeans<T>::Kmeans(vector<vector<T> > dataSet, int catNum)
{
	this->dataSet = dataSet;
	this->catNum = catNum;
}


template<typename T>
Kmeans<T>::Kmeans(int catNum)
{
	this->catNum = catNum;
}


template<typename T>
bool Kmeans<T>::readData(string fileName)
{
	ifstream file(fileName);
	if(!file)
	{
		cout<<fileName<<" OPEN FAILE!!!!"<<endl;
		return false;
	}

	
	//this->p = new myPoint;
	//vector<vector<float> > points;
	
	string temp;
	while(getline(file, temp)){
		stringstream stringin(temp); 
		string t;
		vector<string> line;
		vector< T > pp;
		while(stringin>>t){
			line.push_back(t);
		}
		int i;
#if 1
		//因为这个数据集里有标签，所以减1
		for (i = 0; i<line.size() ; i++)
		{
			/*char* cov;
			strcpy(cov,line[i]);*/
			pp.push_back(atof(line[i].data()));
		}
#endif
#if 0
		//因为这个数据集里有标签，所以减1
		for(i=0; i<line.size()-1; i++)
		{
			/*char* cov;
			strcpy(cov,line[i]);*/
			pp.push_back(atof(line[i].data()));
		}
#endif
		
		this->dataSet.push_back(pp);
	}

	if(this->dataSet.empty())
		cout<<"数据为空！"<<endl;
	//cout<<this->dataSet.size()<<endl;

	file.close();
	return true;
}


template<typename T>
void Kmeans<T>::initClusterCenter()
{
	this->clusterCenter = new ClusterNode[catNum];
	const int row_num = this->dataSet.size();
	const int col_num = this->dataSet[0].size();
	int k = this->catNum;
	/*初始化聚类中心*/
	//clusterCenter = new ClusterNode[k];
	//vector<ClusterNode> clusters(k);
	int seed = ((unsigned)time(NULL)); 
	srand(seed);
	for (int i = 0; i < k; i++)
	{
		
		int c = rand() % row_num;
		clusterCenter[i].centroid = this->dataSet[c];
		//seed = rand();
	}

	/*for (int i = 0; i < catNum; i++)
	{

		for (int j = 0; j < clusterCenter[i].centroid.size(); j++)
		{
			cout<<clusterCenter[i].centroid[j]<<" ";
		}
		cout<<endl;
	}*/

	//return clusterCenter;
}


template<typename T>
void Kmeans<T>::trainKmeans()
{
	this->initClusterCenter();
	vector<vector<T> > posCenter;
	float changed = 0;
	int n = 0;
	while(n<=100){
		cout<<"第"<<++n<<"次迭代"<<endl;
		for (int i = 0; i < catNum; i++)
		{
			clusterCenter[i].samplesIdx.clear();
		}
		//计算每个样本到每个中心的距离，换分为较小的一类。
		for(int i = 0; i < dataSet.size(); i++){
			float min = 100000;
			int idx = -1;
			for(int j = 0; j < this->catNum; j++){
				float dist = clacEucDist(clusterCenter[j].centroid, dataSet[i]);
				if(min > dist){
					min = dist;
					idx = j;
				}
			}
			clusterCenter[idx].samplesIdx.push_back(i);
		}

		//统计每一类的所有样本到中心距离的均值，并据此更改类别中心
		for(int i = 0; i < catNum; i++){
			vector<float> temp(dataSet[0].size(), 0.0);
			for(int j = 0; j < clusterCenter[i].samplesIdx.size(); j++){
				for (int e = 0; e < dataSet[0].size(); e++)
				{
					temp[e] += dataSet[clusterCenter[i].samplesIdx[j]][e];
					if (j == clusterCenter[i].samplesIdx.size() - 1)
					{
						clusterCenter[i].centroid[e] = temp[e]/clusterCenter[i].samplesIdx.size();
					}
				}
			}
			
		
		}

		if(n == 1){
			for(int i = 0; i < catNum; i++){
				posCenter.push_back(clusterCenter[i].centroid);
				//changed += clacEucDist(clusterCenter[i].centroid, posCenter[i]);
			}
			cout<<endl;
			changed = 0;
		}else{

			changed = 0;
			for(int i = 0; i < catNum; i++){
				changed += clacEucDist(clusterCenter[i].centroid, posCenter[i]);
				posCenter[i].clear();
				//cout<<clusterCenter[i].centroid[0]<<" ";
			}
			cout<<endl;
			for(int i = 0; i < catNum; i++){
				posCenter[i] = clusterCenter[i].centroid;
				
			}
			changed = changed / catNum;
			if(changed < 0.01)
				break;
		}
		
	}
}



template<typename T>
void Kmeans<T>::dataShow(vector<T> dataPoints)
{
	for (int i = 0; i < dataPoints.size(); i++)
	{
		
		cout<<dataPoints[i]<<" ";
		
	}
	cout<<endl;;
}


template<typename T>
void Kmeans<T>::resultShow()
{
	cout<<"聚类结果："<<endl;
	for(int i = 0; i < catNum; i++){
		cout<<"第"<<i+1<<"类样本:{"<<endl;
		for(int j = 0; j < this->clusterCenter[i].samplesIdx.size(); j++){
			dataShow(dataSet[clusterCenter[i].samplesIdx[j]]);
		}
		cout<<"}"<<endl;
	}
}



template<typename T>
Kmeans<T>::~Kmeans()
{
}
#endif

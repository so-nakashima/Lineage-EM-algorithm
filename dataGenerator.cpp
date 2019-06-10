#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <assert.h>
#include "Tree.h"
#include <chrono>

//LDS
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <functional>
#include "dataGenerator.h"

using namespace Eigen;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;


class Distribution //base class
{
public:
	virtual double sample(std::mt19937_64& generator) { return 0.0; }
private:

};

class Exponential : public Distribution //exponential distribution
{
public:
	Exponential(double lambda) {dist = std::exponential_distribution<double>(lambda); }
	virtual double sample(std::mt19937_64& generator) { return dist(generator); }
private:
	std::exponential_distribution<double> dist;
};

class Gamma : public Distribution //gamma distribution
{
public:
	Gamma(double alpha, double beta) { dist = std::gamma_distribution<double>(alpha, 1 / beta); } //alpha = k, beta  = 1 / theta; **NOTICE**STL's beta is theta in my note (or Wikipedia) 
	virtual double sample(std::mt19937_64& generator) { return dist(generator); }
private:
	std::gamma_distribution<double> dist;
};


//decleartion
void generateOneTree(int type, int typeno, int& current, int parent, double experimentTime, const std::vector<std::discrete_distribution<int>>& transit,
	const std::vector<Distribution*>& distributions, std::mt19937_64& generator, std::ofstream& file, std::ofstream& nonLeave, std::ofstream& TrueTree, std::string stringRep, std::string parentRep, int totalTime);
void writeOneTree(int type, int typeno, double experimentTime, const std::vector<std::discrete_distribution<int>>& transit,
	const std::vector<Distribution*>& distributions, std::mt19937_64& generator, std::string filename, std::string filename_nonLeave = "nonLeaveInfoTree.txt", std::string filename_true = "trueTree.txt");



void generateTrees()  //see note on the 19 June 2017
{
	const double experimentTime = 120;
	const int treeNo = 10000;

	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, 0.3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Exponential(0.05));
	distributions.push_back(new Exponential(0.025));

	//for initial node sampling
	std::discrete_distribution<int> rootType{.471371, .528629};

	//filename
	std::string namebase = ".\\Tree1_120min\\Tree1_";

	for (int i = 1000; i != treeNo; i++) {
		std::string filename = namebase + std::to_string(i) + ".dat";
		int type = rootType(generator);
		writeOneTree(type, 2, experimentTime, transit, distributions, generator, filename);
	}

}

void estimateTree1() { //see note on the 19 June 2017
	const double experimentTime = 60;
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, 0.3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Exponential(0.05));
	distributions.push_back(new Exponential(0.025));
	

	writeOneTree(1, 2, experimentTime, transit, distributions, generator, "estimateTree.dat");

	//estimation
	Tree t;
	t.load("estimateTree.dat");
	t.steadyState(experimentTime);

}


void writeOneTree(int type, int typeno, double experimentTime, const std::vector<std::discrete_distribution<int>>& transit,
	const std::vector<Distribution*>& distributions, std::mt19937_64& generator, std::string filename, std::string filename_nonLeave, std::string filename_true) {
	int current = 0;
	std::ofstream nonLeave(filename_nonLeave);
	std::ofstream TrueTree(filename_true);
	std::string stringRep = "1";

	//generate tree
	std::ofstream temp("temp.dat");
	std::ofstream file(filename);
	generateOneTree(type, typeno, current, -1, experimentTime, transit, distributions, generator, temp, nonLeave, TrueTree, stringRep, "-1", experimentTime);
	temp.close();
	file << current << " " << typeno << std::endl;

	//copying
	std::ifstream temp2("temp.dat");
	file << temp2.rdbuf() << std::flush;
	temp.clear();

	//generate 2nd Tree
}

void generateOneTree(int type, int typeno, int& current, int parent, double experimentTime, const std::vector<std::discrete_distribution<int>>& transit, 
	const std::vector<Distribution*>& distributions, std::mt19937_64& generator, std::ofstream& file, std::ofstream& nonLeave, std::ofstream& TrueTree, std::string stringRep, std::string parentRep, int totalTime) {//current = # written lines
	assert(distributions.size() == typeno);
	assert(transit.size() == typeno);
	assert(0 <= type && type < typeno);

	//get survival time
	double survivalTime = distributions[type]->sample(generator);
	if (survivalTime > experimentTime) { //leaf node case
		file << parent << " " << 1 << " " << type << " " << experimentTime << std::endl;
		current++;

		//nonLeave
		nonLeave << parent << " " << 1 << " " << -1 << " " << experimentTime << std::endl;

		//TrueTree
		TrueTree << stringRep + ", " + parentRep + ", " << 1 << ", " << totalTime - experimentTime << ", " << totalTime << "," << type + 1 << std::endl;
	}
	else { //internal node
		int id = current;
		file << parent << " " << 0 << " " << survivalTime << std::endl;


		//nonLeave
		nonLeave << parent << " " << 0 << " " << -1 << " " << survivalTime << std::endl;

		//TrueTree
		TrueTree << stringRep + ", " + parentRep + ", " << 0 << ", " << totalTime - experimentTime << ", " << totalTime - experimentTime + survivalTime << ", " << type + 1 << std::endl;

		//recursive construction
		current++;
		int leftType = transit[type](generator);
		int rightType = transit[type](generator);
		generateOneTree(leftType, typeno, current, id, experimentTime - survivalTime, transit, distributions, generator, file, nonLeave, TrueTree, stringRep + "1", stringRep, totalTime);
		generateOneTree(rightType, typeno, current, id, experimentTime - survivalTime, transit, distributions, generator, file, nonLeave, TrueTree, stringRep + "2", stringRep, totalTime);
	}
}

void generateTree2forest()  //see note on the 25 June 2017
{
	const double experimentTime = 250;
	const int treeNo = 1;

	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, .3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Gamma(20, 1)); 
	distributions.push_back(new Gamma(30, 1)); 

	//for initial node sampling
	std::discrete_distribution<int> rootType{ .476386, .523614 };  //tree2
	//std::discrete_distribution<int> rootType{ .4338, .5661 }; //tree4


	//filename
	std::string namebase = ".\\paper\\tree\\Tree2_220min_";
	std::string namebase2 = ".\\paper\\tree\\Tree2_220min_trueTree";
	std::string namebase3 = ".\\paper\\tree\\Tree2_220min_nonLeave";

	for (int i = 0; i != treeNo; i++) {
		std::string filename = namebase + std::to_string(i) + ".txt";
		std::string filename2 = ".\\paper\\tree\\Tree2_220min_trueTree" + std::to_string(i) + ".txt";
		std::string filename3 = ".\\paper\\tree\\Tree2_220min_nonLeave" + std::to_string(i) + ".txt";
		int type = rootType(generator);
		writeOneTree(type, 2, experimentTime, transit, distributions, generator, filename, filename3, filename2);
	}

}

void estimateTree2() { //see note on the 25 June 2017
	const double experimentTime = 1000;
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, .3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Gamma(5, .1));
	distributions.push_back(new Gamma(15, .1));


	writeOneTree(0, 2, experimentTime, transit, distributions, generator, "estimateTree.dat");

	//estimation
	Tree t;
	t.load("estimateTree.dat");
	t.steadyState(experimentTime);

}

void estimateTree3() { //see note on the 29 June 2017
	const double experimentTime = 500;
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, 0.3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Gamma(200, 10));
	distributions.push_back(new Gamma(300, 10));


	writeOneTree(1, 2, experimentTime, transit, distributions, generator, "estimateTree.dat");

	//estimation
	Tree t;
	t.load("estimateTree.dat");
	t.steadyState(experimentTime);

}

void generateTree3forest()  //see note on the 29 June 2017
{
	const double experimentTime = 300;
	const int treeNo = 1000;

	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed1);
	//int current = 0;

	//transit
	std::vector<std::discrete_distribution<int>> transit;
	transit.push_back(std::discrete_distribution<int>{0.7, 0.3});
	transit.push_back(std::discrete_distribution<int>{0.3, 0.7});

	//distributions
	std::vector<Distribution*> distributions;
	distributions.push_back(new Gamma(200, 10));
	distributions.push_back(new Gamma(300, 10));

	//for initial node sampling
	std::discrete_distribution<int> rootType{ .476022, .523978 };  //update later

																   //filename
	std::string namebase = ".\\Tree3_300min\\Tree3_";

	for (int i = 0; i != treeNo; i++) {
		std::string filename = namebase + std::to_string(i) + ".dat";
		int type = rootType(generator);
		writeOneTree(type, 2, experimentTime, transit, distributions, generator, filename);
	}

}

void generateOneLDSTree(
	int n,
	int m,
	int & current, // # of written cells
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV,
	int generation,
	int parent,
	const Eigen::VectorXd& currentState,
	std::ofstream& file,
	std::ofstream& trueState,
	std::function<int()> childrenNo
) {
	//assertions
	assert(A.rows() == n && A.cols() == n);
	assert(C.rows() == m && C.cols() == n);
	assert(sigmaW.rows() == n && sigmaW.cols() == n);
	assert(sigmaV.rows() == m && sigmaV.cols() == m);

	//get observation
	normal_random_variable observation(C * currentState, sigmaV);
	Vec observed = observation();

	//write result
	file << parent << " " << (generation == 1);
	for (int i = 0; i != observed.size(); i++)
		file << " " << observed[i];
	file << std::endl;
	for (int i = 0; i != currentState.size(); i++)
		trueState << currentState[i] << " ";
	trueState << std::endl;
	current++; 

	//recursion
	int id = current - 1;
	if (generation > 1) { //only bulk node
		int childrens = childrenNo();
		for (int i = 0; i != childrens; i++) {
			normal_random_variable stateTransition(A * currentState, sigmaW);
			generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation - 1, id, stateTransition(), file, trueState, childrenNo);
		}
	}
}